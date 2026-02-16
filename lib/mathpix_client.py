"""Mathpix session-based handwriting transcription at the page level.

One Mathpix session per (session_id, page) pair. Strokes accumulate within
a session (append-only). Erase/clear events invalidate the session so a
fresh one is created on the next draw.

Requires MATHPIX_APP_ID and MATHPIX_APP_KEY env vars. If missing,
transcription is silently skipped.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

from lib.database import get_pool

MATHPIX_BASE = "https://api.mathpix.com"
DEBOUNCE_SECONDS = 1.5


@dataclass
class MathpixPageSession:
    strokes_session_id: str
    app_token: str
    expires_at: datetime
    sent_stroke_log_ids: set[int] = field(default_factory=set)
    last_latex: str = ""
    last_confidence: float = 0.0


# (session_id, page) → MathpixPageSession
_sessions: dict[tuple[str, int], MathpixPageSession] = {}

# (session_id, page) → pending debounce asyncio.Task
_debounce_tasks: dict[tuple[str, int], asyncio.Task] = {}

# WebSocket refs for sending transcription updates back to iPad
_ws_by_session: dict[str, set] = {}


def _get_credentials() -> tuple[str, str]:
    app_id = os.environ.get("MATHPIX_APP_ID", "")
    app_key = os.environ.get("MATHPIX_APP_KEY", "")
    if not app_id or not app_key:
        raise RuntimeError("MATHPIX_APP_ID and MATHPIX_APP_KEY not set")
    return app_id, app_key


def register_ws(session_id: str, ws) -> None:
    _ws_by_session.setdefault(session_id, set()).add(ws)


def unregister_ws(session_id: str, ws) -> None:
    if session_id in _ws_by_session:
        _ws_by_session[session_id].discard(ws)
        if not _ws_by_session[session_id]:
            del _ws_by_session[session_id]


async def create_session() -> MathpixPageSession:
    app_id, app_key = _get_credentials()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{MATHPIX_BASE}/v3/app-tokens",
            headers={
                "app_id": app_id,
                "app_key": app_key,
                "Content-Type": "application/json",
            },
            json={"include_strokes_session_id": True},
        )
        resp.raise_for_status()
        data = resp.json()

    return MathpixPageSession(
        strokes_session_id=data["strokes_session_id"],
        app_token=data["app_token"],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=4, seconds=30),
        # 4.5 min safety margin on 5 min TTL
    )


def _is_expired(session: MathpixPageSession) -> bool:
    return datetime.now(timezone.utc) >= session.expires_at


async def get_or_create_session(
    session_id: str, page: int
) -> MathpixPageSession:
    key = (session_id, page)
    existing = _sessions.get(key)
    if existing and not _is_expired(existing):
        return existing
    session = await create_session()
    _sessions[key] = session
    return session


def invalidate_session(session_id: str, page: int) -> None:
    key = (session_id, page)
    _sessions.pop(key, None)
    task = _debounce_tasks.pop(key, None)
    if task:
        task.cancel()
    print(f"[mathpix] invalidated session ({session_id}, page={page})")


def cleanup_sessions(session_id: str) -> None:
    keys_to_remove = [k for k in _sessions if k[0] == session_id]
    for key in keys_to_remove:
        _sessions.pop(key, None)
        task = _debounce_tasks.pop(key, None)
        if task:
            task.cancel()
    if keys_to_remove:
        print(f"[mathpix] cleaned up {len(keys_to_remove)} page session(s) for {session_id}")


def reef_strokes_to_mathpix(strokes: list[dict]) -> dict:
    """Convert Reef stroke format to Mathpix strokes format.

    Reef format (per stroke):
        {"points": [{"x": float, "y": float, "t": float, ...}, ...]}

    Mathpix format:
        {"strokes": {"x": [[x1, x2, ...], ...], "y": [[y1, y2, ...], ...]}}
    """
    all_x = []
    all_y = []

    for stroke in strokes:
        points = stroke.get("points", [])
        if not points:
            continue
        all_x.append([p["x"] for p in points])
        all_y.append([p["y"] for p in points])

    return {"strokes": {"x": all_x, "y": all_y}}


async def send_strokes(session: MathpixPageSession, strokes: list[dict]) -> dict:
    payload = reef_strokes_to_mathpix(strokes)
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{MATHPIX_BASE}/v3/strokes",
            headers={
                "app_token": session.app_token,
                "Content-Type": "application/json",
            },
            json={
                "strokes_session_id": session.strokes_session_id,
                "strokes": payload,
            },
        )
        resp.raise_for_status()
        return resp.json()


def schedule_transcription(session_id: str, page: int) -> None:
    key = (session_id, page)
    existing = _debounce_tasks.pop(key, None)
    if existing:
        existing.cancel()
    _debounce_tasks[key] = asyncio.create_task(
        _debounced_transcription(session_id, page)
    )


async def _debounced_transcription(session_id: str, page: int) -> None:
    await asyncio.sleep(DEBOUNCE_SECONDS)
    _debounce_tasks.pop((session_id, page), None)
    await _do_transcription(session_id, page)


async def _do_transcription(session_id: str, page: int) -> None:
    try:
        _get_credentials()
    except RuntimeError:
        return

    pool = get_pool()
    if not pool:
        return

    try:
        # Fetch all visible stroke_logs (resolve erases)
        async with pool.acquire() as conn:
            all_rows = await conn.fetch(
                """
                SELECT id, strokes, event_type
                FROM stroke_logs
                WHERE session_id = $1 AND page = $2 AND event_type IN ('draw', 'erase')
                ORDER BY received_at
                """,
                session_id, page,
            )

        visible_rows: list[dict] = []
        for row in all_rows:
            if row["event_type"] == "erase":
                visible_rows = [dict(row)]
            else:
                visible_rows.append(dict(row))

        if not visible_rows:
            return

        # Collect all visible strokes with their log IDs
        strokes_with_ids: list[tuple[int, dict]] = []
        for row in visible_rows:
            log_id = row["id"]
            strokes_data = row["strokes"]
            if isinstance(strokes_data, str):
                strokes_data = json.loads(strokes_data)
            for stroke in strokes_data:
                if stroke.get("points"):
                    strokes_with_ids.append((log_id, stroke))

        if not strokes_with_ids:
            return

        session = await get_or_create_session(session_id, page)

        # Filter to unsent strokes
        unsent = [
            (log_id, stroke) for log_id, stroke in strokes_with_ids
            if log_id not in session.sent_stroke_log_ids
        ]

        if not unsent:
            return

        unsent_strokes = [stroke for _, stroke in unsent]
        unsent_log_ids = {log_id for log_id, _ in unsent}

        result = await send_strokes(session, unsent_strokes)

        # Mark as sent
        session.sent_stroke_log_ids.update(unsent_log_ids)

        latex = result.get("latex_styled", "") or result.get("text", "")
        text = result.get("text", "")
        confidence = result.get("confidence", 0.0)

        session.last_latex = latex
        session.last_confidence = confidence

        # UPSERT into page_transcriptions
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO page_transcriptions (session_id, page, latex, text, confidence, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (session_id, page) DO UPDATE SET
                    latex = EXCLUDED.latex,
                    text = EXCLUDED.text,
                    confidence = EXCLUDED.confidence,
                    updated_at = NOW()
                """,
                session_id, page, latex, text, confidence,
            )

        print(
            f"[mathpix] ({session_id}, page={page}): "
            f"sent {len(unsent_strokes)} strokes, "
            f"confidence={confidence:.2f}, "
            f"latex={latex[:80]}"
        )

        # Notify connected WebSockets
        ws_set = _ws_by_session.get(session_id, set())
        msg = json.dumps({
            "type": "transcription",
            "page": page,
            "latex": latex,
            "text": text,
            "confidence": confidence,
        })
        for ws in list(ws_set):
            try:
                await ws.send_text(msg)
            except Exception:
                pass

    except Exception as e:
        print(f"[mathpix] error for ({session_id}, page={page}): {e}")
