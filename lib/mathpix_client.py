"""Mathpix strokes-session transcription.

Sends ALL visible strokes to Mathpix /v3/strokes on each debounced draw,
reusing the same session within its 5-min TTL (billed once per session).

Requires MATHPIX_APP_ID and MATHPIX_APP_KEY env vars.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from lib.database import get_pool

MATHPIX_BASE = "https://api.mathpix.com"
DEBOUNCE_SECONDS = 0.5


@dataclass
class MathpixSession:
    strokes_session_id: str
    app_token: str
    expires_at: datetime


# (session_id, page) → MathpixSession
_sessions: dict[tuple[str, int], MathpixSession] = {}

# (session_id, page) → pending debounce asyncio.Task
_debounce_tasks: dict[tuple[str, int], asyncio.Task] = {}


def _get_credentials() -> tuple[str, str]:
    app_id = os.environ.get("MATHPIX_APP_ID", "")
    app_key = os.environ.get("MATHPIX_APP_KEY", "")
    if not app_id or not app_key:
        raise RuntimeError("MATHPIX_APP_ID and MATHPIX_APP_KEY not set")
    return app_id, app_key


async def create_session() -> MathpixSession:
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

    return MathpixSession(
        strokes_session_id=data["strokes_session_id"],
        app_token=data["app_token"],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=4, seconds=30),
    )


async def get_or_create_session(
    session_id: str, page: int
) -> MathpixSession:
    key = (session_id, page)
    existing = _sessions.get(key)
    if existing and datetime.now(timezone.utc) < existing.expires_at:
        return existing
    session = await create_session()
    _sessions[key] = session
    print(f"[mathpix] opened session ({session_id}, page={page})")
    return session


def invalidate_session(session_id: str, page: int) -> None:
    key = (session_id, page)
    _sessions.pop(key, None)
    task = _debounce_tasks.pop(key, None)
    if task:
        task.cancel()
    print(f"[mathpix] invalidated session ({session_id}, page={page})")


def get_session_info(session_id: str, page: int) -> dict | None:
    """Return session expiry and strokes_session_id, or None if no session."""
    key = (session_id, page)
    session = _sessions.get(key)
    if session and datetime.now(timezone.utc) < session.expires_at:
        return {
            "expires_at": session.expires_at.isoformat(),
            "strokes_session_id": session.strokes_session_id,
        }
    return None


def cleanup_sessions(session_id: str) -> None:
    keys_to_remove = [k for k in _sessions if k[0] == session_id]
    for key in keys_to_remove:
        _sessions.pop(key, None)
        task = _debounce_tasks.pop(key, None)
        if task:
            task.cancel()
    if keys_to_remove:
        print(f"[mathpix] cleaned up {len(keys_to_remove)} session(s) for {session_id}")


# ── Transcription ─────────────────────────────────────────


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
            rows = await conn.fetch(
                """
                SELECT strokes, event_type
                FROM stroke_logs
                WHERE session_id = $1 AND page = $2 AND event_type IN ('draw', 'erase')
                ORDER BY received_at
                """,
                session_id, page,
            )

        # Resolve erases: erase event contains remaining visible strokes
        visible: list[dict] = []
        for row in rows:
            strokes_data = row["strokes"]
            if isinstance(strokes_data, str):
                strokes_data = json.loads(strokes_data)
            if row["event_type"] == "erase":
                visible = [s for s in strokes_data if s.get("points")]
            else:
                for s in strokes_data:
                    if s.get("points"):
                        visible.append(s)

        if not visible:
            return

        # Convert to Mathpix format
        all_x = []
        all_y = []
        for stroke in visible:
            pts = stroke["points"]
            all_x.append([p["x"] for p in pts])
            all_y.append([p["y"] for p in pts])

        # Get or create session, then send
        session = await get_or_create_session(session_id, page)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{MATHPIX_BASE}/v3/strokes",
                headers={
                    "app_token": session.app_token,
                    "Content-Type": "application/json",
                },
                json={
                    "strokes_session_id": session.strokes_session_id,
                    "strokes": {"strokes": {"x": all_x, "y": all_y}},
                },
            )
            resp.raise_for_status()
            result = resp.json()

        latex = result.get("latex_styled", "") or result.get("text", "")
        text = result.get("text", "")
        confidence = result.get("confidence", 0.0)

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
            f"sent {len(visible)} strokes, "
            f"confidence={confidence:.2f}, "
            f"latex={latex[:80]}"
        )

    except Exception as e:
        print(f"[mathpix] error for ({session_id}, page={page}): {e}")
