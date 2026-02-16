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


# ── Per-cluster transcription ─────────────────────────────


def schedule_cluster_and_transcribe(session_id: str, page: int) -> None:
    """Debounce 500ms, then re-cluster + transcribe only dirty clusters."""
    key = (session_id, page)
    existing = _debounce_tasks.pop(key, None)
    if existing:
        existing.cancel()
    _debounce_tasks[key] = asyncio.create_task(
        _debounced_cluster_and_transcribe(session_id, page)
    )


async def _debounced_cluster_and_transcribe(session_id: str, page: int) -> None:
    await asyncio.sleep(DEBOUNCE_SECONDS)
    _debounce_tasks.pop((session_id, page), None)

    from lib.stroke_clustering import update_cluster_labels

    try:
        # 1. Re-cluster and get dirty labels
        dirty_labels = await update_cluster_labels(session_id, page)

        if not dirty_labels:
            print(f"[mathpix] ({session_id}, page={page}): no dirty clusters, skipping transcription")
            # Still concatenate in case cluster order changed
            await _concatenate_cluster_transcriptions(session_id, page)
            return

        # 2. Transcribe each dirty cluster sequentially
        for label in dirty_labels:
            await _do_cluster_transcription(session_id, page, label)

        # 3. Concatenate all cluster transcriptions into page_transcriptions
        await _concatenate_cluster_transcriptions(session_id, page)

    except Exception as e:
        print(f"[mathpix] error for ({session_id}, page={page}): {e}")


async def _do_cluster_transcription(
    session_id: str, page: int, cluster_label: int
) -> None:
    """Transcribe a single cluster's strokes via Mathpix."""
    try:
        _get_credentials()
    except RuntimeError:
        return

    pool = get_pool()
    if not pool:
        return

    # Fetch visible stroke_logs + their cluster_labels
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, strokes, event_type, cluster_labels
            FROM stroke_logs
            WHERE session_id = $1 AND page = $2 AND event_type IN ('draw', 'erase')
            ORDER BY received_at
            """,
            session_id, page,
        )

    # Resolve erases to get visible rows
    visible_rows: list[dict] = []
    for row in rows:
        if row["event_type"] == "erase":
            visible_rows = [dict(row)]
        else:
            visible_rows.append(dict(row))

    # Filter to strokes belonging to this cluster
    cluster_x: list[list[float]] = []
    cluster_y: list[list[float]] = []
    stroke_count = 0

    for row in visible_rows:
        strokes_data = row["strokes"]
        if isinstance(strokes_data, str):
            strokes_data = json.loads(strokes_data)
        labels_data = row.get("cluster_labels")
        if not labels_data:
            continue
        if isinstance(labels_data, str):
            labels_data = json.loads(labels_data)

        for idx, stroke in enumerate(strokes_data):
            if idx < len(labels_data) and labels_data[idx] == cluster_label:
                pts = stroke.get("points", [])
                if pts:
                    cluster_x.append([p["x"] for p in pts])
                    cluster_y.append([p["y"] for p in pts])
                    stroke_count += 1

    if not cluster_x:
        return

    # Send to Mathpix (reuse same page session)
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
                "strokes": {"strokes": {"x": cluster_x, "y": cluster_y}},
                "include_smiles": True,
                "include_geometry_data": True,
                "include_line_data": True,
            },
        )
        resp.raise_for_status()
        result = resp.json()

    latex = result.get("latex_styled", "") or result.get("text", "")
    raw_line_data = result.get("line_data")

    # Determine content_type from line_data
    content_type = "math"
    if raw_line_data and len(raw_line_data) > 0:
        first_line = raw_line_data[0]
        line_type = first_line.get("type", "")
        subtype = first_line.get("subtype", "")
        if line_type == "diagram" and subtype.startswith("chemistry"):
            content_type = "chemistry"
        elif line_type == "diagram":
            content_type = "other"
        # "math" stays as default

    # Update cluster row
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE clusters SET transcription = $1, content_type = $2
            WHERE session_id = $3 AND page = $4 AND cluster_label = $5
            """,
            latex, content_type, session_id, page, cluster_label,
        )

    from collections import Counter
    line_types = ""
    if raw_line_data:
        counts = Counter(ld.get("type", "unknown") for ld in raw_line_data)
        line_types = ", ".join(f"{t}={n}" for t, n in counts.most_common())
    print(
        f"[mathpix] cluster {cluster_label} ({session_id}, page={page}): "
        f"sent {stroke_count} strokes, "
        f"content_type={content_type}, "
        f"line_data=[{line_types}], "
        f"latex={latex[:80]}"
    )


async def _concatenate_cluster_transcriptions(session_id: str, page: int) -> None:
    """Concatenate all cluster transcriptions (ordered by centroid_y) into page_transcriptions."""
    pool = get_pool()
    if not pool:
        return

    async with pool.acquire() as conn:
        cluster_rows = await conn.fetch(
            """
            SELECT transcription, content_type FROM clusters
            WHERE session_id = $1 AND page = $2
            ORDER BY centroid_y ASC
            """,
            session_id, page,
        )

    if not cluster_rows:
        return

    # Concatenate non-empty transcriptions
    parts = [r["transcription"] for r in cluster_rows if r["transcription"]]
    latex = "\n\n".join(parts)
    text = latex  # use same for text field

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO page_transcriptions (session_id, page, latex, text, confidence, line_data, updated_at)
            VALUES ($1, $2, $3, $4, $5, NULL, NOW())
            ON CONFLICT (session_id, page) DO UPDATE SET
                latex = EXCLUDED.latex,
                text = EXCLUDED.text,
                confidence = EXCLUDED.confidence,
                line_data = NULL,
                updated_at = NOW()
            """,
            session_id, page, latex, text, 1.0,
        )

    print(f"[mathpix] concatenated {len(parts)} cluster transcriptions into page_transcriptions")
