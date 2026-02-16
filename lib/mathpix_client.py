"""Mathpix strokes-session management.

Opens a Mathpix strokes session on first draw for a (session_id, page) pair.
Sessions are cached and reused within their 5-min TTL.

Requires MATHPIX_APP_ID and MATHPIX_APP_KEY env vars.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

MATHPIX_BASE = "https://api.mathpix.com"


@dataclass
class MathpixSession:
    strokes_session_id: str
    app_token: str
    expires_at: datetime


# (session_id, page) → MathpixSession
_sessions: dict[tuple[str, int], MathpixSession] = {}


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
    removed = _sessions.pop(key, None)
    if removed:
        print(f"[mathpix] invalidated session ({session_id}, page={page})")


def cleanup_sessions(session_id: str) -> None:
    keys_to_remove = [k for k in _sessions if k[0] == session_id]
    for key in keys_to_remove:
        _sessions.pop(key, None)
    if keys_to_remove:
        print(f"[mathpix] cleaned up {len(keys_to_remove)} session(s) for {session_id}")
