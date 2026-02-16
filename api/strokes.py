"""
REST endpoints for stroke logging.

iOS sends debounced stroke data via POST; server logs
each batch to the stroke_logs table in Postgres.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from lib.database import get_pool
from lib.mathpix_client import (
    cleanup_sessions,
    get_or_create_session,
    get_session_expiry,
    invalidate_session,
)
from lib.stroke_clustering import update_cluster_labels

router = APIRouter()

# session_id → {document_name, question_number, last_seen}
_active_sessions: dict[str, dict] = {}


# ── Pydantic request models ────────────────────────────────

class ConnectRequest(BaseModel):
    session_id: str
    user_id: str = ""
    document_name: Optional[str] = None
    question_number: Optional[int] = None


class DisconnectRequest(BaseModel):
    session_id: str


class StrokesRequest(BaseModel):
    session_id: str
    user_id: str = ""
    page: int = 1
    strokes: list = []
    event_type: str = "draw"
    deleted_count: int = 0


class ClearRequest(BaseModel):
    session_id: str
    page: int = 1


# ── REST endpoints ──────────────────────────────────────────

@router.post("/api/strokes/connect")
async def strokes_connect(req: ConnectRequest):
    # Evict stale session metadata (e.g. question-switch sessions)
    # Only remove from _active_sessions — don't destroy Mathpix page
    # sessions, which need to persist for incremental transcription
    stale = [sid for sid in _active_sessions if sid != req.session_id]
    for sid in stale:
        _active_sessions.pop(sid, None)

    _active_sessions[req.session_id] = {
        "document_name": req.document_name or "",
        "question_number": req.question_number,
        "last_seen": datetime.now(timezone.utc).isoformat(),
    }
    print(f"[strokes] session {req.session_id} connected (evicted {len(stale)} stale)")

    pool = get_pool()
    if pool:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO stroke_logs (session_id, page, strokes, event_type, message, user_id)
                VALUES ($1, 0, '[]'::jsonb, 'system', $2, $3)
                """,
                req.session_id,
                "session started",
                req.user_id,
            )

    return {"status": "connected"}


@router.post("/api/strokes/disconnect")
async def strokes_disconnect(req: DisconnectRequest):
    _active_sessions.pop(req.session_id, None)
    cleanup_sessions(req.session_id)
    print(f"[strokes] session {req.session_id} disconnected")
    return {"status": "disconnected"}


@router.post("/api/strokes")
async def strokes_post(req: StrokesRequest):
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO stroke_logs (session_id, page, strokes, event_type, deleted_count, user_id)
            VALUES ($1, $2, $3::jsonb, $4, $5, $6)
            """,
            req.session_id,
            req.page,
            json.dumps(req.strokes),
            req.event_type,
            req.deleted_count,
            req.user_id,
        )

    # Re-cluster in background
    asyncio.create_task(update_cluster_labels(req.session_id, req.page))

    if req.event_type == "erase":
        invalidate_session(req.session_id, req.page)
    elif req.event_type == "draw":
        await get_or_create_session(req.session_id, req.page)

    # Update last_seen
    if req.session_id in _active_sessions:
        _active_sessions[req.session_id]["last_seen"] = datetime.now(timezone.utc).isoformat()

    return {"status": "ok"}


@router.post("/api/strokes/clear")
async def strokes_clear(req: ClearRequest):
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")

    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM stroke_logs WHERE session_id = $1 AND page = $2",
            req.session_id,
            req.page,
        )
        await conn.execute(
            "DELETE FROM clusters WHERE session_id = $1 AND page = $2",
            req.session_id,
            req.page,
        )
        await conn.execute(
            "DELETE FROM page_transcriptions WHERE session_id = $1 AND page = $2",
            req.session_id,
            req.page,
        )

    invalidate_session(req.session_id, req.page)
    return {"status": "ok"}


# ── Existing GET / DELETE endpoints ─────────────────────────

@router.get("/api/stroke-logs")
async def get_stroke_logs(
    limit: int = Query(default=50, ge=1, le=200),
    session_id: Optional[str] = Query(default=None),
    page: Optional[int] = Query(default=None),
):
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")

    async with pool.acquire() as conn:
        # Build query with optional filters
        conditions = []
        params = []
        idx = 1

        if session_id:
            conditions.append(f"session_id = ${idx}")
            params.append(session_id)
            idx += 1

        if page is not None:
            conditions.append(f"page = ${idx}")
            params.append(page)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM stroke_logs {where}", *params
        )

        rows = await conn.fetch(
            f"""
            SELECT id, session_id, page, received_at,
                   jsonb_array_length(strokes) AS stroke_count,
                   strokes, event_type, deleted_count, message, user_id,
                   cluster_labels
            FROM stroke_logs
            {where}
            ORDER BY received_at DESC
            LIMIT ${idx}
            """,
            *params,
            limit,
        )

    # Fetch cluster order (sorted by centroid_y for reading order)
    cluster_order: list[int] = []
    if session_id:
        async with pool.acquire() as conn:
            cluster_rows = await conn.fetch(
                """
                SELECT cluster_label, centroid_y FROM clusters
                WHERE session_id = $1
                ORDER BY centroid_y ASC
                """,
                session_id,
            )
        cluster_order = [r["cluster_label"] for r in cluster_rows]

    # Look up document_name and matched question label from active session
    active_doc_name = ""
    matched_question_label = ""
    if session_id and session_id in _active_sessions:
        info = _active_sessions[session_id]
        active_doc_name = info.get("document_name", "")
        qn = info.get("question_number")
        if active_doc_name and qn is not None:
            # Strip extension — iOS sends "foo.pdf" but DB stores "foo"
            doc_stem = active_doc_name.rsplit(".", 1)[0] if "." in active_doc_name else active_doc_name
            async with pool.acquire() as conn:
                q_row = await conn.fetchrow(
                    """
                    SELECT q.label FROM questions q
                    JOIN documents d ON q.document_id = d.id
                    WHERE d.filename = $1 AND q.number = $2
                    """,
                    doc_stem,
                    qn,
                )
                if q_row:
                    matched_question_label = q_row["label"] or ""

    return {
        "logs": [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "page": r["page"],
                "received_at": r["received_at"].isoformat(),
                "stroke_count": r["stroke_count"],
                "strokes": json.loads(r["strokes"]),
                "event_type": r["event_type"],
                "deleted_count": r["deleted_count"],
                "message": r["message"],
                "user_id": r["user_id"],
                "cluster_labels": json.loads(r["cluster_labels"]),
            }
            for r in rows
        ],
        "total": total,
        "active_connections": len(_active_sessions),
        "active_sessions": sorted(
            _active_sessions.keys(),
            key=lambda sid: _active_sessions[sid].get("last_seen", ""),
        ),
        "cluster_order": cluster_order,
        "document_name": active_doc_name,
        "matched_question_label": matched_question_label,
        "mathpix_session_expires_at": get_session_expiry(session_id, page or 1) if session_id else None,
    }


@router.delete("/api/stroke-logs")
async def clear_stroke_logs(
    session_id: Optional[str] = Query(default=None),
):
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")

    async with pool.acquire() as conn:
        if session_id:
            result = await conn.execute(
                "DELETE FROM stroke_logs WHERE session_id = $1", session_id
            )
            await conn.execute(
                "DELETE FROM clusters WHERE session_id = $1", session_id
            )
            await conn.execute(
                "DELETE FROM page_transcriptions WHERE session_id = $1", session_id
            )
        else:
            result = await conn.execute("DELETE FROM stroke_logs")
            await conn.execute("DELETE FROM clusters")
            await conn.execute("DELETE FROM page_transcriptions")

    count = int(result.split()[-1])
    return {"deleted": count}


@router.get("/api/page-transcription")
async def get_page_transcription(
    session_id: str = Query(...),
    page: int = Query(default=1),
):
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT latex, text, confidence, updated_at
            FROM page_transcriptions
            WHERE session_id = $1 AND page = $2
            """,
            session_id,
            page,
        )

    if not row:
        return {"latex": "", "text": "", "confidence": 0, "updated_at": None}

    return {
        "latex": row["latex"],
        "text": row["text"],
        "confidence": row["confidence"],
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }
