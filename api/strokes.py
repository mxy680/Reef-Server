"""
WebSocket endpoint for real-time stroke logging.

iOS streams full PKStrokePoint data per page; server logs
each batch to the stroke_logs table in Postgres.
"""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from lib.database import get_pool
from lib.stroke_clustering import update_cluster_labels

router = APIRouter()

_active_ws: set[WebSocket] = set()
_active_sessions: dict[WebSocket, str] = {}


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

    # Fetch transcriptions for this session
    transcriptions: dict[int, str] = {}
    if session_id:
        async with pool.acquire() as conn:
            transcription_rows = await conn.fetch(
                """
                SELECT cluster_label, transcription FROM clusters
                WHERE session_id = $1 AND transcription != ''
                """,
                session_id,
            )
        transcriptions = {r["cluster_label"]: r["transcription"] for r in transcription_rows}

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
        "ws_connections": len(_active_ws),
        "active_sessions": list(_active_sessions.values()),
        "transcriptions": transcriptions,
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
        else:
            result = await conn.execute("DELETE FROM stroke_logs")

    count = int(result.split()[-1])
    return {"deleted": count}


@router.websocket("/ws/strokes")
async def strokes_websocket(ws: WebSocket, session_id: str = "", user_id: str = ""):
    await ws.accept()
    _active_ws.add(ws)

    # Register session from query param and insert system log
    if session_id:
        _active_sessions[ws] = session_id
        print(f"[strokes_ws] session {session_id} connected via query param")
        pool = get_pool()
        if pool:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO stroke_logs (session_id, page, strokes, event_type, message, user_id)
                    VALUES ($1, 0, '[]'::jsonb, 'system', $2, $3)
                    """,
                    session_id,
                    "session started",
                    user_id,
                )

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            msg_type = msg.get("type")

            print(f"[strokes_ws] received msg_type={msg_type}")

            if msg_type == "hello":
                sid = msg.get("session_id", "")
                if sid:
                    _active_sessions[ws] = sid
                    print(f"[strokes_ws] hello from session {sid}")
                    pool = get_pool()
                    if pool:
                        async with pool.acquire() as conn:
                            await conn.execute(
                                """
                                INSERT INTO stroke_logs (session_id, page, strokes, event_type, message)
                                VALUES ($1, 0, '[]'::jsonb, 'system', $2)
                                """,
                                sid,
                                "session started",
                            )
                await ws.send_text(json.dumps({"type": "ack"}))
                continue

            if msg_type == "clear":
                session_id = msg.get("session_id", "")
                page = msg.get("page", 1)
                pool = get_pool()
                if pool:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "DELETE FROM stroke_logs WHERE session_id = $1 AND page = $2",
                            session_id,
                            page,
                        )
                await ws.send_text(json.dumps({"type": "ack"}))
                continue

            if msg_type == "system":
                sid = msg.get("session_id", "")
                page = msg.get("page", 0)
                message = msg.get("message", "")
                pool = get_pool()
                if pool:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            """
                            INSERT INTO stroke_logs (session_id, page, strokes, event_type, message, user_id)
                            VALUES ($1, $2, '[]'::jsonb, 'system', $3, $4)
                            """,
                            sid,
                            page,
                            message,
                            msg.get("user_id", user_id),
                        )
                await ws.send_text(json.dumps({"type": "ack"}))
                continue

            if msg_type != "strokes":
                continue

            session_id = msg.get("session_id", "")
            if session_id:
                _active_sessions[ws] = session_id
            page = msg.get("page", 1)
            strokes = msg.get("strokes", [])
            event_type = msg.get("event_type", "draw")
            deleted_count = msg.get("deleted_count", 0)

            pool = get_pool()
            if pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO stroke_logs (session_id, page, strokes, event_type, deleted_count, user_id)
                        VALUES ($1, $2, $3::jsonb, $4, $5, $6)
                        """,
                        session_id,
                        page,
                        json.dumps(strokes),
                        event_type,
                        deleted_count,
                        msg.get("user_id", user_id),
                    )
                # Re-cluster in background (don't block ack)
                asyncio.create_task(update_cluster_labels(session_id, page))

            await ws.send_text(json.dumps({"type": "ack"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[strokes_ws] error: {e}")
        try:
            await ws.close(code=1011, reason=str(e)[:120])
        except Exception:
            pass
    finally:
        _active_ws.discard(ws)
        _active_sessions.pop(ws, None)
