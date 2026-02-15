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
from lib.reasoning import run_reasoning
from lib.stroke_clustering import clear_session_usage, get_session_usage, update_cluster_labels

router = APIRouter()

_active_ws: set[WebSocket] = set()
_active_sessions: dict[WebSocket, str] = {}

# Per-session stroke sequence counter — increments on each stroke event.
# After transcription finishes, reasoning only runs if no new strokes arrived.
_stroke_seq: dict[str, int] = {}

REASONING_DELAY_S = 2.5


async def _cluster_then_reason(session_id: str, page: int, seq: int):
    """Run clustering/transcription, then reasoning if no new strokes arrived."""
    try:
        await update_cluster_labels(session_id, page)
        # Early exit if new strokes arrived during transcription
        if _stroke_seq.get(session_id, 0) != seq:
            print(f"[reasoning] new strokes during transcription, skipping")
            return
        # Debounce: wait before firing reasoning
        print(f"[reasoning] transcription done, waiting {REASONING_DELAY_S}s (seq={seq})")
        await asyncio.sleep(REASONING_DELAY_S)
        # Re-check: did new strokes arrive during the delay?
        if _stroke_seq.get(session_id, 0) != seq:
            print(f"[reasoning] new strokes during delay, skipping")
            return
        print(f"[reasoning] no new strokes since seq={seq}, triggering reasoning")
        result = await run_reasoning(session_id, page)
        print(f"[reasoning] result: {result}")
    except Exception as e:
        print(f"[reasoning] ERROR: {e}")
        import traceback
        traceback.print_exc()


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

    # Fetch transcriptions and content types for this session
    transcriptions: dict[int, str] = {}
    content_types: dict[int, str] = {}
    cluster_order: list[int] = []
    if session_id:
        async with pool.acquire() as conn:
            transcription_rows = await conn.fetch(
                """
                SELECT cluster_label, transcription, content_type, centroid_y FROM clusters
                WHERE session_id = $1 AND transcription != ''
                ORDER BY centroid_y ASC
                """,
                session_id,
            )
        transcriptions = {r["cluster_label"]: r["transcription"] for r in transcription_rows}
        content_types = {r["cluster_label"]: r["content_type"] for r in transcription_rows}
        cluster_order = [r["cluster_label"] for r in transcription_rows]

    # Fetch latest problem context for this session
    problem_context = ""
    if session_id:
        async with pool.acquire() as conn:
            ctx_row = await conn.fetchrow(
                """
                SELECT problem_context FROM stroke_logs
                WHERE session_id = $1 AND problem_context != ''
                ORDER BY received_at DESC LIMIT 1
                """,
                session_id,
            )
            if ctx_row:
                problem_context = ctx_row["problem_context"]

    # Compute token usage and cost for session
    usage = None
    if session_id:
        raw_usage = get_session_usage(session_id)
        prompt_tokens = raw_usage["prompt_tokens"]
        completion_tokens = raw_usage["completion_tokens"]
        # Gemini 3 Flash Preview via OpenRouter: $0.50/M input, $3.00/M output
        estimated_cost = (prompt_tokens * 0.50 + completion_tokens * 3.00) / 1_000_000
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "calls": raw_usage["calls"],
            "estimated_cost": round(estimated_cost, 6),
        }

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
        "content_types": content_types,
        "cluster_order": cluster_order,
        "usage": usage,
        "problem_context": problem_context,
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
                "DELETE FROM reasoning_logs WHERE session_id = $1", session_id
            )
            clear_session_usage(session_id)
            _stroke_seq.pop(session_id, None)
        else:
            result = await conn.execute("DELETE FROM stroke_logs")
            await conn.execute("DELETE FROM clusters")
            await conn.execute("DELETE FROM reasoning_logs")

    count = int(result.split()[-1])
    return {"deleted": count}


@router.get("/api/reasoning-logs")
async def get_reasoning_logs(
    session_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")

    async with pool.acquire() as conn:
        if session_id:
            rows = await conn.fetch(
                """
                SELECT id, session_id, page, created_at, context,
                       action, level, target, error_type, delay_ms, message,
                       prompt_tokens, completion_tokens, cached_tokens, estimated_cost
                FROM reasoning_logs
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                session_id, limit,
            )
            total_row = await conn.fetchval(
                "SELECT COUNT(*) FROM reasoning_logs WHERE session_id = $1",
                session_id,
            )
            # Aggregate reasoning usage for this session
            usage_row = await conn.fetchrow(
                """
                SELECT COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
                       COALESCE(SUM(estimated_cost), 0) AS estimated_cost,
                       COUNT(*) AS calls
                FROM reasoning_logs WHERE session_id = $1
                """,
                session_id,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, session_id, page, created_at, context,
                       action, level, target, error_type, delay_ms, message,
                       prompt_tokens, completion_tokens, cached_tokens, estimated_cost
                FROM reasoning_logs
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
            total_row = await conn.fetchval("SELECT COUNT(*) FROM reasoning_logs")
            usage_row = None

    reasoning_usage = None
    if usage_row:
        reasoning_usage = {
            "prompt_tokens": usage_row["prompt_tokens"],
            "completion_tokens": usage_row["completion_tokens"],
            "cached_tokens": usage_row["cached_tokens"],
            "estimated_cost": float(usage_row["estimated_cost"]),
            "calls": usage_row["calls"],
        }

    return {
        "logs": [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "page": r["page"],
                "created_at": r["created_at"].isoformat(),
                "context": r["context"],
                "action": r["action"],
                "level": r["level"],
                "target": r["target"],
                "error_type": r["error_type"],
                "delay_ms": r["delay_ms"],
                "message": r["message"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "cached_tokens": r["cached_tokens"],
                "estimated_cost": r["estimated_cost"],
            }
            for r in rows
        ],
        "total": total_row,
        "reasoning_usage": reasoning_usage,
    }


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
                        await conn.execute(
                            "DELETE FROM clusters WHERE session_id = $1 AND page = $2",
                            session_id,
                            page,
                        )
                clear_session_usage(session_id)
                _stroke_seq.pop(session_id, None)
                await ws.send_text(json.dumps({"type": "ack"}))
                continue

            if msg_type == "context":
                sid = msg.get("session_id", "")
                page = msg.get("page", 0)
                problem_context = msg.get("problem_context", "")
                pool = get_pool()
                if pool and problem_context:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            """
                            INSERT INTO stroke_logs (session_id, page, strokes, event_type, message, user_id, problem_context)
                            VALUES ($1, $2, '[]'::jsonb, 'system', 'problem_context', $3, $4)
                            """,
                            sid,
                            page,
                            msg.get("user_id", user_id),
                            problem_context,
                        )
                    print(f"[strokes_ws] stored problem context for session {sid}: {problem_context[:80]}...")
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
                # Bump stroke sequence, then cluster + reason in background
                _stroke_seq[session_id] = _stroke_seq.get(session_id, 0) + 1
                seq = _stroke_seq[session_id]
                asyncio.create_task(_cluster_then_reason(session_id, page, seq))

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
