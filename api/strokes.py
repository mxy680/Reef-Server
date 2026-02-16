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
from lib.reasoning import _find_matching_question, _get_cached_question, clear_reasoning_usage, get_reasoning_usage, run_reasoning
from lib.stroke_clustering import clear_session_usage, get_session_usage, update_cluster_labels

router = APIRouter()

_active_ws: set[WebSocket] = set()
# Each value: {"session_id": str, "document_name": str, "question_number": int|None, "question_id": int|None}
_active_sessions: dict[WebSocket, dict] = {}

# Reasoning debounce: per-(session, page) sequence counter
_stroke_seq: dict[tuple[str, int], int] = {}
REASONING_DELAY_S = 2.5


async def _cluster_then_reason(session_id: str, page: int):
    """Re-cluster, then debounce-trigger reasoning after a delay."""
    # Bump sequence counter
    key = (session_id, page)
    _stroke_seq[key] = _stroke_seq.get(key, 0) + 1
    seq_at_start = _stroke_seq[key]

    # Run clustering (transcription happens inside)
    await update_cluster_labels(session_id, page)

    # Check if more strokes arrived during clustering
    if _stroke_seq.get(key, 0) != seq_at_start:
        return

    # Wait for debounce period
    await asyncio.sleep(REASONING_DELAY_S)

    # Check again after delay
    if _stroke_seq.get(key, 0) != seq_at_start:
        return

    # Run reasoning
    try:
        result = await run_reasoning(session_id, page)
        if result["action"] == "speak":
            # Send reasoning message to all WebSocket clients for this session
            message = json.dumps({
                "type": "reasoning",
                "message": result["message"],
                "session_id": session_id,
                "page": page,
            })
            for ws, info in list(_active_sessions.items()):
                if info["session_id"] == session_id:
                    try:
                        await ws.send_text(message)
                    except Exception:
                        pass
    except Exception as exc:
        print(f"[reasoning] error for session {session_id}: {exc}")


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
    matched_question_text = ""
    matched_question_label = ""
    document_name = ""
    answer_key = ""
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

            # Check if this session has an active WebSocket with question metadata
            active_question_id = None
            active_doc_name = ""
            for ws_info in _active_sessions.values():
                if ws_info["session_id"] == session_id:
                    active_question_id = ws_info.get("question_id")
                    active_doc_name = ws_info.get("document_name", "")
                    break

            if active_question_id:
                # Direct lookup — no matching/caching needed
                q_row = await conn.fetchrow(
                    "SELECT id, text, label FROM questions WHERE id = $1", active_question_id
                )
                if q_row:
                    matched_question_text = q_row["text"]
                    matched_question_label = q_row["label"] or ""
                    document_name = active_doc_name
                    ak_rows = await conn.fetch(
                        "SELECT part_label, answer FROM answer_keys WHERE question_id = $1 ORDER BY id",
                        q_row["id"],
                    )
                    if ak_rows:
                        parts = []
                        for r in ak_rows:
                            label = f"({r['part_label']}) " if r["part_label"] else ""
                            parts.append(f"{label}{r['answer']}")
                        answer_key = "\n".join(parts)
            else:
                # Fallback: trigram matching for sessions without active WebSocket
                canvas_text = " ".join(transcriptions.get(l, "") for l in cluster_order)
                matched_q = None
                if canvas_text.strip():
                    matched_q = await _find_matching_question(conn, canvas_text, session_id=session_id)
                if not matched_q:
                    matched_q = await _get_cached_question(conn, session_id)
                if matched_q:
                    matched_question_text = matched_q["text"]
                    matched_question_label = matched_q.get("label", "")
                    document_name = matched_q.get("doc_filename", "")
                    ak_rows = await conn.fetch(
                        "SELECT part_label, answer FROM answer_keys WHERE question_id = $1 ORDER BY id",
                        matched_q["id"],
                    )
                    if ak_rows:
                        parts = []
                        for r in ak_rows:
                            label = f"({r['part_label']}) " if r["part_label"] else ""
                            parts.append(f"{label}{r['answer']}")
                        answer_key = "\n".join(parts)

    # Compute token usage and cost for session
    usage = None
    if session_id:
        raw_usage = get_session_usage(session_id)
        prompt_tokens = raw_usage["prompt_tokens"]
        completion_tokens = raw_usage["completion_tokens"]
        # Llama 4 Maverick on Groq: $0.20/M input, $0.60/M output
        estimated_cost = (prompt_tokens * 0.20 + completion_tokens * 0.60) / 1_000_000
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
        "active_sessions": list(set(v["session_id"] for v in _active_sessions.values())),
        "transcriptions": transcriptions,
        "content_types": content_types,
        "cluster_order": cluster_order,
        "usage": usage,
        "problem_context": matched_question_text or problem_context,
        "answer_key": answer_key,
        "matched_question_label": matched_question_label,
        "document_name": document_name,
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
            # Keep reasoning_logs — the model needs its conversation history
            # to recognize when a student corrects a mistake it pointed out
            clear_session_usage(session_id)
        else:
            result = await conn.execute("DELETE FROM stroke_logs")
            await conn.execute("DELETE FROM clusters")
            # Keep reasoning_logs for conversation continuity

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
                SELECT id, session_id, page, created_at, action, message,
                       prompt_tokens, completion_tokens, estimated_cost
                FROM reasoning_logs
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                session_id, limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, session_id, page, created_at, action, message,
                       prompt_tokens, completion_tokens, estimated_cost
                FROM reasoning_logs
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )

    # Compute reasoning usage for session
    usage = None
    if session_id:
        raw = get_reasoning_usage(session_id)
        prompt_tokens = raw["prompt_tokens"]
        completion_tokens = raw["completion_tokens"]
        # GPT-OSS 120B on Groq
        estimated_cost = (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "calls": raw["calls"],
            "estimated_cost": round(estimated_cost, 6),
        }

    return {
        "logs": [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "page": r["page"],
                "created_at": r["created_at"].isoformat(),
                "action": r["action"],
                "message": r["message"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "estimated_cost": r["estimated_cost"],
            }
            for r in rows
        ],
        "usage": usage,
    }


@router.websocket("/ws/strokes")
async def strokes_websocket(ws: WebSocket, session_id: str = "", user_id: str = "",
                            document_name: str = "", question_number: int | None = None):
    await ws.accept()
    _active_ws.add(ws)

    # Strip extension from document_name for DB matching
    doc_name = document_name.rsplit(".", 1)[0] if "." in document_name else document_name

    # Immediate question lookup from connection params
    question_id = None
    if doc_name and question_number is not None:
        pool = get_pool()
        if pool:
            async with pool.acquire() as conn:
                q_row = await conn.fetchrow("""
                    SELECT q.id FROM questions q
                    JOIN documents d ON q.document_id = d.id
                    WHERE d.filename = $1 AND q.number = $2
                    ORDER BY d.id DESC LIMIT 1
                """, doc_name, question_number)
                if q_row:
                    question_id = q_row["id"]
                    print(f"[strokes_ws] immediate question match: doc={doc_name!r} q_num={question_number} -> Q{question_id}")

    # Register session from query param and insert system log
    if session_id:
        _active_sessions[ws] = {
            "session_id": session_id,
            "document_name": document_name,  # Keep original with extension for display
            "question_number": question_number,
            "question_id": question_id,
        }
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
                    _active_sessions[ws] = {
                        "session_id": sid,
                        "document_name": document_name,
                        "question_number": question_number,
                        "question_id": question_id,
                    }
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
                        # Keep reasoning_logs — model needs conversation history
                clear_session_usage(session_id)
                await ws.send_text(json.dumps({"type": "ack"}))
                continue

            if msg_type == "context":
                sid = msg.get("session_id", "")
                page = msg.get("page", 0)
                problem_context = msg.get("problem_context", "")
                doc_name_raw = msg.get("document_name", "")
                # Strip extension — DB stores stem only (e.g. "document" not "document.pdf")
                doc_name = doc_name_raw.rsplit(".", 1)[0] if "." in doc_name_raw else doc_name_raw
                question_number = msg.get("question_number")
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
                        # Try exact lookup first (iOS sends document_name + question_number)
                        matched = False
                        if doc_name and question_number is not None:
                            q_row = await conn.fetchrow(
                                """
                                SELECT q.id FROM questions q
                                JOIN documents d ON q.document_id = d.id
                                JOIN answer_keys ak ON ak.question_id = q.id
                                WHERE d.filename = $1 AND q.number = $2
                                ORDER BY d.id DESC LIMIT 1
                                """,
                                doc_name, question_number,
                            )
                            if q_row:
                                await conn.execute(
                                    """
                                    INSERT INTO session_question_cache (session_id, question_id, document_name, updated_at)
                                    VALUES ($1, $2, $3, NOW())
                                    ON CONFLICT (session_id) DO UPDATE SET question_id = $2, document_name = $3, updated_at = NOW()
                                    """,
                                    sid, q_row["id"], doc_name_raw,
                                )
                                matched = True
                                print(f"[strokes_ws] exact match: doc={doc_name!r} q_num={question_number} -> Q{q_row['id']}")
                        # Fall back to trigram matching
                        if not matched:
                            await _find_matching_question(conn, problem_context, session_id=sid)
                    print(f"[strokes_ws] stored problem context for session {sid}: {problem_context[:80]}...")
                elif pool and doc_name and question_number is not None:
                    # Even without problem_context text, try exact match
                    async with pool.acquire() as conn:
                        q_row = await conn.fetchrow(
                            """
                            SELECT q.id FROM questions q
                            JOIN documents d ON q.document_id = d.id
                            JOIN answer_keys ak ON ak.question_id = q.id
                            WHERE d.filename = $1 AND q.number = $2
                            ORDER BY d.id DESC LIMIT 1
                            """,
                            doc_name, question_number,
                        )
                        if q_row:
                            await conn.execute(
                                """
                                INSERT INTO session_question_cache (session_id, question_id, document_name, updated_at)
                                VALUES ($1, $2, $3, NOW())
                                ON CONFLICT (session_id) DO UPDATE SET question_id = $2, document_name = $3, updated_at = NOW()
                                """,
                                sid, q_row["id"], doc_name_raw,
                            )
                            print(f"[strokes_ws] exact match (no context): doc={doc_name!r} q_num={question_number} -> Q{q_row['id']}")
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
                existing = _active_sessions.get(ws, {})
                existing["session_id"] = session_id
                _active_sessions[ws] = existing
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
                # Re-cluster + reasoning in background (don't block ack)
                asyncio.create_task(_cluster_then_reason(session_id, page))

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
