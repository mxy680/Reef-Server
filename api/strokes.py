"""
WebSocket endpoint for real-time stroke logging.

iOS streams full PKStrokePoint data per page; server logs
each batch to the stroke_logs table in Postgres.
"""

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from lib.database import get_pool

router = APIRouter()


@router.websocket("/ws/strokes")
async def strokes_websocket(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") != "strokes":
                continue

            session_id = msg.get("session_id", "")
            page = msg.get("page", 1)
            strokes = msg.get("strokes", [])

            pool = get_pool()
            if pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO stroke_logs (session_id, page, strokes)
                        VALUES ($1, $2, $3::jsonb)
                        """,
                        session_id,
                        page,
                        json.dumps(strokes),
                    )

            await ws.send_text(json.dumps({"type": "ack"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[strokes_ws] error: {e}")
        try:
            await ws.close(code=1011, reason=str(e)[:120])
        except Exception:
            pass
