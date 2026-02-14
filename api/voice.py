"""
WebSocket endpoint for voice message transcription.

Protocol:
  Client sends:  {"type": "voice_start", "session_id": "...", "user_id": "...", "page": 1}
  Client sends:  binary audio data (WAV)
  Client sends:  {"type": "voice_end"}
  Server sends:  {"type": "ack", "transcription": "..."}
"""

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from lib.database import get_pool
from lib.groq_transcribe import transcribe

router = APIRouter()


@router.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    """Receive audio from iPad, transcribe with Groq, store in DB."""
    await ws.accept()

    try:
        while True:
            # Wait for voice_start
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") != "voice_start":
                await ws.send_json({"type": "error", "detail": "Expected voice_start"})
                continue

            session_id = msg.get("session_id", "")
            user_id = msg.get("user_id", "")
            page = msg.get("page", 0)

            # Accumulate binary audio chunks until voice_end
            audio_buffer = bytearray()
            while True:
                ws_msg = await ws.receive()
                if "text" in ws_msg:
                    inner = json.loads(ws_msg["text"])
                    if inner.get("type") == "voice_end":
                        break
                elif "bytes" in ws_msg:
                    audio_buffer.extend(ws_msg["bytes"])

            if not audio_buffer:
                await ws.send_json({"type": "error", "detail": "No audio received"})
                continue

            # Transcribe in a thread (blocking OpenAI SDK call)
            text = await asyncio.to_thread(transcribe, bytes(audio_buffer))

            # Store in DB
            pool = get_pool()
            if pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO stroke_logs
                            (session_id, page, strokes, event_type, message, user_id)
                        VALUES ($1, $2, '[]'::jsonb, 'voice', $3, $4)
                        """,
                        session_id,
                        page,
                        text,
                        user_id,
                    )

            await ws.send_json({"type": "ack", "transcription": text})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
