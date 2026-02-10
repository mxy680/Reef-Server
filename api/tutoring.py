"""WebSocket endpoint for real-time AI tutoring with two-tier pipeline.

Tier 1: Handwriting transcription (Gemini Flash) — raw LaTeX, runs on every screenshot
Tier 2: Reasoning + feedback (Gemini Flash + thinking) — always runs after transcription
"""

import asyncio
import base64
import json
import os
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from lib.models.tutoring import ReasoningResponse, TutoringSession
from lib.prompts.transcription import build_transcription_prompt
from lib.prompts.reasoning import REASONING_SYSTEM_PROMPT, build_reasoning_prompt

router = APIRouter()

HELP_MIN_INTERVAL = 3  # Minimum interval for help button spam prevention (seconds)


def _create_llm_clients():
    """Create the two LLM clients for transcription and reasoning."""
    from lib.openai_client import LLMClient

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = "https://openrouter.ai/api/v1"

    transcription_client = LLMClient(
        api_key=api_key,
        model="google/gemini-2.5-flash",
        base_url=base_url,
    )
    reasoning_client = LLMClient(
        api_key=api_key,
        model="google/gemini-2.5-flash",
        base_url=base_url,
    )
    return transcription_client, reasoning_client


async def _tier1_transcribe(
    client,
    session: TutoringSession,
    image_bytes: bytes,
    batch_index: int,
    has_erasures: bool = False,
) -> str:
    """Run Tier 1 transcription on a screenshot. Returns raw LaTeX text."""
    prompt = build_transcription_prompt(
        previous_transcript=session.full_transcript,
        has_erasures=has_erasures,
    )

    raw = await asyncio.to_thread(
        client.generate,
        prompt=prompt,
        images=[image_bytes],
        temperature=0.1,
    )

    # Clean up: strip markdown fences if model wrapped output
    delta = raw.strip()
    if delta.startswith("```"):
        # Remove opening fence (```latex, ```tex, ```, etc.)
        first_newline = delta.index("\n") if "\n" in delta else len(delta)
        delta = delta[first_newline + 1:]
    if delta.endswith("```"):
        delta = delta[:-3]
    delta = delta.strip()

    # Update session transcript
    if has_erasures:
        # Erasure mode: the model returned the complete corrected transcript
        session.full_transcript = delta
        session.last_activity = time.time()
    else:
        session.append_transcript(batch_index, delta)

    return delta


async def _tier2_reason(
    client,
    session: TutoringSession,
    subquestion: str | None = None,
) -> ReasoningResponse:
    """Run Tier 2 reasoning. Returns structured assessment."""
    prompt = build_reasoning_prompt(
        problem_text=session.problem_text,
        problem_parts=session.problem_parts,
        course_name=session.course_name,
        full_transcript=session.full_transcript,
        subquestion=subquestion,
        previous_status=session.last_status,
        previous_feedback=session.last_feedback,
    )

    schema = ReasoningResponse.model_json_schema()

    raw = await asyncio.to_thread(
        client.generate,
        prompt=prompt,
        temperature=0.3,
        system_message=REASONING_SYSTEM_PROMPT,
        response_schema=schema,
    )

    response = ReasoningResponse.model_validate_json(raw)

    # Update session state
    session.last_reasoning_time = time.time()
    session.last_status = response.status
    session.last_feedback = response.feedback

    return response


async def _maybe_send_audio(
    websocket: WebSocket,
    response: ReasoningResponse,
) -> None:
    """If intervention is warranted, synthesize TTS and send audio."""
    needs_voice = response.status in ("minor_error", "major_error", "stuck", "completed")

    if needs_voice and response.feedback:
        try:
            from lib.tts import synthesize
            audio_bytes = await synthesize(response.feedback)
            audio_b64 = base64.b64encode(audio_bytes).decode()
            await websocket.send_json({
                "type": "tutor_audio",
                "audio_b64": audio_b64,
                "text": response.feedback,
                "status": response.status,
                "confidence": response.confidence,
            })
        except ImportError:
            print("[Tutor WS] TTS not available, sending text-only feedback")
            await websocket.send_json({
                "type": "tutor_feedback",
                "text": response.feedback,
                "status": response.status,
                "confidence": response.confidence,
            })
        except Exception as e:
            print(f"[Tutor WS] TTS error: {e}, sending text-only feedback")
            await websocket.send_json({
                "type": "tutor_feedback",
                "text": response.feedback,
                "status": response.status,
                "confidence": response.confidence,
            })
    else:
        # on_track or no feedback — text-only (no audio interruption)
        if response.feedback:
            await websocket.send_json({
                "type": "tutor_feedback",
                "text": response.feedback,
                "status": response.status,
                "confidence": response.confidence,
            })


@router.websocket("/ws/tutor")
async def tutor_websocket(websocket: WebSocket):
    await websocket.accept()
    print("[Tutor WS] Client connected")

    transcription_client, reasoning_client = _create_llm_clients()
    session: TutoringSession | None = None
    reasoning_lock = asyncio.Lock()

    async def run_reasoning(subquestion: str | None = None):
        """Run Tier 2 reasoning with lock to prevent overlap."""
        nonlocal session
        if session is None:
            return

        async with reasoning_lock:
            try:
                print("[Tutor WS] Running Tier 2 reasoning")
                response = await _tier2_reason(
                    reasoning_client, session, subquestion
                )
                print(
                    f"[Tutor WS] Reasoning: status={response.status}, "
                    f"confidence={response.confidence:.2f}, "
                    f"feedback={response.feedback!r}"
                )
                await _maybe_send_audio(websocket, response)
            except Exception as e:
                print(f"[Tutor WS] Reasoning error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Reasoning error: {e}",
                })

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "session_start":
                session = TutoringSession(
                    problem_id=msg.get("problem_id", ""),
                    question_number=msg.get("question_number", 0),
                    problem_text=msg.get("problem_text", ""),
                    problem_parts=msg.get("problem_parts", []),
                    course_name=msg.get("course_name", ""),
                )
                session.last_reasoning_time = time.time()
                print(
                    f"[Tutor WS] Session started: Q{session.question_number} "
                    f"({session.course_name})"
                )

            elif msg_type == "session_end":
                print("[Tutor WS] Session ended")
                session = None

            elif msg_type == "screenshot":
                if session is None:
                    continue

                batch_index = msg.get("batch_index", 0)
                image_b64 = msg.get("image", "")
                image_bytes = base64.b64decode(image_b64)
                subquestion = msg.get("subquestion")
                has_erasures = msg.get("has_erasures", False)

                print(
                    f"[Tutor WS] Screenshot: batch={batch_index}, "
                    f"q={msg.get('question_number')}, "
                    f"has_erasures={has_erasures}, "
                    f"size={len(image_bytes)} bytes"
                )

                # Tier 1: Transcription (raw LaTeX)
                try:
                    delta = await _tier1_transcribe(
                        transcription_client, session, image_bytes, batch_index,
                        has_erasures=has_erasures,
                    )
                    await websocket.send_json({
                        "type": "transcription",
                        "batch_index": batch_index,
                        "delta_latex": delta,
                        "full_latex": session.full_transcript,
                    })
                    print(
                        f"[Tutor WS] Transcription: batch={batch_index}, "
                        f"delta={delta[:80]!r}"
                    )
                except Exception as e:
                    print(f"[Tutor WS] Transcription error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                        "batch_index": batch_index,
                    })
                    continue

                # Tier 2: Always reason after transcription
                asyncio.create_task(run_reasoning(subquestion=subquestion))

            elif msg_type == "pause":
                if session is None:
                    continue
                print(
                    f"[Tutor WS] Pause detected: duration={msg.get('duration', 0):.1f}s"
                )

            elif msg_type == "help":
                if session is None:
                    continue
                # Spam guard: ignore rapid help presses
                elapsed = time.time() - session.last_reasoning_time
                if elapsed < HELP_MIN_INTERVAL:
                    continue
                subquestion = msg.get("subquestion")
                print("[Tutor WS] Help requested")
                asyncio.create_task(run_reasoning(subquestion=subquestion))

    except WebSocketDisconnect:
        print("[Tutor WS] Client disconnected")
    except Exception as e:
        print(f"[Tutor WS] Error: {e}")
