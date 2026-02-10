"""WebSocket endpoint for real-time AI tutoring with two-tier pipeline.

Tier 1: Handwriting transcription (Gemini Flash) — runs on every screenshot
Tier 2: Reasoning + feedback (Gemini Flash + thinking) — triggered by pauses, intervals, or help
"""

import asyncio
import base64
import json
import os
import re
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from lib.models.tutoring import ReasoningResponse, TranscriptionResponse, TutoringSession
from lib.prompts.transcription import build_transcription_prompt
from lib.prompts.reasoning import REASONING_SYSTEM_PROMPT, build_reasoning_prompt

router = APIRouter()

# Trigger parameters
MIN_REASONING_INTERVAL = 8     # Floor: never reason more often than this (seconds)
MAX_REASONING_INTERVAL = 30    # Ceiling: always reason after this if new work exists (seconds)
MIN_BATCHES_FOR_REASONING = 2  # Need at least 2 transcription batches before first reasoning
HELP_MIN_INTERVAL = 3          # Minimum interval for help button spam prevention (seconds)


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
) -> TranscriptionResponse:
    """Run Tier 1 transcription on a screenshot. Returns structured response."""
    prompt = build_transcription_prompt(
        previous_transcript=session.full_transcript,
        problem_text=session.problem_text,
        course_name=session.course_name,
        batches_since_check=session.batches_since_reasoning,
    )

    raw = await asyncio.to_thread(
        client.generate,
        prompt=prompt,
        images=[image_bytes],
        temperature=0.1,
    )

    # Parse JSON from raw text (no response_schema — it hurts Gemini vision via OpenRouter)
    try:
        result = TranscriptionResponse.model_validate_json(raw)
    except Exception:
        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        result = TranscriptionResponse.model_validate_json(cleaned)

    session.append_transcript(batch_index, result.delta_latex)
    return result


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
    session.batches_since_reasoning = 0

    return response


def _should_trigger_reasoning(session: TutoringSession, force: bool = False) -> bool:
    """Check if Tier 2 reasoning should fire."""
    now = time.time()
    elapsed = now - session.last_reasoning_time

    if force:
        return elapsed >= HELP_MIN_INTERVAL

    # Need minimum batches
    if session.batches_since_reasoning < MIN_BATCHES_FOR_REASONING:
        return False

    # Floor check
    if elapsed < MIN_REASONING_INTERVAL:
        return False

    return True


def _check_max_interval(session: TutoringSession) -> bool:
    """Check if the max interval ceiling has been exceeded."""
    if session.batches_since_reasoning < MIN_BATCHES_FOR_REASONING:
        return False
    elapsed = time.time() - session.last_reasoning_time
    return elapsed >= MAX_REASONING_INTERVAL


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
            # TTS not available — fall back to text-only
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

    async def run_reasoning(subquestion: str | None = None, force: bool = False):
        """Run Tier 2 reasoning if conditions are met, with lock to prevent overlap."""
        nonlocal session
        if session is None:
            return
        if not _should_trigger_reasoning(session, force=force):
            return

        async with reasoning_lock:
            # Re-check after acquiring lock (another task may have run reasoning)
            if not _should_trigger_reasoning(session, force=force):
                return

            try:
                print(f"[Tutor WS] Running Tier 2 reasoning (force={force})")
                response = await _tier2_reason(
                    reasoning_client, session, subquestion
                )
                print(
                    f"[Tutor WS] Reasoning result: status={response.status}, "
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

                print(
                    f"[Tutor WS] Screenshot: batch={batch_index}, "
                    f"q={msg.get('question_number')}, "
                    f"size={len(image_bytes)} bytes"
                )

                # Tier 1: Transcription (structured output)
                try:
                    result = await _tier1_transcribe(
                        transcription_client, session, image_bytes, batch_index
                    )
                    await websocket.send_json({
                        "type": "transcription",
                        "batch_index": batch_index,
                        "delta_latex": result.delta_latex,
                        "full_latex": session.full_transcript,
                    })
                    print(
                        f"[Tutor WS] Transcription: batch={batch_index}, "
                        f"should_check={result.should_check}, "
                        f"delta={result.delta_latex[:80]!r}"
                    )
                except Exception as e:
                    print(f"[Tutor WS] Transcription error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                        "batch_index": batch_index,
                    })
                    continue

                # Fire reasoning if model flagged should_check or max interval exceeded
                if result.should_check or _check_max_interval(session):
                    asyncio.create_task(run_reasoning(subquestion=subquestion))

            elif msg_type == "pause":
                if session is None:
                    continue
                subquestion = msg.get("subquestion")
                print(
                    f"[Tutor WS] Pause detected: duration={msg.get('duration', 0):.1f}s"
                )
                asyncio.create_task(run_reasoning(subquestion=subquestion))

            elif msg_type == "help":
                if session is None:
                    continue
                subquestion = msg.get("subquestion")
                print("[Tutor WS] Help requested")
                asyncio.create_task(
                    run_reasoning(subquestion=subquestion, force=True)
                )

    except WebSocketDisconnect:
        print("[Tutor WS] Client disconnected")
    except Exception as e:
        print(f"[Tutor WS] Error: {e}")
