"""Adaptive coaching via LLM reasoning.

Watches student work in real time and decides whether to speak.
Output is free-form text intended for TTS — no JSON, no LaTeX.
"""

import asyncio
import json
import logging
import os
import time

from openai import OpenAI

from lib.database import get_pool

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_MODEL = "openai/gpt-oss-120b"

# Pricing: GPT-OSS 120B on Groq
_PRICE_INPUT = 0.15  # $/M tokens
_PRICE_OUTPUT = 0.60  # $/M tokens

# Per-session reasoning usage accumulator
_reasoning_usage: dict[str, dict] = {}


def get_reasoning_usage(session_id: str) -> dict:
    return _reasoning_usage.get(session_id, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})


def clear_reasoning_usage(session_id: str) -> None:
    _reasoning_usage.pop(session_id, None)


def _accumulate_usage(session_id: str, usage: dict) -> None:
    if session_id not in _reasoning_usage:
        _reasoning_usage[session_id] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    _reasoning_usage[session_id]["prompt_tokens"] += usage.get("prompt_tokens", 0)
    _reasoning_usage[session_id]["completion_tokens"] += usage.get("completion_tokens", 0)
    _reasoning_usage[session_id]["calls"] += 1


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


_SYSTEM_PROMPT = """\
You are a tutor watching a student work on an assignment on a tablet in real time. \
You see their handwritten work transcribed. You have the answer key. The subject \
could be anything — math, science, history, literature, languages, or something else \
entirely. Adapt your coaching to whatever the student is working on.

Your job is to decide IF and WHAT to say. Your words will be spoken aloud via \
text-to-speech — keep them natural, brief, and conversational. Talk like a patient \
tutor sitting next to them.

WHEN TO STAY SILENT (return empty string):
- The student is making steady progress
- They just started writing and you don't have enough context yet
- They're working through something and haven't made a mistake
- Their approach is valid even if different from the answer key

WHEN TO SPEAK:
- They made an error — point to where, not what the fix is
- They've been stuck (no new writing for a while) — ask what they're thinking or offer a small nudge
- They finished a step correctly — brief encouragement ("good", "that's right", "keep going")
- They're going down a completely wrong path — gently redirect
- They got the final answer — confirm or note if something's off

HOW TO SPEAK:
- Short sentences. 1-3 sentences max.
- No LaTeX, no symbols, no formatting. Speak in plain words a person would say out loud.
- Use the student's own words and notation when referencing their work.
- Never say the answer. Guide them to find it.
- Be warm but not patronizing. No "Great job!" every time.
- Reference specific parts of their work: "Look at your second line — check that part."
"""


async def _assemble_context(session_id: str, page: int) -> str | None:
    """Build the user message with problem, answer key, canvas, and timeline."""
    pool = get_pool()
    if not pool:
        return None

    async with pool.acquire() as conn:
        # Problem context
        ctx_row = await conn.fetchrow(
            """
            SELECT problem_context FROM stroke_logs
            WHERE session_id = $1 AND problem_context != ''
            ORDER BY received_at DESC LIMIT 1
            """,
            session_id,
        )
        problem_context = ctx_row["problem_context"] if ctx_row else ""

        # Answer key — find via problem_context matching a question
        answer_key = ""
        if problem_context:
            # Look for a question whose text matches the problem context
            ak_rows = await conn.fetch(
                """
                SELECT ak.part_label, ak.answer
                FROM answer_keys ak
                JOIN questions q ON ak.question_id = q.id
                WHERE q.text = $1 OR q.label = $1
                ORDER BY ak.id
                """,
                problem_context,
            )
            if ak_rows:
                parts = []
                for r in ak_rows:
                    label = f"({r['part_label']}) " if r["part_label"] else ""
                    parts.append(f"{label}{r['answer']}")
                answer_key = "\n".join(parts)

        # Cluster transcriptions in reading order
        cluster_rows = await conn.fetch(
            """
            SELECT cluster_label, transcription, content_type, centroid_y
            FROM clusters
            WHERE session_id = $1 AND page = $2 AND transcription != ''
            ORDER BY centroid_y ASC
            """,
            session_id, page,
        )

        # Recent stroke timeline (last 10 events)
        timeline_rows = await conn.fetch(
            """
            SELECT event_type, message, received_at,
                   jsonb_array_length(strokes) AS stroke_count
            FROM stroke_logs
            WHERE session_id = $1 AND page = $2
            ORDER BY received_at DESC
            LIMIT 10
            """,
            session_id, page,
        )

    if not cluster_rows:
        return None  # Nothing on canvas yet

    # Format context
    sections = []

    if problem_context:
        sections.append(f"PROBLEM:\n{problem_context}")

    if answer_key:
        sections.append(f"ANSWER KEY (for your reference only — never reveal):\n{answer_key}")

    canvas_lines = []
    for r in cluster_rows:
        label = r["cluster_label"]
        ctype = r["content_type"]
        text = r["transcription"]
        if ctype == "diagram":
            canvas_lines.append(f"[Cluster {label}]: [diagram: {text}]")
        else:
            canvas_lines.append(f"[Cluster {label}]: {text}")
    sections.append("CANVAS (student's current work, top to bottom):\n" + "\n".join(canvas_lines))

    if timeline_rows:
        timeline_lines = []
        for r in reversed(timeline_rows):  # chronological order
            ts = r["received_at"].strftime("%H:%M:%S")
            evt = r["event_type"]
            if evt == "draw":
                timeline_lines.append(f"{ts} — drew {r['stroke_count']} stroke(s)")
            elif evt == "erase":
                timeline_lines.append(f"{ts} — erased canvas")
            elif evt == "system" and r["message"]:
                timeline_lines.append(f"{ts} — {r['message']}")
        if timeline_lines:
            sections.append("RECENT ACTIVITY:\n" + "\n".join(timeline_lines))

    return "\n\n".join(sections)


async def run_reasoning(session_id: str, page: int) -> dict:
    """Run reasoning model and return action + message.

    Returns:
        {"action": "silent" | "speak", "message": "...", "usage": {...}}
    """
    t0 = time.perf_counter()

    context = await _assemble_context(session_id, page)
    if context is None:
        return {"action": "silent", "message": "", "usage": {}}

    client = _get_client()

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            max_tokens=256,
        )
    except Exception:
        logger.exception("Reasoning LLM call failed")
        return {"action": "silent", "message": "", "usage": {}}

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if response.usage:
        usage["prompt_tokens"] = response.usage.prompt_tokens or 0
        usage["completion_tokens"] = response.usage.completion_tokens or 0

    _accumulate_usage(session_id, usage)

    message = (response.choices[0].message.content or "").strip()
    action = "silent" if not message else "speak"

    elapsed = time.perf_counter() - t0
    print(f"[reasoning] session={session_id} page={page} action={action} "
          f"tokens={usage['prompt_tokens']}+{usage['completion_tokens']} "
          f"time={elapsed:.2f}s")
    if message:
        print(f"[reasoning] message: {message}")

    # Estimate cost — GPT-OSS 120B on Groq
    estimated_cost = (usage["prompt_tokens"] * _PRICE_INPUT + usage["completion_tokens"] * _PRICE_OUTPUT) / 1_000_000

    # Store to database
    pool = get_pool()
    if pool:
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO reasoning_logs
                        (session_id, page, context, action, message,
                         prompt_tokens, completion_tokens, estimated_cost)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    session_id, page, context, action, message or None,
                    usage["prompt_tokens"], usage["completion_tokens"], estimated_cost,
                )
        except Exception as exc:
            logger.error("Failed to insert reasoning log: %s", exc)

    return {"action": action, "message": message, "usage": usage}
