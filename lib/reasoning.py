"""Reasoning model client — Groq Gemini 2.5 Flash with structured output.

Called after transcription completes, only if no new strokes arrived
during the transcription call. Prompt caching is automatic on Groq —
consistent system prompt prefix gets cached at $0.075/M.
"""

import asyncio
import json
import logging
import os

from openai import OpenAI

from lib.database import get_pool

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_MODEL = "google/gemini-2.5-flash"

# Pricing: Gemini 2.5 Flash on OpenRouter
_PRICE_INPUT = 0.15  # $/M tokens
_PRICE_INPUT_CACHED = 0.0375  # $/M tokens (cached)
_PRICE_OUTPUT = 0.60  # $/M tokens


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        _client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return _client


SYSTEM_PROMPT = """You are Reef's AI tutor analyzing a college STEM student's handwritten work on iPad. You receive a canvas state (transcribed LaTeX clusters), a timeline of stroke activity, and the answer key for the problem. Your job is to decide WHETHER to respond and, if so, HOW. Your message will be spoken aloud via Kokoro TTS during the student's natural pause.

## Answer Key Usage

You have the correct solution. Use it to:
- Accurately detect errors in the student's work by comparing against the correct steps
- Identify exactly WHERE the student diverged from the correct approach
- Distinguish real errors from alternative valid approaches (some problems have multiple solution paths)

NEVER reveal the answer or any part of the solution directly. The answer key is for YOUR reference only.

## Core Principle: Silence Is the Default

Most pauses are productive. Students learn more from struggle than from explanation. Do NOT intervene just because you detect an error or a pause. Intervene ONLY when struggle has become unproductive.

Signs of UNPRODUCTIVE struggle (intervene):
- Same error repeated 3+ times with no variation in approach
- Long inactivity after a clear dead end (no partial work toward next step)
- Work trailing off mid-step (started an expression, stopped, no continuation)
- Regressive moves: crossing out correct work, replacing it with worse attempts

Signs of PRODUCTIVE struggle (stay silent):
- Student paused but has a plausible partial step on the canvas
- Student is trying different approaches, even if incorrect
- Student just made an error but hasn't had time to self-correct
- Steady writing pace with brief pauses between steps

If the student is making progress, even slowly, say nothing. Return:
{"action": "none"}

## When You Do Intervene: Graduated Escalation

Always use the MINIMUM effective intervention. Never skip levels.

Level 1 — FLAG: Signal that something in a specific step deserves a second look. No explanation.
  Example: "Take another look at your second step."

Level 2 — QUESTION: A single Socratic question targeting the exact misconception.
  Example: "What happens to the constant of integration when you split the integral?"

Level 3 — HINT: Name the relevant concept or principle without applying it.
  Example: "Think about the power rule for integration here. What's the antiderivative of 2x?"

Level 4 — EXPLAIN: Walk through the reasoning for the specific step that's wrong. Never solve the actual problem.

Start at Level 1. Only escalate if the student has already seen a lower-level intervention for the SAME issue and remained stuck.

## Feedback Rules

- Be brief. 1-2 sentences max for Levels 1-3. 3-4 sentences max for Level 4.
- Address the STEP, not the student. Say "this step" not "you."
- Process over outcome. Say "check the algebra in the second term" not "wrong answer."
- When an error reflects a conceptual misconception, name the misconception and create cognitive conflict.
- Never show the final answer. Never write out a complete solution.

## Distinguishing Error Types

PROCEDURAL SLIP: Sign error, dropped term, arithmetic mistake. Use Level 1-2.
CONCEPTUAL MISCONCEPTION: Wrong mental model. Use Level 2-3 with cognitive conflict.
STRATEGIC ERROR: Valid but inefficient approach. Only flag if it prevents solving the problem.

## Voice Formatting (Kokoro TTS)

Your message is spoken aloud, not displayed as text. Write for the ear, not the eye.

- Write in natural, conversational spoken English. No bullet points, no markdown.
- Keep sentences short and direct.
- Never use LaTeX, symbols, or formatting that only makes sense visually.
- Speak math naturally: "x squared" not "x^2", "x over 2" not "\\frac{x}{2}".
- Greek letters: use pronunciation overrides: [theta](/θˈiːtə/), [alpha](/ˈælfə/).
- Never reference cluster IDs. Say "your second line" or describe the content.
- Never spell out equations symbol by symbol."""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "tutor_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["none", "feedback"]},
                "level": {"anyOf": [{"type": "integer", "enum": [1, 2, 3, 4]}, {"type": "null"}]},
                "target": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "error_type": {"anyOf": [{"type": "string", "enum": ["procedural", "conceptual", "strategic"]}, {"type": "null"}]},
                "delay_ms": {"type": "integer"},
                "message": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["action", "level", "target", "error_type", "delay_ms", "message"],
            "additionalProperties": False,
        },
    },
}


async def _assemble_context(session_id: str, page: int) -> str:
    """Build the user context message from DB state."""
    pool = get_pool()
    if not pool:
        return ""

    sections: list[str] = []

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
        if ctx_row and ctx_row["problem_context"]:
            sections.append(f"## Problem\n\n{ctx_row['problem_context']}")

            # Look up answer key for this problem
            question_row = await conn.fetchrow(
                """
                SELECT q.id FROM questions q
                WHERE q.text != '' AND position(q.text in $1) > 0
                LIMIT 1
                """,
                ctx_row["problem_context"],
            )
            if question_row:
                answer_keys = await conn.fetch(
                    """
                    SELECT part_label, answer FROM answer_keys
                    WHERE question_id = $1
                    ORDER BY part_label NULLS FIRST
                    """,
                    question_row["id"],
                )
                if answer_keys:
                    ak_lines = []
                    for ak in answer_keys:
                        label = f"({ak['part_label']}) " if ak["part_label"] else ""
                        ak_lines.append(f"{label}{ak['answer']}")
                    sections.append("## Answer Key\n\n" + "\n\n".join(ak_lines))

        # Canvas state: cluster transcriptions in reading order
        clusters = await conn.fetch(
            """
            SELECT cluster_label, transcription, content_type FROM clusters
            WHERE session_id = $1 AND transcription != ''
            ORDER BY centroid_y ASC
            """,
            session_id,
        )
        if clusters:
            lines = []
            for c in clusters:
                label = c["cluster_label"]
                ctype = c["content_type"]
                tx = c["transcription"]
                if ctype == "diagram":
                    lines.append(f"[C{label}] (diagram): [diagram]")
                else:
                    lines.append(f"[C{label}] ({ctype}): {tx}")
            sections.append("## Canvas State\n\n" + "\n".join(lines))

        # Build cluster transcription lookup for timeline references
        cluster_tx: dict[int, str] = {}
        if clusters:
            for c in clusters:
                ctype = c["content_type"]
                tx = c["transcription"]
                if ctype == "diagram":
                    cluster_tx[c["cluster_label"]] = "[diagram]"
                else:
                    cluster_tx[c["cluster_label"]] = tx

        # Timeline: recent stroke activity with cluster context
        logs = await conn.fetch(
            """
            SELECT event_type, message, received_at, strokes, deleted_count, cluster_labels
            FROM stroke_logs
            WHERE session_id = $1
            ORDER BY received_at DESC
            LIMIT 30
            """,
            session_id,
        )
        if logs:
            timeline_lines = []
            for log in reversed(logs):
                time_str = log["received_at"].strftime("%H:%M:%S")
                etype = log["event_type"]
                if etype == "system":
                    if log["message"] == "session started":
                        continue
                    timeline_lines.append(f"{time_str}  [{etype}] {log['message']}")
                elif etype == "voice":
                    timeline_lines.append(f"{time_str}  [{etype}] {log['message']}")
                elif etype == "erase":
                    timeline_lines.append(f"{time_str}  [erase] deleted {log['deleted_count']} strokes")
                else:
                    # Show which clusters were written to and their current transcriptions
                    labels_raw = log["cluster_labels"]
                    labels = labels_raw if isinstance(labels_raw, list) else json.loads(labels_raw)
                    affected = sorted(set(labels))
                    parts = []
                    for cl in affected:
                        tx = cluster_tx.get(cl, "")
                        parts.append(f"C{cl}: {tx}" if tx else f"C{cl}")
                    timeline_lines.append(f"{time_str}  [draw] {'; '.join(parts)}")
            if timeline_lines:
                sections.append("## Timeline\n\n" + "\n".join(timeline_lines))

    return "\n\n".join(sections)


def _estimate_cost(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> float:
    """Estimate cost in USD."""
    uncached_input = prompt_tokens - cached_tokens
    cost = (
        uncached_input * _PRICE_INPUT
        + cached_tokens * _PRICE_INPUT_CACHED
        + completion_tokens * _PRICE_OUTPUT
    ) / 1_000_000
    return round(cost, 6)


async def run_reasoning(session_id: str, page: int) -> dict | None:
    """Assemble prompt, call Gemini 2.5 Flash, store result."""
    context = await _assemble_context(session_id, page)
    if not context:
        logger.info("[reasoning] No context for session %s, skipping", session_id)
        return None

    client = _get_client()

    try:
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                response_format=RESPONSE_SCHEMA,
                max_tokens=512,
            )
        )
    except Exception:
        logger.exception("[reasoning] Gemini 2.5 Flash call failed")
        return None

    # Parse usage
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    if response.usage:
        prompt_tokens = response.usage.prompt_tokens or 0
        completion_tokens = response.usage.completion_tokens or 0
        # Groq reports cached tokens in prompt_tokens_details
        if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
            cached_tokens = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0

    estimated_cost = _estimate_cost(prompt_tokens, completion_tokens, cached_tokens)

    # Parse response
    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[reasoning] Failed to parse response: %s", raw[:200])
        return None

    action = result.get("action", "none")
    level = result.get("level")
    target = result.get("target")
    error_type = result.get("error_type")
    delay_ms = result.get("delay_ms", 0)
    message = result.get("message")

    logger.info(
        "[reasoning] session=%s action=%s level=%s target=%s msg=%s tokens=%d+%d (cached=%d) cost=$%.6f",
        session_id, action, level, target,
        (message or "")[:60], prompt_tokens, completion_tokens, cached_tokens, estimated_cost,
    )
    print(
        f"[reasoning] session={session_id} action={action} level={level} "
        f"target={target} msg={(message or '')[:60]} "
        f"tokens={prompt_tokens}+{completion_tokens} (cached={cached_tokens}) cost=${estimated_cost:.6f}"
    )

    # Store to DB
    pool = get_pool()
    if pool:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO reasoning_logs
                    (session_id, page, context, action, level, target, error_type,
                     delay_ms, message, prompt_tokens, completion_tokens, cached_tokens, estimated_cost)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                session_id, page, context, action, level, target, error_type,
                delay_ms, message, prompt_tokens, completion_tokens, cached_tokens, estimated_cost,
            )

    return {
        "action": action,
        "level": level,
        "target": target,
        "error_type": error_type,
        "delay_ms": delay_ms,
        "message": message,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "estimated_cost": estimated_cost,
    }


