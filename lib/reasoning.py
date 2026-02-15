"""Adaptive coaching via LLM reasoning.

Watches student work in real time and decides whether to speak.
Output is free-form text intended for TTS — no JSON, no LaTeX.
"""

import asyncio
import json
import logging
import os
import re
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

CRITICAL: Default to SILENCE. Speaking interrupts the student's flow. Only speak \
when you have something genuinely useful to say. An empty response is always better \
than a mediocre one.

WHEN TO STAY SILENT (return empty string):
- The student is making progress — even slowly. Do not interrupt.
- They just started writing and you don't have enough context yet.
- They're working through something correctly. Do not narrate their progress.
- Their approach is valid even if different from the answer key.
- They're between steps. Pausing to think is normal — do not fill the silence.
- You already spoke recently. Do not pile on.

WHEN TO SPEAK (the ONLY reasons to break silence):
1. Something they WROTE is wrong — a specific, concrete error in their work.
2. They're going down a completely wrong path — gently redirect.
3. They got the final answer wrong — note something's off without giving the answer.
4. CORRECTED MISTAKE — You previously pointed out a mistake (check YOUR PREVIOUS \
   MESSAGES below), and the student's canvas now shows they fixed it. You MUST \
   acknowledge this with a brief word like "that's it", "yes", or "there you go". \
   This is important — the student needs to know their correction was right.

INCOMPLETE WORK IS NOT AN ERROR:
- If they've only written part of a solution, they're probably still working on it.
- Missing steps, missing terms, or unfinished expressions are NOT mistakes — they just \
  haven't gotten there yet. Do not point out things they haven't written.
- Only flag what IS written, never what ISN'T written yet.

YOUR PREVIOUS MESSAGES:
- You will see what you already told the student below. Do not repeat yourself.
- Do not say the same thing in different words.
- If you already pointed out an error and the student hasn't fixed it yet, stay silent — \
  they heard you and are working on it.
- If you pointed out an error and the student HAS now fixed it, acknowledge it (rule 4 above).

DO NOT SPEAK just to:
- Encourage or praise unprompted (only after a corrected mistake per rule 4)
- Confirm correct intermediate steps that had no prior error
- Summarize what they wrote
- Fill silence while they're thinking
- Point out missing terms, steps, or parts they haven't written yet
- Repeat something you already said

HOW TO SPEAK (when you must):
- 1-2 sentences max. Be precise about what's wrong.
- No LaTeX, no symbols, no formatting. Plain spoken words.
- Never say the answer. Point to the error, not the fix.
- Reference specific parts of their work: "Check your second line."
"""


def _trigrams(text: str) -> set[str]:
    """Extract character trigrams from text (for fuzzy matching)."""
    s = re.sub(r'\s+', ' ', text.strip().lower())
    return {s[i:i+3] for i in range(max(0, len(s) - 2))}


def _trigram_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity of character trigrams between two strings."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


async def _find_matching_question(conn, canvas_text: str, session_id: str | None = None) -> dict | None:
    """Try to find the question the student is working on by matching canvas content against DB questions.

    First tries exact label match (e.g. "Problem 4" in the text matches questions.label).
    Falls back to trigram similarity for canvas transcription text.
    If session_id is provided and a match is found, caches the result in session_question_cache.
    Returns {"id": int, "text": str, "label": str} or None.
    """
    if not canvas_text.strip():
        return None

    # Get all questions that have answer keys (from most recent document first)
    rows = await conn.fetch(
        """
        SELECT DISTINCT q.id, q.text, q.label, d.id AS doc_id, d.filename AS doc_filename
        FROM questions q
        JOIN documents d ON q.document_id = d.id
        JOIN answer_keys ak ON ak.question_id = q.id
        ORDER BY d.id DESC, q.id ASC
        """,
    )
    if not rows:
        return None

    matched_row = None

    # Strategy 1: Extract "Problem N" from the text and match on label exactly
    label_match = re.match(r'(Problem\s+\d+)', canvas_text.strip(), re.IGNORECASE)
    if label_match:
        target_label = label_match.group(1)
        for r in rows:
            if r["label"] and r["label"].lower() == target_label.lower():
                matched_row = r
                print(f"[reasoning] exact label match: {target_label!r} -> Q{r['id']}")
                break

    # Strategy 2: Trigram similarity fallback (for canvas transcription text without labels)
    if not matched_row:
        best_score = 0.0
        best_row = None
        for r in rows:
            score = _trigram_similarity(canvas_text, r["text"] or "")
            if score > best_score:
                best_score = score
                best_row = r
        if best_row and best_score >= 0.05:
            matched_row = best_row
            print(f"[reasoning] trigram match: Q{best_row['id']} "
                  f"label={best_row['label']} similarity={best_score:.3f}")

    if matched_row:
        # Cache the match for this session
        if session_id:
            try:
                await conn.execute(
                    """
                    INSERT INTO session_question_cache (session_id, question_id, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (session_id) DO UPDATE SET question_id = $2, updated_at = NOW()
                    """,
                    session_id, matched_row["id"],
                )
            except Exception as exc:
                logger.error("Failed to cache question match: %s", exc)
        return {"id": matched_row["id"], "text": matched_row["text"], "label": matched_row["label"], "doc_filename": matched_row["doc_filename"]}

    return None


async def _get_cached_question(conn, session_id: str) -> dict | None:
    """Retrieve the cached question match for a session, if any."""
    row = await conn.fetchrow(
        """
        SELECT q.id, q.text, q.label, d.filename AS doc_filename,
               sqc.document_name AS ios_doc_name
        FROM session_question_cache sqc
        JOIN questions q ON q.id = sqc.question_id
        JOIN documents d ON q.document_id = d.id
        WHERE sqc.session_id = $1
        """,
        session_id,
    )
    if row:
        print(f"[reasoning] using cached question match: id={row['id']} label={row['label']}")
        # Prefer the iOS-provided document name over the DB filename
        doc_name = row["ios_doc_name"] or row["doc_filename"]
        return {"id": row["id"], "text": row["text"], "label": row["label"], "doc_filename": doc_name}
    return None


async def _assemble_context(session_id: str, page: int) -> str | None:
    """Build the user message with problem, answer key, canvas, and timeline."""
    pool = get_pool()
    if not pool:
        return None

    async with pool.acquire() as conn:
        # Problem context from iOS app (sent as a "context" WebSocket message)
        ctx_row = await conn.fetchrow(
            """
            SELECT problem_context FROM stroke_logs
            WHERE session_id = $1 AND problem_context != ''
            ORDER BY received_at DESC LIMIT 1
            """,
            session_id,
        )
        problem_context = ctx_row["problem_context"] if ctx_row else ""

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

        if not cluster_rows:
            return None  # Nothing on canvas yet

        # Build canvas text for question matching
        canvas_text = " ".join(r["transcription"] for r in cluster_rows)

        # Find the relevant question and answer key
        question_text = problem_context  # Prefer iOS-provided context
        answer_key = ""

        # Try to match against DB questions (works even without problem_context)
        matched_q = await _find_matching_question(conn, canvas_text, session_id=session_id)
        if matched_q:
            # Use matched question text if iOS didn't provide context
            if not question_text:
                question_text = matched_q["text"]
                print(f"[reasoning] no problem_context from iOS, using matched question: {matched_q['label']}")

            # Look up answer key by question_id
            ak_rows = await conn.fetch(
                """
                SELECT part_label, answer
                FROM answer_keys
                WHERE question_id = $1
                ORDER BY id
                """,
                matched_q["id"],
            )
            if ak_rows:
                parts = []
                for r in ak_rows:
                    label = f"({r['part_label']}) " if r["part_label"] else ""
                    parts.append(f"{label}{r['answer']}")
                answer_key = "\n".join(parts)
                print(f"[reasoning] found answer key for question {matched_q['id']} ({len(ak_rows)} parts)")
        elif problem_context:
            print(f"[reasoning] WARNING: problem_context set but no matching question found")
        else:
            print(f"[reasoning] WARNING: no problem_context and no matching question for canvas: {canvas_text[:80]}...")

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

        # Previous reasoning messages (what you already said)
        reasoning_rows = await conn.fetch(
            """
            SELECT action, message, created_at
            FROM reasoning_logs
            WHERE session_id = $1 AND page = $2 AND action = 'speak'
            ORDER BY created_at DESC
            LIMIT 10
            """,
            session_id, page,
        )

    # Format context
    sections = []

    if question_text:
        sections.append(f"PROBLEM:\n{question_text}")

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

    if reasoning_rows:
        prev_lines = []
        for r in reversed(reasoning_rows):  # chronological order
            ts = r["created_at"].strftime("%H:%M:%S")
            prev_lines.append(f"{ts} — You said: \"{r['message']}\"")
        sections.append("YOUR PREVIOUS MESSAGES (what you already told the student):\n" + "\n".join(prev_lines))

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
    except Exception as exc:
        logger.exception("Reasoning LLM call failed")
        error_msg = f"[ERROR] {type(exc).__name__}: {exc}"
        print(f"[reasoning] LLM call failed: {error_msg}")
        # Log the error to DB so it shows up in the dashboard
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
                        session_id, page, context, "error", error_msg, 0, 0, 0.0,
                    )
            except Exception:
                pass
        return {"action": "error", "message": error_msg, "usage": {}}

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
