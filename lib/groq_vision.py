"""Groq vision client for stroke transcription.

Two-stage pipeline for diagrams:
  Stage 1 (Maverick, vision): classify + describe the image
  Stage 2 (Llama 3.3 70B, text-only): generate TikZ from the description
Math/text goes through stage 1 only.
"""

import base64
import json
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

_MAVERICK_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
_CODEGEN_MODEL = "llama-3.3-70b-versatile"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


# --- Stage 1: Classify + Describe (Maverick vision) ---

_DESCRIBE_PROMPT = """\
Look at this image of handwritten strokes. Classify it and either transcribe or describe it.

Step 1: Decide the content type.
- "math" — text, numbers, math expressions, equations, words, or any written symbols
- "diagram" — drawings, graphs, coordinate planes, geometric shapes, circuits, arrows, sketches, or any non-text visual

Step 2: Based on the type, fill in the appropriate fields.

If "math":
- Set "transcription" to the LaTeX (e.g. x^2 + 3x). For plain text use \\text{} (e.g. \\text{hello}).
- Leave "description" and "elements" empty.

If "diagram":
- Set "description" to a detailed natural-language description of the diagram.
- Set "elements" to a list of diagram elements, each with a "type" and relevant properties.
  Element types: axis, curve, line, point, shape, label, arrow, region, or other.
  Include coordinates, labels, equations, styles (solid/dashed/dotted), and any other relevant details.
- Leave "transcription" empty.

Common handwriting ambiguities — prefer these interpretations:
- "2" not "z", "x" not "×", "1" not "l", "0" not "O", "5" not "S"

Respond with JSON only, no other text:
{"content_type": "math" or "diagram", "transcription": "...", "description": "...", "elements": [...]}"""

_CONTEXT_TEMPLATE = """

The student is working on this problem:
{problem_context}

Use this context to disambiguate characters, notation, and diagram meaning."""


def _describe_image(client: OpenAI, data_url: str, problem_context: str = "") -> dict:
    """Stage 1: Classify and describe the stroke image using Maverick vision."""
    prompt = _DESCRIBE_PROMPT
    if problem_context:
        prompt += _CONTEXT_TEMPLATE.format(problem_context=problem_context)

    response = client.chat.completions.create(
        model=_MAVERICK_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=256,
    )
    usage = {}
    if response.usage:
        usage = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
    raw = response.choices[0].message.content.strip()
    return json.loads(raw), usage


# --- Stage 2: Generate TikZ (Llama 3.3 70B, text-only) ---

_TIKZ_PROMPT = """\
Generate TikZ code for the following diagram.

Description: {description}

Elements:
{elements}

Rules:
- Output ONLY the TikZ code, starting with \\begin{{tikzpicture}} and ending with \\end{{tikzpicture}}.
- No preamble, no \\documentclass, no \\usepackage, no explanation.
- Use accurate coordinates and labels from the element list.
- Use appropriate TikZ libraries (arrows, shapes, etc.) via \\usetikzlibrary inside the tikzpicture if needed."""


def _generate_tikz(client: OpenAI, description: str, elements: list) -> str:
    """Stage 2: Generate TikZ from a structured diagram description (text-only)."""
    elements_str = json.dumps(elements, indent=2) if elements else "[]"
    prompt = _TIKZ_PROMPT.format(description=description, elements=elements_str)

    response = client.chat.completions.create(
        model=_CODEGEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    usage = {}
    if response.usage:
        usage = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    return raw, usage


# --- Public API ---


def transcribe_strokes_image(image_bytes: bytes, problem_context: str = "") -> dict:
    """Classify and transcribe stroke image. Returns {"content_type": str, "transcription": str}."""
    client = _get_client()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    # Stage 1: classify + describe
    try:
        stage1, stage1_usage = _describe_image(client, data_url, problem_context)
        total_usage["prompt_tokens"] += stage1_usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += stage1_usage.get("completion_tokens", 0)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Stage 1 JSON parse failed, falling back to math")
        return {"content_type": "math", "transcription": "", "usage": total_usage}

    content_type = stage1.get("content_type", "math")
    if content_type not in ("math", "diagram"):
        content_type = "math"

    # Math: return transcription directly from stage 1
    if content_type == "math":
        return {
            "content_type": "math",
            "transcription": stage1.get("transcription", ""),
            "usage": total_usage,
        }

    # Diagram: stage 2 — generate TikZ from description
    description = stage1.get("description", "")
    elements = stage1.get("elements", [])
    logger.info("Stage 1 diagram description: %s", description)

    try:
        tikz, stage2_usage = _generate_tikz(client, description, elements)
        total_usage["prompt_tokens"] += stage2_usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += stage2_usage.get("completion_tokens", 0)
    except Exception:
        logger.exception("Stage 2 TikZ generation failed, returning description")
        tikz = f"% TikZ generation failed\n% Description: {description}"

    return {"content_type": "diagram", "transcription": tikz, "usage": total_usage}
