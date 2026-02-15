"""Stroke transcription via Llama 4 Maverick on Groq.

Single-stage pipeline: Maverick handles classification, math transcription,
and diagram TikZ generation in one multimodal call.
"""

import base64
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


_PROMPT = """\
Transcribe this image of handwritten strokes.

- "math": text, numbers, math expressions, equations, words, or any written symbols. \
Transcribe as LaTeX (e.g. x^2 + 3x). For plain text use \\text{}.
- "diagram": drawings, graphs, coordinate planes, geometric shapes, circuits, arrows, sketches. \
Set transcription to a brief description. Set tikz to TikZ code (\\begin{tikzpicture}...\\end{tikzpicture}), no preamble.

Crossed-out content (X drawn over writing) should be OMITTED entirely.

Handwriting preferences: "2" not "z", "x" not "×", "1" not "l", "0" not "O", "5" not "S", \
"4" not "11", "\\int" not "int"."""

_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "stroke_transcription",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "content_type": {
                    "type": "string",
                    "enum": ["math", "diagram"],
                },
                "transcription": {
                    "type": "string",
                    "description": "LaTeX for math, brief description for diagrams",
                },
                "tikz": {
                    "type": "string",
                    "description": "TikZ code for diagrams, empty string for math",
                },
            },
            "required": ["content_type", "transcription", "tikz"],
            "additionalProperties": False,
        },
    },
}


def transcribe_strokes_image(image_bytes: bytes, problem_context: str = "") -> dict:
    """Classify and transcribe stroke image. Returns {"content_type": str, "transcription": str, "usage": dict}."""
    client = _get_client()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            response_format=_RESPONSE_SCHEMA,
            max_tokens=512,
        )
    except Exception:
        logger.exception("Maverick transcription call failed")
        return {"content_type": "math", "transcription": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if response.usage:
        usage["prompt_tokens"] = response.usage.prompt_tokens or 0
        usage["completion_tokens"] = response.usage.completion_tokens or 0

    result = response.choices[0].message.parsed or {}
    if not result:
        # Fallback to raw content parsing if parsed is None
        import json
        raw = response.choices[0].message.content or ""
        try:
            result = json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            logger.warning("JSON parse failed: %s", raw[:200])
            return {"content_type": "math", "transcription": "", "usage": usage}

    content_type = result.get("content_type", "math")
    if content_type not in ("math", "diagram"):
        content_type = "math"

    transcription = result.get("transcription", "")

    # For diagrams, prefer tikz if available
    if content_type == "diagram":
        tikz = result.get("tikz", "")
        if tikz:
            transcription = tikz

    return {"content_type": content_type, "transcription": transcription, "usage": usage}
