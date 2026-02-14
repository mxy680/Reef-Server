"""Groq Llama 4 Maverick vision client for LaTeX transcription."""

import base64
import os

from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


def transcribe_strokes_image(image_bytes: bytes) -> str:
    """Send stroke image to Groq Llama 4 Maverick, return LaTeX transcription."""
    client = _get_client()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe this handwritten math into LaTeX. Return only the LaTeX expression, no explanation or wrapping.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()
