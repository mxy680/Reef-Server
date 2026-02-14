"""Groq Whisper transcription client."""

import os
import tempfile
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


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using Groq Whisper.

    Args:
        audio_bytes: Raw WAV audio data.

    Returns:
        Transcribed text string.
    """
    client = _get_client()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.seek(0)
        result = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=tmp,
        )
    return result.text
