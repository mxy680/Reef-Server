"""Prompt templates for Tier 1 handwriting transcription."""


def build_transcription_prompt(
    previous_transcript: str,
    has_erasures: bool = False,
) -> str:
    """Build the transcription prompt.

    The transcription model has ONE job: convert handwriting to LaTeX.
    No problem context, no reasoning, no JSON — just raw LaTeX text.

    Args:
        previous_transcript: Last ~2000 chars of existing transcript for continuity.
        has_erasures: Whether the current image contains erased strokes (red).

    Returns:
        The complete prompt string for the transcription model.
    """
    context_section = ""
    if previous_transcript:
        trimmed = previous_transcript[-2000:]
        context_section = (
            f"\n\nPrevious transcription (continue from here):\n{trimmed}"
        )

    if has_erasures:
        return (
            "You are a handwriting-to-LaTeX transcription model.\n\n"
            "The image shows the student's full canvas.\n"
            "Colors:\n"
            "- GRAY strokes = already transcribed\n"
            "- BLACK strokes = new work\n"
            "- RED strokes = erased by the student\n\n"
            "The student has ERASED some work (red strokes). "
            "Output the COMPLETE corrected transcript: take the previous transcription, "
            "REMOVE the content corresponding to the red strokes, and INTEGRATE any new "
            "black strokes. Output ONLY the corrected full transcript as LaTeX, nothing else."
            f"{context_section}"
        )

    return (
        "You are a handwriting-to-LaTeX transcription model.\n\n"
        "The image shows the student's full canvas.\n"
        "Colors:\n"
        "- GRAY strokes = already transcribed. Do NOT re-transcribe.\n"
        "- BLACK strokes = new work. Transcribe these.\n\n"
        "Output ONLY the LaTeX for the new (black) strokes. "
        "Do not repeat content from the previous transcription or gray strokes. "
        "Use inline $...$ for math expressions. "
        "If the new strokes continue a previous line, output only the new part. "
        "Output raw LaTeX text only — no JSON, no markdown, no explanation."
        f"{context_section}"
    )
