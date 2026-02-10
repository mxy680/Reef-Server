"""Prompt templates for Tier 1 handwriting transcription."""


def build_transcription_prompt(
    previous_transcript: str,
    problem_text: str = "",
    course_name: str = "",
    batches_since_check: int = 0,
) -> str:
    """Build the transcription prompt with rolling context.

    Args:
        previous_transcript: Last ~2000 chars of existing transcript for continuity.
        problem_text: The problem the student is working on.
        course_name: Name of the course (e.g. "AP Calculus BC").
        batches_since_check: Number of screenshot batches since the last reasoning check.

    Returns:
        The complete prompt string for the transcription model.
    """
    context_section = ""
    if previous_transcript:
        # Trim to last 2000 chars for context window efficiency
        trimmed = previous_transcript[-2000:]
        context_section = (
            f"\n\nPrevious transcription (continue from here):\n{trimmed}"
        )

    problem_section = ""
    if problem_text or course_name:
        parts = []
        if course_name:
            parts.append(f"Course: {course_name}")
        if problem_text:
            parts.append(f"Problem: {problem_text[:500]}")
        problem_section = "\n\nStudent is working on:\n" + "\n".join(parts)

    return (
        "You are a handwriting-to-LaTeX transcription model. "
        "The image shows handwritten math/science work on paper. "
        "New strokes appear in BLACK. Previously transcribed strokes appear in GRAY for context.\n\n"
        "You have TWO tasks. Respond with JSON: {\"delta_latex\": \"...\", \"should_check\": true/false}\n\n"
        "TASK 1 — TRANSCRIPTION:\n"
        "Output ONLY the LaTeX for the new (black) strokes. "
        "Do not repeat content from the previous transcription. "
        "Use inline $...$ for math expressions. "
        "If the new strokes are a continuation of a previous line, output only the new part.\n\n"
        "TASK 2 — CHECK SIGNAL:\n"
        "Set should_check to true if ANY of these apply:\n"
        "- Student just completed a logical step or sub-answer (finished an equation, boxed something)\n"
        "- Visible error (wrong sign, dropped variable, arithmetic mistake)\n"
        "- Student appears to be writing a final answer\n"
        "- New strokes show hesitation or repeated crossing-out\n"
        f"- Significant new work since last check (batches_since_check = {batches_since_check}; high means lots of unchecked work)\n\n"
        "Set should_check to false if the student is mid-expression or nothing notable changed. "
        "Default to false when uncertain."
        f"{problem_section}"
        f"{context_section}"
    )
