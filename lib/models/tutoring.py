"""In-memory models for real-time tutoring sessions."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel


@dataclass
class TranscriptSegment:
    """A single transcription result from Tier 1."""

    batch_index: int
    delta_latex: str
    timestamp: float


@dataclass
class TutoringSession:
    """In-memory state for a single tutoring WebSocket connection."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    problem_id: str = ""
    question_number: int = 0
    problem_text: str = ""
    problem_parts: list[dict] = field(default_factory=list)
    course_name: str = ""

    # Rolling transcript (Tier 1 output)
    transcript_segments: list[TranscriptSegment] = field(default_factory=list)
    full_transcript: str = ""

    # Reasoning state (Tier 2)
    last_reasoning_time: float = 0.0
    last_status: str | None = None
    last_feedback: str | None = None

    # Lifecycle
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def append_transcript(self, batch_index: int, delta_latex: str) -> None:
        """Append a new transcription segment and update full transcript."""
        segment = TranscriptSegment(
            batch_index=batch_index,
            delta_latex=delta_latex,
            timestamp=time.time(),
        )
        self.transcript_segments.append(segment)
        if self.full_transcript:
            self.full_transcript += "\n" + delta_latex
        else:
            self.full_transcript = delta_latex
        self.last_activity = time.time()


class ReasoningResponse(BaseModel):
    """Structured output from Tier 2 reasoning."""

    status: Literal["on_track", "minor_error", "major_error", "stuck", "completed"]
    confidence: float
    feedback: str | None = None
