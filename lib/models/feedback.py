"""Models for the /ai/feedback endpoint (handwriting analysis)."""

from typing import Optional
from pydantic import BaseModel, Field

from .common import RAGContext, AIPreferences, ImageData, DetailLevel


class FeedbackPreferences(AIPreferences):
    """Preferences specific to feedback endpoint."""
    model: str = Field("Gemini 3 Pro", description="Selected model for handwriting analysis")


class FeedbackRequest(BaseModel):
    """Request body for handwriting feedback."""
    images: list[ImageData] = Field(..., min_length=1, description="Images of handwritten work")
    prompt: Optional[str] = Field(None, description="Optional custom prompt or question")
    rag_context: Optional[RAGContext] = Field(None, description="RAG context from course materials")
    preferences: FeedbackPreferences = Field(default_factory=FeedbackPreferences)


class FeedbackResponse(BaseModel):
    """Response from handwriting feedback."""
    feedback: str = Field(..., description="Generated feedback text")
    detected_content: Optional[str] = Field(None, description="Detected/transcribed content from images")
    model: str = Field(..., description="Model used for generation")
    provider: str = Field(..., description="Provider used (google, openai, anthropic)")
    mode: str = Field("prod", description="Mode used (prod or mock)")
