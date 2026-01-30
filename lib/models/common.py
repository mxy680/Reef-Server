"""Common models shared across AI endpoints."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DetailLevel(str, Enum):
    """Level of detail for AI responses."""
    CONCISE = "concise"
    BALANCED = "balanced"
    DETAILED = "detailed"


class ContextChunk(BaseModel):
    """A chunk of context from RAG retrieval."""
    text: str = Field(..., description="The text content of the chunk")
    source_type: str = Field(..., description="Type of source: 'note' or 'assignment'")
    source_id: str = Field(..., description="Unique identifier of the source document")
    heading: Optional[str] = Field(None, description="Section heading if available")
    page_number: Optional[int] = Field(None, description="Page number if from PDF")
    similarity: float = Field(..., ge=0, le=1, description="Similarity score from RAG retrieval")


class RAGContext(BaseModel):
    """Context retrieved from RAG for AI prompts."""
    chunks: list[ContextChunk] = Field(default_factory=list, description="Retrieved context chunks")
    query: str = Field(..., description="The original query used for retrieval")


class AIPreferences(BaseModel):
    """User preferences for AI behavior."""
    model: str = Field("GPT-4o", description="Selected AI model")
    detail_level: DetailLevel = Field(DetailLevel.BALANCED, description="Response detail level")
    language: str = Field("en", description="Response language code")


class ImageData(BaseModel):
    """Base64-encoded image data."""
    data: str = Field(..., description="Base64-encoded image data")
    mime_type: str = Field("image/png", description="MIME type of the image")
