"""Models for the /ai/chat endpoint (RAG chat)."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from .common import RAGContext, AIPreferences, ContextChunk


class MessageRole(str, Enum):
    """Role in a chat conversation."""
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A message in a chat conversation."""
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class ChatPreferences(AIPreferences):
    """Preferences specific to chat endpoint."""
    model: str = Field("GPT-4o", description="Selected model for chat")


class ChatRequest(BaseModel):
    """Request body for RAG chat."""
    message: str = Field(..., min_length=1, description="User's message")
    rag_context: Optional[RAGContext] = Field(None, description="RAG context from course materials")
    conversation_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages in the conversation"
    )
    preferences: ChatPreferences = Field(default_factory=ChatPreferences)


class SourceReference(BaseModel):
    """A reference to a source used in the response."""
    source_id: str = Field(..., description="ID of the source document")
    source_type: str = Field(..., description="Type of source")
    heading: Optional[str] = Field(None, description="Section heading if available")
    relevance: float = Field(..., description="Relevance score")


class ChatResponse(BaseModel):
    """Response from RAG chat."""
    message: str = Field(..., description="Assistant's response")
    sources: list[SourceReference] = Field(default_factory=list, description="Sources used in response")
    model: str = Field(..., description="Model used for generation")
    provider: str = Field(..., description="Provider used")
    mode: str = Field("prod", description="Mode used (prod or mock)")
