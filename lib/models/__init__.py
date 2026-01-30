"""Pydantic models for AI endpoints."""

from .common import ContextChunk, RAGContext, AIPreferences, ImageData
from .feedback import FeedbackRequest, FeedbackResponse
from .quiz import QuizRequest, QuizConfig, QuizQuestion, QuizOption, QuizResponse, QUIZ_RESPONSE_SCHEMA
from .chat import ChatRequest, ChatMessage, ChatResponse, SourceReference

__all__ = [
    # Common
    "ContextChunk",
    "RAGContext",
    "AIPreferences",
    "ImageData",
    # Feedback
    "FeedbackRequest",
    "FeedbackResponse",
    # Quiz
    "QuizRequest",
    "QuizConfig",
    "QuizQuestion",
    "QuizOption",
    "QuizResponse",
    "QUIZ_RESPONSE_SCHEMA",
    # Chat
    "ChatRequest",
    "ChatMessage",
    "ChatResponse",
    "SourceReference",
]
