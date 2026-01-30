"""Models for the /ai/quiz endpoint (quiz generation)."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from .common import RAGContext, AIPreferences


class QuestionType(str, Enum):
    """Types of quiz questions."""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    FILL_BLANK = "fill_blank"


class Difficulty(str, Enum):
    """Quiz difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuizConfig(BaseModel):
    """Configuration for quiz generation."""
    count: int = Field(5, ge=1, le=20, description="Number of questions to generate")
    types: list[QuestionType] = Field(
        default_factory=lambda: [QuestionType.MULTIPLE_CHOICE],
        description="Types of questions to include"
    )
    difficulty: Difficulty = Field(Difficulty.MEDIUM, description="Difficulty level")


class QuizPreferences(AIPreferences):
    """Preferences specific to quiz endpoint."""
    model: str = Field("GPT-4o", description="Selected model for quiz generation")


class QuizRequest(BaseModel):
    """Request body for quiz generation."""
    rag_context: RAGContext = Field(..., description="RAG context (required for quiz generation)")
    config: QuizConfig = Field(default_factory=QuizConfig, description="Quiz configuration")
    preferences: QuizPreferences = Field(default_factory=QuizPreferences)
    additional_instructions: Optional[str] = Field(None, description="Additional instructions for the AI")


class QuizOption(BaseModel):
    """An option for multiple choice questions."""
    label: str = Field(..., description="Option label (A, B, C, D)")
    text: str = Field(..., description="Option text")


class QuizQuestion(BaseModel):
    """A generated quiz question."""
    id: str = Field(..., description="Unique question identifier")
    type: QuestionType = Field(..., description="Question type")
    question: str = Field(..., description="The question text")
    options: Optional[list[QuizOption]] = Field(None, description="Options for multiple choice")
    correct_answer: str = Field(..., description="The correct answer")
    explanation: str = Field(..., description="Explanation of the correct answer")
    source_chunk_ids: list[str] = Field(default_factory=list, description="IDs of source chunks used")


class QuizResponse(BaseModel):
    """Response from quiz generation."""
    questions: list[QuizQuestion] = Field(..., description="Generated questions")
    model: str = Field(..., description="Model used for generation")
    provider: str = Field(..., description="Provider used")
    mode: str = Field("prod", description="Mode used (prod or mock)")


# JSON Schema for structured output
QUIZ_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["multiple_choice", "true_false", "short_answer", "fill_blank"]},
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "text": {"type": "string"}
                            },
                            "required": ["label", "text"]
                        }
                    },
                    "correct_answer": {"type": "string"},
                    "explanation": {"type": "string"},
                    "source_chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["id", "type", "question", "correct_answer", "explanation"]
            }
        }
    },
    "required": ["questions"]
}
