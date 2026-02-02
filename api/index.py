"""
Reef Server - FastAPI application for text embeddings.

Provides:
- Text embeddings using MiniLM-L6-v2
- Mock mode for testing
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os

# Import lib modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.mock_responses import get_mock_embedding
from lib.models import (
    EmbedRequest,
    EmbedResponse,
    ExtractQuestionsRequest,
    ExtractQuestionsResponse,
    QuestionData,
)
from lib.embedding import get_embedding_service

# Cache for Marker models (loaded once at startup)
_marker_models = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload small models at startup. Marker models loaded lazily on first request."""
    print("[Startup] Preloading embedding model...")
    embedding_service = get_embedding_service()
    embedding_service._load_model()
    print("[Startup] Embedding model loaded! Marker models will load on first request.")

    yield

    print("[Shutdown] Cleaning up...")


def get_marker_models():
    """Get cached Marker models."""
    global _marker_models
    if _marker_models is None:
        from marker.models import create_model_dict
        _marker_models = create_model_dict()
    return _marker_models


app = FastAPI(
    title="Reef Server",
    description="Embedding service for Reef iOS app",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "reef-server",
        "version": "1.0.0"
    }


@app.post("/ai/embed", response_model=EmbedResponse)
async def ai_embed(
    request_body: EmbedRequest,
    request: Request,
    mode: str = Query(default="prod", pattern="^(mock|prod)$"),
):
    """
    Generate text embeddings using MiniLM-L6-v2.

    Accepts a single text string or a list of texts. Returns 384-dimensional
    normalized vectors suitable for semantic search.

    Query Parameters:
    - mode: "mock" for testing, "prod" for real embeddings
    """
    # Normalize input to list
    texts = request_body.texts if isinstance(request_body.texts, list) else [request_body.texts]
    text_count = len(texts)

    try:
        # Validate input
        if text_count == 0:
            raise HTTPException(status_code=422, detail="texts cannot be empty")
        if text_count > 100:
            raise HTTPException(status_code=422, detail="Maximum 100 texts per request")

        # Mock mode
        if mode == "mock":
            embeddings = get_mock_embedding(count=text_count, dimensions=384)
            return EmbedResponse(
                embeddings=embeddings,
                model="all-MiniLM-L6-v2",
                dimensions=384,
                count=text_count,
                mode="mock",
            )

        # Production mode
        embedding_service = get_embedding_service()
        embeddings = embedding_service.embed(texts, normalize=request_body.normalize)

        return EmbedResponse(
            embeddings=embeddings,
            model=embedding_service.model_name,
            dimensions=embedding_service.dimensions,
            count=text_count,
            mode="prod",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/extract-questions", response_model=ExtractQuestionsResponse)
async def ai_extract_questions(
    request_body: ExtractQuestionsRequest,
    request: Request,
):
    """
    Extract questions from a PDF document.

    Takes a base64-encoded PDF and returns individual questions as separate PDFs.
    Uses Marker for OCR/layout extraction and Gemini for question segmentation.
    """
    try:
        from lib.question_extractor import QuestionExtractor
        from lib.latex_compiler import LaTeXCompiler

        # Initialize services
        extractor = QuestionExtractor()
        compiler = LaTeXCompiler()

        # Extract questions from PDF
        extracted_questions = await extractor.extract_questions(request_body.pdf_base64)

        if not extracted_questions:
            return ExtractQuestionsResponse(
                questions=[],
                note_id=request_body.note_id,
                total_count=0
            )

        # Compile each question to PDF
        compiled_questions = []
        for eq in extracted_questions:
            try:
                compiled = compiler.compile_question(
                    order_index=eq.order_index,
                    question_number=eq.question_number,
                    latex_content=eq.latex_content,
                    has_images=eq.has_images,
                    has_tables=eq.has_tables,
                    image_data=eq.image_data if eq.image_data else None
                )
                compiled_questions.append(QuestionData(
                    order_index=compiled.order_index,
                    question_number=compiled.question_number,
                    pdf_base64=compiled.pdf_base64,
                    has_images=compiled.has_images,
                    has_tables=compiled.has_tables
                ))
            except Exception as compile_error:
                # Log but continue with other questions
                print(f"Failed to compile question {eq.question_number}: {compile_error}")
                continue

        return ExtractQuestionsResponse(
            questions=compiled_questions,
            note_id=request_body.note_id,
            total_count=len(compiled_questions)
        )

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Required dependencies not available: {e}"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
