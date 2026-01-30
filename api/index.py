"""
Reef Server - FastAPI application for text embeddings.

Provides:
- Text embeddings using MiniLM-L6-v2
- Mock mode for testing
"""

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os

# Import lib modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.mock_responses import get_mock_embedding
from lib.models import EmbedRequest, EmbedResponse
from lib.embedding import get_embedding_service

app = FastAPI(
    title="Reef Server",
    description="Embedding service for Reef iOS app",
    version="1.0.0"
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
