"""Models for text embedding endpoint."""

from typing import Union
from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    """Request body for text embedding."""
    texts: Union[str, list[str]] = Field(
        ...,
        description="Single text string or list of texts to embed"
    )
    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize the output embeddings"
    )


class EmbedResponse(BaseModel):
    """Response from text embedding endpoint."""
    embeddings: list[list[float]] = Field(
        ...,
        description="List of embedding vectors (always a list, even for single input)"
    )
    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="The embedding model used"
    )
    dimensions: int = Field(
        default=384,
        description="Dimensionality of the embedding vectors"
    )
    count: int = Field(
        ...,
        description="Number of texts embedded"
    )
    mode: str = Field(
        ...,
        description="'mock' or 'prod'"
    )
