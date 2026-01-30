"""
Embedding Service - Generates text embeddings using sentence-transformers.

Uses all-MiniLM-L6-v2 model which produces 384-dimensional vectors.
Implements lazy loading to avoid startup delay.
"""

from typing import Union
import numpy as np

# Global singleton instance
_embedding_service = None


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers."""

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSIONS = 384

    def __init__(self):
        self._model = None

    def _load_model(self):
        """Lazily load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
            print(f"[EmbeddingService] Loaded model: {self.MODEL_NAME}")

    def embed(
        self,
        texts: Union[str, list[str]],
        normalize: bool = True
    ) -> list[list[float]]:
        """
        Generate embeddings for one or more texts.

        Args:
            texts: Single text string or list of texts
            normalize: Whether to L2-normalize the embeddings (default True)

        Returns:
            List of embedding vectors (always a list, even for single input)
        """
        self._load_model()

        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Generate embeddings
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        # Convert to list of lists for JSON serialization
        return embeddings.tolist()

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.MODEL_NAME

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self.DIMENSIONS


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
