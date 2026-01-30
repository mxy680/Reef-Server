"""Mock responses for testing."""

import random
import math


def get_mock_embedding(count: int = 1, dimensions: int = 384) -> list[list[float]]:
    """
    Generate mock embedding vectors for testing.

    Produces random normalized vectors that mimic real embedding behavior.

    Args:
        count: Number of embedding vectors to generate
        dimensions: Dimensionality of each vector (default 384 for MiniLM)

    Returns:
        List of normalized embedding vectors
    """
    embeddings = []
    for _ in range(count):
        # Generate random vector
        vector = [random.gauss(0, 1) for _ in range(dimensions)]

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        embeddings.append(vector)

    return embeddings
