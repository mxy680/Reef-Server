"""Tests for Reef Server API endpoints."""

import pytest
from fastapi.testclient import TestClient
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.index import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health check should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Health check should return healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "reef-server"


class TestEmbedEndpoint:
    """Tests for /ai/embed endpoint."""

    def test_embed_single_text_mock(self, client):
        """Embed single text in mock mode."""
        response = client.post(
            "/ai/embed?mode=mock",
            json={"texts": "Hello world"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1
        assert len(data["embeddings"][0]) == 384
        assert data["model"] == "all-MiniLM-L6-v2"
        assert data["dimensions"] == 384
        assert data["count"] == 1
        assert data["mode"] == "mock"

    def test_embed_batch_mock(self, client):
        """Embed multiple texts in mock mode."""
        response = client.post(
            "/ai/embed?mode=mock",
            json={"texts": ["Hello world", "Test embedding", "Another text"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["embeddings"]) == 3
        assert data["count"] == 3

    def test_embed_normalized_mock(self, client):
        """Embeddings should be normalized (L2 norm ~1)."""
        response = client.post(
            "/ai/embed?mode=mock",
            json={"texts": "Test normalization", "normalize": True}
        )
        assert response.status_code == 200
        data = response.json()
        embedding = data["embeddings"][0]
        norm = sum(x * x for x in embedding) ** 0.5
        assert 0.99 < norm < 1.01

    def test_embed_requires_texts(self, client):
        """Embed should require texts field."""
        response = client.post(
            "/ai/embed?mode=mock",
            json={}
        )
        assert response.status_code == 422

    def test_embed_empty_list_fails(self, client):
        """Embed should reject empty texts list."""
        response = client.post(
            "/ai/embed?mode=mock",
            json={"texts": []}
        )
        assert response.status_code == 422

    def test_embed_invalid_mode(self, client):
        """Embed with invalid mode should fail."""
        response = client.post(
            "/ai/embed?mode=invalid",
            json={"texts": "Test"}
        )
        assert response.status_code == 422
