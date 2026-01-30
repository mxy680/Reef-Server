"""
Tests for Reef Server API endpoints.

Uses pytest and FastAPI's TestClient for testing.
"""

import pytest
from fastapi.testclient import TestClient
import os
import sys

# Add parent directory to path for imports
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
        assert "version" in data


class TestGenerateEndpoint:
    """Tests for /gemini/generate endpoint."""

    def test_generate_mock_mode(self, client):
        """Generate in mock mode should return mock response."""
        response = client.post(
            "/gemini/generate?mode=mock",
            json={"prompt": "Hello, world!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["mode"] == "mock"

    def test_generate_mock_with_scenario(self, client):
        """Generate with X-Mock-Scenario header."""
        response = client.post(
            "/gemini/generate?mode=mock",
            json={"prompt": "Test"},
            headers={"X-Mock-Scenario": "latex_simple"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "$x + y = z$" in data["text"]

    def test_generate_requires_prompt(self, client):
        """Generate should require prompt field."""
        response = client.post(
            "/gemini/generate?mode=mock",
            json={}
        )
        assert response.status_code == 422  # Validation error

    def test_generate_invalid_mode(self, client):
        """Generate with invalid mode should fail."""
        response = client.post(
            "/gemini/generate?mode=invalid",
            json={"prompt": "Test"}
        )
        assert response.status_code == 422


class TestVisionEndpoint:
    """Tests for /gemini/vision endpoint."""

    def test_vision_mock_mode(self, client):
        """Vision in mock mode should return mock response."""
        response = client.post(
            "/gemini/vision?mode=mock",
            json={
                "prompt": "What do you see?",
                "images": [{"data": "base64data", "mime_type": "image/png"}]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["mode"] == "mock"

    def test_vision_requires_images(self, client):
        """Vision should require images field."""
        response = client.post(
            "/gemini/vision?mode=mock",
            json={"prompt": "Test"}
        )
        assert response.status_code == 422


class TestErrorSimulation:
    """Tests for error simulation."""

    def test_rate_limit_error(self, client):
        """Rate limit simulation should return 429."""
        response = client.post(
            "/gemini/generate?mode=mock&error=rate_limit",
            json={"prompt": "Test"}
        )
        assert response.status_code == 429

    def test_timeout_error(self, client):
        """Timeout simulation should return 504."""
        response = client.post(
            "/gemini/generate?mode=mock&error=timeout",
            json={"prompt": "Test"}
        )
        assert response.status_code == 504

    def test_server_error(self, client):
        """Server error simulation should return 500."""
        response = client.post(
            "/gemini/generate?mode=mock&error=500",
            json={"prompt": "Test"}
        )
        assert response.status_code == 500


class TestDelaySimulation:
    """Tests for delay simulation."""

    def test_delay_parameter_accepted(self, client):
        """Delay parameter should be accepted."""
        # Use small delay for test speed
        response = client.post(
            "/gemini/generate?mode=mock&delay=100",
            json={"prompt": "Test"}
        )
        assert response.status_code == 200

    def test_delay_max_limit(self, client):
        """Delay over 30000ms should fail validation."""
        response = client.post(
            "/gemini/generate?mode=mock&delay=50000",
            json={"prompt": "Test"}
        )
        assert response.status_code == 422


class TestLogsEndpoint:
    """Tests for /logs endpoint."""

    def test_logs_returns_list(self, client):
        """Logs endpoint should return a list."""
        # Make a request first to generate a log
        client.post(
            "/gemini/generate?mode=mock",
            json={"prompt": "Test for logging"}
        )

        response = client.get("/logs")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert isinstance(data["logs"], list)

    def test_logs_limit_parameter(self, client):
        """Logs should respect limit parameter."""
        response = client.get("/logs?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data["logs"]) <= 5


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
        # Each embedding should be 384 dimensions
        for emb in data["embeddings"]:
            assert len(emb) == 384

    def test_embed_normalized_mock(self, client):
        """Embeddings should be normalized (L2 norm ~1)."""
        response = client.post(
            "/ai/embed?mode=mock",
            json={"texts": "Test normalization", "normalize": True}
        )
        assert response.status_code == 200
        data = response.json()
        embedding = data["embeddings"][0]
        # Calculate L2 norm
        norm = sum(x * x for x in embedding) ** 0.5
        # Should be approximately 1.0
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

    def test_embed_rate_limit_error(self, client):
        """Embed rate limit simulation should return 429."""
        response = client.post(
            "/ai/embed?mode=mock&error=rate_limit",
            json={"texts": "Test"}
        )
        assert response.status_code == 429

    def test_embed_timeout_error(self, client):
        """Embed timeout simulation should return 504."""
        response = client.post(
            "/ai/embed?mode=mock&error=timeout",
            json={"texts": "Test"}
        )
        assert response.status_code == 504

    def test_embed_server_error(self, client):
        """Embed server error simulation should return 500."""
        response = client.post(
            "/ai/embed?mode=mock&error=500",
            json={"texts": "Test"}
        )
        assert response.status_code == 500

    def test_embed_invalid_mode(self, client):
        """Embed with invalid mode should fail."""
        response = client.post(
            "/ai/embed?mode=invalid",
            json={"texts": "Test"}
        )
        assert response.status_code == 422
