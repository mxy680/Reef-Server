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
