"""Google Gemini provider implementation."""

import httpx
from typing import Optional

from .base import AIProvider, GenerationConfig, GenerationResult, ProviderCapability


class GoogleProvider(AIProvider):
    """Provider for Google's Gemini models."""

    PROVIDER_NAME = "google"
    CAPABILITIES = {
        ProviderCapability.TEXT,
        ProviderCapability.VISION,
        ProviderCapability.JSON_OUTPUT,
        ProviderCapability.STRUCTURED_OUTPUT,
    }

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    # Model name mapping from user-facing names to API model IDs
    MODEL_MAP = {
        "gemini-pro": "gemini-2.5-pro",
        "gemini-3-pro": "gemini-2.5-pro",
        "gemini-2-flash": "gemini-2.5-flash",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
    }

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        super().__init__(api_key)
        # Map model name if needed
        self.model = self.MODEL_MAP.get(model, model)
        self._client = httpx.AsyncClient(timeout=60.0)

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text using Gemini."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured")

        config = config or GenerationConfig()
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        request_body = self._build_request_body(
            parts=[{"text": prompt}],
            config=config,
        )

        response = await self._client.post(
            url,
            json=request_body,
            headers={"Content-Type": "application/json"},
        )

        text = self._parse_response(response)
        return GenerationResult(
            text=text,
            model=self.model,
            provider=self.PROVIDER_NAME,
        )

    async def generate_with_images(
        self,
        prompt: str,
        images: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text with images using Gemini."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured")

        config = config or GenerationConfig()
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        # Build parts: images first, then text
        parts = []
        for image in images:
            parts.append({
                "inline_data": {
                    "mime_type": image.get("mime_type", "image/png"),
                    "data": image.get("data", ""),
                }
            })
        parts.append({"text": prompt})

        request_body = self._build_request_body(parts=parts, config=config)

        response = await self._client.post(
            url,
            json=request_body,
            headers={"Content-Type": "application/json"},
        )

        text = self._parse_response(response)
        return GenerationResult(
            text=text,
            model=self.model,
            provider=self.PROVIDER_NAME,
        )

    def _build_request_body(
        self,
        parts: list[dict],
        config: GenerationConfig,
    ) -> dict:
        """Build the Gemini API request body."""
        body = {
            "contents": [{"parts": parts}]
        }

        gen_config = {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens,
        }

        if config.schema:
            gen_config["responseMimeType"] = "application/json"
            gen_config["responseSchema"] = config.schema
        elif config.json_output:
            gen_config["responseMimeType"] = "application/json"

        body["generationConfig"] = gen_config
        return body

    def _parse_response(self, response: httpx.Response) -> str:
        """Parse Gemini API response and extract text."""
        if response.status_code != 200:
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg = error_data["error"].get("message", "Unknown error")
                    raise Exception(f"Gemini API error: {error_msg}")
            except Exception:
                pass
            raise Exception(f"Gemini API returned HTTP {response.status_code}")

        data = response.json()

        if "error" in data:
            raise Exception(f"Gemini API error: {data['error'].get('message', 'Unknown error')}")

        candidates = data.get("candidates", [])
        if not candidates:
            raise Exception("Gemini API returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise Exception("Gemini API returned no content")

        text = parts[0].get("text")
        if text is None:
            raise Exception("Gemini API returned no text")

        return text

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
