"""
Gemini API Proxy - Forwards requests to Google's Gemini API.

Handles authentication, request formatting, and response parsing.
"""

import httpx
from typing import Optional


class GeminiProxy:
    """Proxy for Gemini API requests."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """
        Initialize the Gemini proxy.

        Args:
            api_key: Gemini API key. Required for production mode.
            model: Model to use (default: gemini-2.5-flash)
        """
        self.api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(timeout=60.0)

    async def generate_content(
        self,
        prompt: str,
        json_output: bool = False,
        schema: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ) -> str:
        """
        Generate content using Gemini.

        Args:
            prompt: The text prompt to send
            json_output: If True, request JSON output format
            schema: Optional JSON schema for structured output
            generation_config: Optional generation config overrides

        Returns:
            Generated text response

        Raises:
            ValueError: If API key is not configured
            httpx.HTTPStatusError: If API returns an error
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured")

        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        # Build request body
        request_body = self._build_request_body(
            parts=[{"text": prompt}],
            json_output=json_output,
            schema=schema,
            generation_config=generation_config,
        )

        response = await self._client.post(
            url,
            json=request_body,
            headers={"Content-Type": "application/json"},
        )

        return self._parse_response(response)

    async def generate_content_with_images(
        self,
        prompt: str,
        images: list[dict],
        json_output: bool = False,
        schema: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ) -> str:
        """
        Generate content with images (multimodal).

        Args:
            prompt: The text prompt to send
            images: List of image dicts with "data" (base64) and "mime_type"
            json_output: If True, request JSON output format
            schema: Optional JSON schema for structured output
            generation_config: Optional generation config overrides

        Returns:
            Generated text response

        Raises:
            ValueError: If API key is not configured
            httpx.HTTPStatusError: If API returns an error
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured")

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

        # Build request body
        request_body = self._build_request_body(
            parts=parts,
            json_output=json_output,
            schema=schema,
            generation_config=generation_config,
        )

        response = await self._client.post(
            url,
            json=request_body,
            headers={"Content-Type": "application/json"},
        )

        return self._parse_response(response)

    def _build_request_body(
        self,
        parts: list[dict],
        json_output: bool = False,
        schema: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ) -> dict:
        """Build the Gemini API request body."""
        body = {
            "contents": [{"parts": parts}]
        }

        # Build generation config
        config = {}

        if generation_config:
            config.update(generation_config)

        if schema:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = schema
            config.setdefault("temperature", 0)
            config.setdefault("maxOutputTokens", 16384)
        elif json_output:
            config["responseMimeType"] = "application/json"
            config.setdefault("temperature", 0)
            config.setdefault("maxOutputTokens", 16384)

        if config:
            body["generationConfig"] = config

        return body

    def _parse_response(self, response: httpx.Response) -> str:
        """Parse Gemini API response and extract text."""
        if response.status_code != 200:
            # Try to extract error message
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg = error_data["error"].get("message", "Unknown error")
                    raise Exception(f"Gemini API error: {error_msg}")
            except Exception:
                pass
            raise Exception(f"Gemini API returned HTTP {response.status_code}")

        data = response.json()

        # Check for API error in response
        if "error" in data:
            raise Exception(f"Gemini API error: {data['error'].get('message', 'Unknown error')}")

        # Extract text from response
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
        await self._client.aclose()
