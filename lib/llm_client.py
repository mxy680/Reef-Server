"""Thin wrapper for OpenAI API using the openai SDK."""

import base64
import copy
import os

from openai import OpenAI


def _make_strict(schema: dict) -> dict:
    """Recursively patch a Pydantic JSON schema for OpenAI strict mode.

    - Adds additionalProperties: false to all objects
    - Ensures required includes all properties
    """
    schema = copy.deepcopy(schema)

    def _patch(node: dict) -> None:
        # $ref cannot have sibling keywords in strict mode — strip them
        if "$ref" in node:
            ref = node["$ref"]
            node.clear()
            node["$ref"] = ref
            return
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            if "properties" in node:
                node["required"] = list(node["properties"].keys())
            for prop in node.get("properties", {}).values():
                _patch(prop)
        if "items" in node:
            _patch(node["items"])
        if "anyOf" in node:
            for variant in node["anyOf"]:
                _patch(variant)
        for ref in node.get("$defs", {}).values():
            _patch(ref)

    _patch(schema)
    return schema


class LLMClient:
    """Client for interacting with any OpenAI-compatible API (OpenAI, OpenRouter, etc.)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-nano",
        base_url: str | None = None,
    ):
        """
        Initialize the OpenAI client.

        Args:
            api_key: API key. If None, uses OPENAI_API_KEY env var.
            model: Model name to use.
            base_url: Optional base URL for OpenAI-compatible APIs (e.g. OpenRouter).
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY env var or pass api_key."
            )
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def generate(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        temperature: float | None = None,
        response_schema: dict | None = None,
        system_message: str | None = None,
    ) -> str:
        """
        Generate text response, optionally with images.

        Args:
            prompt: Text prompt to send.
            images: Optional list of JPEG image bytes to include.
            temperature: Sampling temperature.
            response_schema: Optional JSON schema for structured output.
            system_message: Optional system prompt prepended to messages.

        Returns:
            Generated text response.
        """
        # Build message content
        content: list[dict] = [{"type": "text", "text": prompt}]
        if images:
            for img_bytes in images:
                b64 = base64.b64encode(img_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        messages: list[dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": content})

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": _make_strict(response_schema)},
            }

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def generate_stream(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        temperature: float | None = None,
        response_schema: dict | None = None,
        system_message: str | None = None,
    ):
        """Yield text chunks as they arrive from the model."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        if images:
            for img_bytes in images:
                b64 = base64.b64encode(img_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        messages: list[dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": content})

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": _make_strict(response_schema)},
            }

        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
