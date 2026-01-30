"""Provider router - maps model names to Gemini provider."""

import os
from typing import Optional

from .base import AIProvider, ProviderCapability
from .google import GoogleProvider


# Mapping from user-facing model names to (provider_class, model_id)
MODEL_MAPPING = {
    # Google Gemini models
    "Gemini Pro": (GoogleProvider, "gemini-pro"),
    "Gemini 3 Pro": (GoogleProvider, "gemini-3-pro"),
    "Gemini 2 Flash": (GoogleProvider, "gemini-2-flash"),
    "gemini-pro": (GoogleProvider, "gemini-pro"),
    "gemini-3-pro": (GoogleProvider, "gemini-3-pro"),
    "gemini-2-flash": (GoogleProvider, "gemini-2-flash"),
    "gemini-2.5-pro": (GoogleProvider, "gemini-2.5-pro"),
    "gemini-2.5-flash": (GoogleProvider, "gemini-2.5-flash"),
}


class ProviderRouter:
    """Routes requests to the Gemini provider based on model selection."""

    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize the router with API key.

        Key can be provided directly or will be read from environment variable.
        """
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        # Cache of active provider instances
        self._providers: dict[str, AIProvider] = {}

    def get_provider(self, model_name: str) -> AIProvider:
        """
        Get the appropriate provider for a model.

        Args:
            model_name: User-facing model name (e.g., "Gemini Pro", "Gemini 2 Flash")

        Returns:
            Configured AIProvider instance

        Raises:
            ValueError: If model is not recognized or API key is missing
        """
        if model_name not in MODEL_MAPPING:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_MAPPING.keys())}")

        provider_class, model_id = MODEL_MAPPING[model_name]

        # Create a cache key based on provider + model
        cache_key = f"{provider_class.PROVIDER_NAME}:{model_id}"

        if cache_key not in self._providers:
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not configured")

            self._providers[cache_key] = provider_class(api_key=self.gemini_api_key, model=model_id)

        return self._providers[cache_key]

    def supports_model(self, model_name: str) -> bool:
        """Check if a model is supported."""
        return model_name in MODEL_MAPPING

    def get_models_for_capability(self, capability: ProviderCapability) -> list[str]:
        """Get all models that support a capability."""
        result = []
        for model_name, (provider_class, _) in MODEL_MAPPING.items():
            if capability in provider_class.CAPABILITIES:
                result.append(model_name)
        return result

    async def close_all(self):
        """Close all provider connections."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()
