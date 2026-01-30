"""Base class for AI providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProviderCapability(Enum):
    """Capabilities that providers may support."""
    TEXT = "text"
    VISION = "vision"
    JSON_OUTPUT = "json_output"
    STRUCTURED_OUTPUT = "structured_output"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: int = 4096
    json_output: bool = False
    schema: Optional[dict] = None

    def __post_init__(self):
        # When requesting JSON/structured output, use lower temperature
        if self.json_output or self.schema:
            self.temperature = 0


@dataclass
class GenerationResult:
    """Result from a generation request."""
    text: str
    model: str
    provider: str
    usage: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    PROVIDER_NAME: str = "base"
    CAPABILITIES: set[ProviderCapability] = set()

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._client = None

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: The text prompt
            config: Optional generation configuration

        Returns:
            GenerationResult with the response
        """
        pass

    @abstractmethod
    async def generate_with_images(
        self,
        prompt: str,
        images: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate text from a prompt with images.

        Args:
            prompt: The text prompt
            images: List of image dicts with "data" (base64) and "mime_type"
            config: Optional generation configuration

        Returns:
            GenerationResult with the response
        """
        pass

    def supports(self, capability: ProviderCapability) -> bool:
        """Check if this provider supports a capability."""
        return capability in self.CAPABILITIES

    @abstractmethod
    async def close(self):
        """Close any open connections."""
        pass
