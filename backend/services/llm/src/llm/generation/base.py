"""Abstract base class for LLM text generation services."""

from abc import ABC, abstractmethod
from typing import Any


class LLMError(Exception):
    """Raised when an LLM request fails."""


class AbstractLLMService(ABC):
    """Interface for LLM text generation."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a text completion from a list of messages.

        Args:
            messages: Conversation messages in {"role": ..., "content": ...} format.
            system: System prompt to guide the model.

        Returns:
            The model's text response.

        Raises:
            LLMError: If the request fails.
        """

    @abstractmethod
    async def complete_with_tool(
        self,
        messages: list[dict[str, str]],
        tool: dict[str, Any],
        system: str = "",
    ) -> dict[str, Any]:
        """Generate a completion that invokes a specified tool.

        Args:
            messages: Conversation messages in {"role": ..., "content": ...} format.
            tool: Tool definition dict with name, description, and input_schema.
            system: System prompt to guide the model.

        Returns:
            The tool input dict extracted from the model's tool_use block.

        Raises:
            LLMError: If the request fails.
        """
