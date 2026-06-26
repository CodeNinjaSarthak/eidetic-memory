"""Claude-backed LLM text generation service."""

import os
from typing import Any

from anthropic import AsyncAnthropic

from llm.generation.base import AbstractLLMService, LLMError


class ClaudeService(AbstractLLMService):
    """LLM service backed by Anthropic's Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-haiku-20241022",
    ) -> None:
        """Initialize the Claude service.

        Args:
            api_key: Anthropic API key. If None, reads ANTHROPIC_API_KEY from environment.
            model: The Claude model to use.

        Raises:
            ValueError: If no API key is provided or found in the environment.
        """
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key is required. Pass api_key= or set ANTHROPIC_API_KEY."
            )
        self._client = AsyncAnthropic(api_key=resolved_key)
        self._model = model

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
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                messages=messages,
            )
            return response.content[0].text
        except Exception as e:
            raise LLMError(f"Claude completion failed: {e}") from e

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
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool["name"]},
            )
            for block in response.content:
                if block.type == "tool_use":
                    return block.input
            raise LLMError("No tool_use block found in response")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Claude tool completion failed: {e}") from e
