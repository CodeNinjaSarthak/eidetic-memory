"""Groq-backed LLM text generation service."""

import json
import os
from typing import Any

from groq import AsyncGroq

from llm.generation.base import AbstractLLMService, LLMError


class GroqService(AbstractLLMService):
    """LLM service backed by Groq's API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        """Initialize the Groq service.

        Args:
            api_key: Groq API key. If None, reads GROQ_API_KEY from environment.
            model: The Groq model to use.

        Raises:
            ValueError: If no API key is provided or found in the environment.
        """
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Groq API key is required. Pass api_key= or set GROQ_API_KEY."
            )
        self._client = AsyncGroq(api_key=resolved_key)
        self._model = model

    async def complete(self, messages: list[dict[str, str]], system: str = "") -> str:
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
            sdk_messages: list[dict[str, str]] = []
            if system:
                sdk_messages.append({"role": "system", "content": system})
            sdk_messages.extend(messages)

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=sdk_messages,
            )
            return response.choices[0].message.content
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Groq completion failed: {e}") from e

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
            The tool input dict extracted from the model's tool_call.

        Raises:
            LLMError: If the request fails.
        """
        try:
            sdk_messages: list[dict[str, str]] = []
            if system:
                sdk_messages.append({"role": "system", "content": system})
            sdk_messages.extend(messages)

            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=sdk_messages,
                tools=[openai_tool],
                tool_choice={"type": "function", "function": {"name": tool["name"]}},
            )

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise LLMError("No tool_call found in response")

            try:
                return json.loads(tool_calls[0].function.arguments)
            except json.JSONDecodeError as e:
                raise LLMError("Failed to parse tool arguments") from e
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Groq tool completion failed: {e}") from e
