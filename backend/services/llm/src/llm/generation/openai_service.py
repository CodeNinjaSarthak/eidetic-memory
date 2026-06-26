"""OpenAI-backed LLM text generation service."""

import json
import os
from typing import Any

from openai import AsyncOpenAI

from llm.generation.base import AbstractLLMService, LLMError


class OpenAIService(AbstractLLMService):
    """LLM service backed by OpenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1",
    ) -> None:
        """Initialize the OpenAI service.

        Args:
            api_key: OpenAI API key. If None, reads OPENAI_API_KEY from env.
            model: OpenAI model name.

        Raises:
            ValueError: If required configuration is missing.
        """
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1")

        if not resolved_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Pass api_key= or set OPENAI_API_KEY."
            )

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            timeout=60.0,
            max_retries=0,
        )
        self._model = resolved_model

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
            sdk_messages: list[dict[str, str]] = []
            if system:
                sdk_messages.append({"role": "system", "content": system})
            sdk_messages.extend(messages)

            create_kwargs: dict = {
                "model": self._model,
                "messages": sdk_messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                create_kwargs["max_tokens"] = max_tokens
            response = await self._client.chat.completions.create(**create_kwargs)
            return response.choices[0].message.content
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"OpenAI completion failed: {e}") from e

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
                temperature=0,
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
            raise LLMError(f"OpenAI tool completion failed: {e}") from e
