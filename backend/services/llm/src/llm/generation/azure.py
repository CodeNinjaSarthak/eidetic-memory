"""Azure OpenAI-backed LLM text generation service."""

import json
import os
from typing import Any

from openai import AsyncAzureOpenAI

from llm.generation.base import AbstractLLMService, LLMError


class AzureService(AbstractLLMService):
    """LLM service backed by Azure OpenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment: str | None = None,
    ) -> None:
        """Initialize the Azure OpenAI service.

        Args:
            api_key: Azure OpenAI API key. If None, reads AZURE_OPENAI_API_KEY from env.
            endpoint: Azure OpenAI endpoint URL. If None, reads AZURE_OPENAI_ENDPOINT from env.
            deployment: Azure OpenAI deployment name. If None, reads AZURE_OPENAI_DEPLOYMENT from env.

        Raises:
            ValueError: If required configuration is missing.
        """
        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        resolved_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        resolved_deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")

        if not resolved_key:
            raise ValueError(
                "Azure OpenAI API key is required. "
                "Pass api_key= or set AZURE_OPENAI_API_KEY."
            )
        if not resolved_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint is required. "
                "Pass endpoint= or set AZURE_OPENAI_ENDPOINT."
            )
        if not resolved_deployment:
            raise ValueError(
                "Azure OpenAI deployment is required. "
                "Pass deployment= or set AZURE_OPENAI_DEPLOYMENT."
            )

        self._client = AsyncAzureOpenAI(
            api_key=resolved_key,
            azure_endpoint=resolved_endpoint,
            api_version="2024-10-21",
            timeout=60.0,
            max_retries=0,
        )
        self._deployment = resolved_deployment

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
                "model": self._deployment,
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
            raise LLMError(f"Azure completion failed: {e}") from e

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
                model=self._deployment,
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
            raise LLMError(f"Azure tool completion failed: {e}") from e
