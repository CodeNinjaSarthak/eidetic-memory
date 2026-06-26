"""Gemini-backed LLM text generation service."""

import os
from typing import Any

from google import genai
from google.genai import types

from llm.generation.base import AbstractLLMService, LLMError


class GeminiService(AbstractLLMService):
    """LLM service backed by Google's Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        """Initialize the Gemini service.

        Args:
            api_key: Google API key. If None, reads GOOGLE_API_KEY from environment.
            model: The Gemini model to use.

        Raises:
            ValueError: If no API key is provided or found in the environment.
        """
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Google API key is required. Pass api_key= or set GOOGLE_API_KEY."
            )
        self._client = genai.Client(api_key=resolved_key)
        self._model = model

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[types.Content]:
        """Convert message dicts to Gemini Content objects.

        Gemini uses "model" instead of "assistant" for the model's role.

        Args:
            messages: Conversation messages in {"role": ..., "content": ...} format.

        Returns:
            List of Gemini Content objects.
        """
        contents: list[types.Content] = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            contents.append(
                types.Content(role=role, parts=[types.Part(text=msg["content"])])
            )
        return contents

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
            contents = self._convert_messages(messages)
            config = types.GenerateContentConfig(
                system_instruction=system if system else None,
            )
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
            return response.text
        except Exception as e:
            raise LLMError(f"Gemini completion failed: {e}") from e

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
            The tool input dict extracted from the model's function_call.

        Raises:
            LLMError: If the request fails.
        """
        try:
            contents = self._convert_messages(messages)
            fn_decl = types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["input_schema"],
            )
            gemini_tool = types.Tool(function_declarations=[fn_decl])
            config = types.GenerateContentConfig(
                system_instruction=system if system else None,
                tools=[gemini_tool],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[tool["name"]],
                    )
                ),
            )
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
            try:
                args = response.candidates[0].content.parts[0].function_call.args
                return dict(args)
            except (AttributeError, IndexError) as e:
                raise LLMError("No function_call found in response") from e
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Gemini tool completion failed: {e}") from e
