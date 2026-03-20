"""Behavioral tests for LLM text generation services."""

from typing import Any

import pytest

from llm.generation.base import AbstractLLMService, LLMError
from llm.generation.claude import ClaudeService


class FakeLLMService(AbstractLLMService):
    """In-memory LLM service that returns canned responses."""

    def __init__(
        self,
        text_response: str = "hello",
        tool_response: dict[str, Any] | None = None,
    ) -> None:
        self._text_response = text_response
        self._tool_response = tool_response or {"facts": ["user likes coffee"]}

    async def complete(self, messages: list[dict[str, str]], system: str) -> str:
        """Return the canned text response."""
        return self._text_response

    async def complete_with_tool(
        self,
        messages: list[dict[str, str]],
        tool: dict[str, Any],
        system: str,
    ) -> dict[str, Any]:
        """Return the canned tool response."""
        return self._tool_response


class FailingLLMService(AbstractLLMService):
    """LLM service that always raises LLMError."""

    async def complete(self, messages: list[dict[str, str]], system: str) -> str:
        """Always fail."""
        raise LLMError("intentional failure")

    async def complete_with_tool(
        self,
        messages: list[dict[str, str]],
        tool: dict[str, Any],
        system: str,
    ) -> dict[str, Any]:
        """Always fail."""
        raise LLMError("intentional tool failure")


@pytest.mark.asyncio
async def test_llm_service_returns_text_response() -> None:
    service = FakeLLMService(text_response="world")

    result = await service.complete(
        messages=[{"role": "user", "content": "hi"}],
        system="be helpful",
    )

    assert result == "world"


@pytest.mark.asyncio
async def test_llm_service_complete_with_tool_returns_tool_input() -> None:
    expected = {"facts": ["user is a developer"]}
    service = FakeLLMService(tool_response=expected)

    result = await service.complete_with_tool(
        messages=[{"role": "user", "content": "I'm a developer"}],
        tool={"name": "extract", "description": "extract", "input_schema": {}},
        system="extract facts",
    )

    assert result == expected


@pytest.mark.asyncio
async def test_llm_error_is_raised_on_failure() -> None:
    service = FailingLLMService()

    with pytest.raises(LLMError, match="intentional failure"):
        await service.complete(
            messages=[{"role": "user", "content": "hi"}],
            system="be helpful",
        )


def test_claude_service_creates_instance_with_api_key() -> None:
    service = ClaudeService(api_key="sk-ant-test")

    assert service._model == "claude-3-5-haiku-20241022"
