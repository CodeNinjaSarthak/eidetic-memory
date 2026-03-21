"""Behavioral tests for Groq generation service."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm.generation.base import LLMError
from llm.generation.groq import GroqService


def _make_groq_service(mock_client: AsyncMock) -> GroqService:
    """Create a GroqService with a pre-injected mock client."""
    service = GroqService(api_key="test-key")
    service._client = mock_client
    return service


def _mock_completion_response(content: str) -> MagicMock:
    """Build a mock chat completion response with the given content."""
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _mock_tool_response(arguments_json: str) -> MagicMock:
    """Build a mock chat completion response with a tool call."""
    tool_call = MagicMock()
    tool_call.function.arguments = arguments_json
    message = MagicMock()
    message.tool_calls = [tool_call]
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


SAMPLE_TOOL = {
    "name": "extract",
    "description": "extract facts",
    "input_schema": {"type": "object", "properties": {"facts": {"type": "array"}}},
}


@pytest.mark.asyncio
async def test_complete_returns_text_content_from_chat_response() -> None:
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_completion_response("hello from groq")
    )
    service = _make_groq_service(mock_client)

    result = await service.complete(
        messages=[{"role": "user", "content": "hi"}],
        system="be helpful",
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"
    assert call_kwargs["messages"][0] == {"role": "system", "content": "be helpful"}
    assert call_kwargs["messages"][1] == {"role": "user", "content": "hi"}
    assert result == "hello from groq"


@pytest.mark.asyncio
async def test_complete_with_tool_returns_parsed_tool_call_arguments() -> None:
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_tool_response('{"facts": ["likes coffee"]}')
    )
    service = _make_groq_service(mock_client)

    result = await service.complete_with_tool(
        messages=[{"role": "user", "content": "I like coffee"}],
        tool=SAMPLE_TOOL,
        system="extract facts",
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tools"][0]["type"] == "function"
    assert call_kwargs["tools"][0]["function"]["name"] == "extract"
    assert call_kwargs["tools"][0]["function"]["parameters"] == SAMPLE_TOOL["input_schema"]
    assert result == {"facts": ["likes coffee"]}


@pytest.mark.asyncio
async def test_complete_with_tool_raises_llm_error_when_response_has_no_tool_calls() -> None:
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_completion_response("no tools here")
    )
    service = _make_groq_service(mock_client)

    with pytest.raises(LLMError, match="No tool_call found in response"):
        await service.complete_with_tool(
            messages=[{"role": "user", "content": "hi"}],
            tool=SAMPLE_TOOL,
        )


@pytest.mark.asyncio
async def test_complete_wraps_groq_sdk_exception_in_llm_error() -> None:
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("sdk error"))
    service = _make_groq_service(mock_client)

    with pytest.raises(LLMError, match="Groq completion failed"):
        await service.complete(
            messages=[{"role": "user", "content": "hi"}],
            system="",
        )


@pytest.mark.asyncio
async def test_complete_includes_system_message_when_system_prompt_provided() -> None:
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_completion_response("response")
    )
    service = _make_groq_service(mock_client)

    result = await service.complete(
        messages=[{"role": "user", "content": "hi"}],
        system="you are a helpful assistant",
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"][0] == {
        "role": "system",
        "content": "you are a helpful assistant",
    }
    assert result == "response"
