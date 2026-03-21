"""Behavioral tests for Gemini generation service."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm.generation.base import LLMError
from llm.generation.gemini import GeminiService


def _make_gemini_service(mock_client: MagicMock) -> GeminiService:
    """Create a GeminiService with a pre-injected mock client."""
    service = GeminiService(api_key="test-key")
    service._client = mock_client
    return service


def _mock_text_response(text: str) -> MagicMock:
    """Build a mock Gemini response with the given text."""
    response = MagicMock()
    response.text = text
    return response


def _mock_function_call_response(args: dict) -> MagicMock:
    """Build a mock Gemini response with a function call."""
    function_call = MagicMock()
    function_call.args = args

    part = MagicMock()
    part.function_call = function_call

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


def _mock_no_function_call_response() -> MagicMock:
    """Build a mock Gemini response with no function call (raises AttributeError on access)."""
    part = MagicMock(spec=[])  # empty spec — any attribute access raises AttributeError

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


SAMPLE_TOOL = {
    "name": "extract",
    "description": "extract facts",
    "input_schema": {"type": "object", "properties": {"facts": {"type": "array"}}},
}


@pytest.mark.asyncio
async def test_complete_returns_generated_text_from_gemini_response() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_mock_text_response("hello from gemini")
    )
    service = _make_gemini_service(mock_client)

    result = await service.complete(
        messages=[{"role": "user", "content": "hi"}],
        system="be helpful",
    )

    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.0-flash"
    assert len(call_kwargs["contents"]) == 1
    assert call_kwargs["contents"][0].role == "user"
    assert result == "hello from gemini"


@pytest.mark.asyncio
async def test_complete_with_tool_returns_function_call_arguments_as_dict() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_mock_function_call_response({"facts": ["likes coffee"]})
    )
    service = _make_gemini_service(mock_client)

    result = await service.complete_with_tool(
        messages=[{"role": "user", "content": "I like coffee"}],
        tool=SAMPLE_TOOL,
        system="extract facts",
    )

    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    tool_config = call_kwargs["config"].tool_config
    assert tool_config.function_calling_config.allowed_function_names == ["extract"]
    tools_passed = call_kwargs["config"].tools
    fn_decl = tools_passed[0].function_declarations[0]
    assert fn_decl.name == SAMPLE_TOOL["name"]
    assert fn_decl.description == SAMPLE_TOOL["description"]
    assert "facts" in fn_decl.parameters.properties
    assert result == {"facts": ["likes coffee"]}


@pytest.mark.asyncio
async def test_complete_with_tool_raises_llm_error_when_response_has_no_function_call() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_mock_no_function_call_response()
    )
    service = _make_gemini_service(mock_client)

    with pytest.raises(LLMError, match="No function_call found in response"):
        await service.complete_with_tool(
            messages=[{"role": "user", "content": "hi"}],
            tool=SAMPLE_TOOL,
        )


@pytest.mark.asyncio
async def test_complete_wraps_gemini_sdk_exception_in_llm_error() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=Exception("sdk error")
    )
    service = _make_gemini_service(mock_client)

    with pytest.raises(LLMError, match="Gemini completion failed"):
        await service.complete(
            messages=[{"role": "user", "content": "hi"}],
            system="",
        )


@pytest.mark.asyncio
async def test_complete_maps_assistant_role_to_model_for_gemini_api() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_mock_text_response("response")
    )
    service = _make_gemini_service(mock_client)

    result = await service.complete(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you"},
        ],
        system="",
    )

    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    contents = call_kwargs["contents"]
    roles_sent = [c.role for c in contents]
    assert "assistant" not in roles_sent
    assert "model" in roles_sent
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[2].role == "user"
    assert result == "response"
