"""Behavioral tests for Gemini embedding service."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm.embeddings.base import EmbeddingError
from llm.embeddings.gemini import GeminiEmbeddingService


def _make_gemini_embedding_service(mock_client: MagicMock) -> GeminiEmbeddingService:
    """Create a GeminiEmbeddingService with a pre-injected mock client."""
    service = GeminiEmbeddingService(api_key="test-key")
    service._client = mock_client
    return service


def _mock_embed_response(values: list[float]) -> MagicMock:
    """Build a mock Gemini embed_content response."""
    embedding = MagicMock()
    embedding.values = values
    response = MagicMock()
    response.embeddings = [embedding]
    return response


@pytest.mark.asyncio
async def test_embed_returns_float_list_for_single_text() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.embed_content = AsyncMock(
        return_value=_mock_embed_response([0.0, 0.6, 0.8])
    )
    service = _make_gemini_embedding_service(mock_client)

    result = await service.embed("hello world")

    call_kwargs = mock_client.aio.models.embed_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-embedding-exp-03-07"
    assert call_kwargs["contents"] == "hello world"
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


@pytest.mark.asyncio
async def test_embed_batch_returns_list_of_embeddings_for_multiple_texts() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.embed_content = AsyncMock(
        side_effect=[
            _mock_embed_response([1.0, 0.0]),
            _mock_embed_response([0.0, 1.0]),
            _mock_embed_response([0.6, 0.8]),
        ]
    )
    service = _make_gemini_embedding_service(mock_client)

    result = await service.embed_batch(["a", "b", "c"])

    assert len(result) == 3
    assert all(isinstance(item, list) for item in result)
    assert all(isinstance(v, float) for item in result for v in item)


@pytest.mark.asyncio
async def test_embed_includes_correct_model_in_sdk_call() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.embed_content = AsyncMock(
        return_value=_mock_embed_response([1.0, 0.0, 0.0])
    )
    service = GeminiEmbeddingService(api_key="test-key", model="custom-model")
    service._client = mock_client

    result = await service.embed("test")

    call_kwargs = mock_client.aio.models.embed_content.call_args.kwargs
    assert call_kwargs["model"] == "custom-model"
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_embed_raises_embedding_error_when_sdk_fails() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.embed_content = AsyncMock(
        side_effect=Exception("sdk error")
    )
    service = _make_gemini_embedding_service(mock_client)

    with pytest.raises(EmbeddingError, match="Failed to embed text"):
        await service.embed("hello")


@pytest.mark.asyncio
async def test_embed_batch_raises_embedding_error_when_any_embed_fails() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.embed_content = AsyncMock(
        side_effect=[
            _mock_embed_response([1.0, 0.0]),
            Exception("second call failed"),
        ]
    )
    service = _make_gemini_embedding_service(mock_client)

    with pytest.raises(EmbeddingError, match="Failed to embed text"):
        await service.embed_batch(["a", "b"])


@pytest.mark.asyncio
async def test_embed_returns_normalized_vector() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.embed_content = AsyncMock(
        return_value=_mock_embed_response([3.0, 4.0])
    )
    service = _make_gemini_embedding_service(mock_client)

    result = await service.embed("normalize me")

    call_kwargs = mock_client.aio.models.embed_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-embedding-exp-03-07"
    assert call_kwargs["contents"] == "normalize me"
    assert result == pytest.approx([0.6, 0.8])
