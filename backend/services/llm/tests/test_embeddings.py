"""Behavioral tests for embedding services."""

import pytest

from llm.embeddings.base import AbstractEmbeddingService, EmbeddingError
from llm.embeddings.openai import OpenAIEmbeddingService


class FakeEmbeddingService(AbstractEmbeddingService):
    """In-memory embedding service for testing."""

    def __init__(self, dimension: int = 3) -> None:
        self._dimension = dimension

    async def embed(self, text: str) -> list[float]:
        """Return a fixed vector for any input text."""
        return [0.0] * self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one distinguishable vector per input text, indexed by position."""
        return [[float(i)] * self._dimension for i in range(len(texts))]


class FailingEmbeddingService(AbstractEmbeddingService):
    """Embedding service that always raises EmbeddingError."""

    async def embed(self, text: str) -> list[float]:
        """Always fail."""
        raise EmbeddingError("intentional failure")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Always fail."""
        raise EmbeddingError("intentional batch failure")


@pytest.mark.asyncio
async def test_embedding_service_returns_vector_for_single_text() -> None:
    service = FakeEmbeddingService()

    result = await service.embed("hello")

    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
async def test_embedding_service_returns_correct_dimension() -> None:
    service = FakeEmbeddingService(dimension=5)

    result = await service.embed("hello")

    assert len(result) == 5


@pytest.mark.asyncio
async def test_embedding_service_embed_batch_returns_one_vector_per_input() -> None:
    service = FakeEmbeddingService()

    results = await service.embed_batch(["a", "b", "c"])

    assert len(results) == 3


@pytest.mark.asyncio
async def test_embedding_service_embed_batch_preserves_input_order() -> None:
    service = FakeEmbeddingService(dimension=2)

    results = await service.embed_batch(["first", "second", "third"])

    assert results[0] == [0.0, 0.0]
    assert results[1] == [1.0, 1.0]
    assert results[2] == [2.0, 2.0]


@pytest.mark.asyncio
async def test_embedding_service_embed_batch_with_single_item() -> None:
    service = FakeEmbeddingService()

    results = await service.embed_batch(["only one"])

    assert len(results) == 1


def test_openai_service_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OpenAI API key is required"):
        OpenAIEmbeddingService(api_key=None)


def test_openai_service_accepts_explicit_api_key() -> None:
    service = OpenAIEmbeddingService(api_key="sk-test")

    assert service._model == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_embedding_error_is_raised_on_failure() -> None:
    service = FailingEmbeddingService()

    with pytest.raises(EmbeddingError, match="intentional failure"):
        await service.embed("anything")
