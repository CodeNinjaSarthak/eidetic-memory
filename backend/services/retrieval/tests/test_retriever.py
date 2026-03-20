"""Behavioral tests for MemoryRetriever."""

import pytest

from retrieval.retriever import MemoryRetriever
from storage.models import MemoryFact

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbeddingService:
    """Records embed calls and returns a fixed vector."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.calls: list[str] = []
        self._vector = vector or [0.1, 0.2, 0.3]

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class FakeMemoryStore:
    """Returns pre-configured facts from search, records call arguments."""

    def __init__(self, facts: list[MemoryFact] | None = None) -> None:
        self._facts = facts or []
        self.search_calls: list[dict] = []

    async def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[MemoryFact]:
        self.search_calls.append(
            {"user_id": user_id, "query_embedding": query_embedding, "top_k": top_k}
        )
        return list(self._facts)

    async def upsert(self, fact: MemoryFact) -> None:
        pass

    async def get(self, memory_id: str, user_id: str) -> MemoryFact | None:
        return None

    async def delete(self, memory_id: str, user_id: str) -> None:
        pass

    async def list_all(self, user_id: str) -> list[MemoryFact]:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fact(
    content: str,
    importance_score: float | None = None,
    user_id: str = "u1",
) -> MemoryFact:
    return MemoryFact(
        user_id=user_id,
        content=content,
        importance_score=importance_score,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_when_store_has_no_results():
    store = FakeMemoryStore(facts=[])
    embedder = FakeEmbeddingService()
    retriever = MemoryRetriever(store=store, embedding_service=embedder)

    results = await retriever.retrieve(query="anything", user_id="u1")

    assert results == []


@pytest.mark.asyncio
async def test_retrieve_returns_facts_from_store():
    facts = [_make_fact("likes coffee"), _make_fact("works at Acme")]
    store = FakeMemoryStore(facts=facts)
    embedder = FakeEmbeddingService()
    retriever = MemoryRetriever(store=store, embedding_service=embedder, rerank_by_importance=False)

    results = await retriever.retrieve(query="tell me about the user", user_id="u1")

    assert [f.content for f in results] == ["likes coffee", "works at Acme"]


@pytest.mark.asyncio
async def test_retrieve_embeds_the_query_text():
    store = FakeMemoryStore()
    embedder = FakeEmbeddingService()
    retriever = MemoryRetriever(store=store, embedding_service=embedder)

    await retriever.retrieve(query="what does the user like?", user_id="u1")

    assert embedder.calls == ["what does the user like?"]


@pytest.mark.asyncio
async def test_retrieve_passes_top_k_to_store():
    store = FakeMemoryStore()
    embedder = FakeEmbeddingService()
    retriever = MemoryRetriever(store=store, embedding_service=embedder, top_k=7)

    await retriever.retrieve(query="q", user_id="u1")

    assert store.search_calls[0]["top_k"] == 7


@pytest.mark.asyncio
async def test_retrieve_uses_instance_top_k_as_default():
    store = FakeMemoryStore()
    embedder = FakeEmbeddingService()
    retriever = MemoryRetriever(store=store, embedding_service=embedder)

    await retriever.retrieve(query="q", user_id="u1")

    assert store.search_calls[0]["top_k"] == 5


@pytest.mark.asyncio
async def test_retrieve_reranks_by_importance_score_when_all_scores_present():
    facts = [
        _make_fact("low", importance_score=0.2),
        _make_fact("high", importance_score=0.9),
        _make_fact("mid", importance_score=0.5),
    ]
    store = FakeMemoryStore(facts=facts)
    embedder = FakeEmbeddingService()
    retriever = MemoryRetriever(store=store, embedding_service=embedder)

    results = await retriever.retrieve(query="q", user_id="u1")

    assert [f.content for f in results] == ["high", "mid", "low"]


@pytest.mark.asyncio
async def test_retrieve_preserves_store_order_when_scores_are_missing():
    facts = [
        _make_fact("first", importance_score=0.9),
        _make_fact("second", importance_score=None),
    ]
    store = FakeMemoryStore(facts=facts)
    embedder = FakeEmbeddingService()
    retriever = MemoryRetriever(store=store, embedding_service=embedder)

    results = await retriever.retrieve(query="q", user_id="u1")

    assert [f.content for f in results] == ["first", "second"]


@pytest.mark.asyncio
async def test_retrieve_preserves_store_order_when_reranking_disabled():
    facts = [
        _make_fact("low", importance_score=0.1),
        _make_fact("high", importance_score=0.9),
    ]
    store = FakeMemoryStore(facts=facts)
    embedder = FakeEmbeddingService()
    retriever = MemoryRetriever(
        store=store, embedding_service=embedder, rerank_by_importance=False
    )

    results = await retriever.retrieve(query="q", user_id="u1")

    assert [f.content for f in results] == ["low", "high"]


@pytest.mark.asyncio
async def test_retrieve_top_k_override_limits_returned_results():
    facts = [
        _make_fact("a", importance_score=0.9),
        _make_fact("b", importance_score=0.8),
        _make_fact("c", importance_score=0.7),
    ]
    store = FakeMemoryStore(facts=facts)
    embedder = FakeEmbeddingService()
    retriever = MemoryRetriever(store=store, embedding_service=embedder, top_k=10)

    results = await retriever.retrieve(query="q", user_id="u1", top_k=2)

    assert len(results) == 2
    assert store.search_calls[0]["top_k"] == 2
