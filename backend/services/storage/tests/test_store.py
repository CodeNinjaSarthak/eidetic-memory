"""Tests for the abstract memory store contract using an in-memory fake."""

from datetime import UTC, datetime

import pytest

from storage.base import AbstractMemoryStore
from storage.models import MemoryFact


class FakeMemoryStore(AbstractMemoryStore):
    """In-memory implementation of AbstractMemoryStore for testing."""

    def __init__(self) -> None:
        self._facts: dict[str, MemoryFact] = {}

    async def upsert(self, fact: MemoryFact) -> None:
        """Store or overwrite a fact by its id."""
        self._facts[fact.id] = fact

    async def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[MemoryFact]:
        """Return all facts for the given user, ignoring embeddings."""
        return [f for f in self._facts.values() if f.user_id == user_id][:top_k]

    async def get(self, memory_id: str, user_id: str) -> MemoryFact | None:
        """Retrieve a fact by id if it belongs to the given user."""
        fact = self._facts.get(memory_id)
        if fact is None or fact.user_id != user_id:
            return None
        return fact

    async def delete(self, memory_id: str, user_id: str) -> None:
        """Delete a fact by id if it belongs to the given user."""
        fact = self._facts.get(memory_id)
        if fact is not None and fact.user_id == user_id:
            del self._facts[memory_id]

    async def list_all(self, user_id: str) -> list[MemoryFact]:
        """Return all facts for a user, ordered by created_at descending."""
        facts = [f for f in self._facts.values() if f.user_id == user_id]
        facts.sort(key=lambda f: f.created_at, reverse=True)
        return facts


@pytest.fixture
def store() -> FakeMemoryStore:
    """Provide a fresh FakeMemoryStore for each test."""
    return FakeMemoryStore()


def _make_fact(
    user_id: str = "user-1",
    content: str = "test fact",
    **kwargs: object,
) -> MemoryFact:
    """Create a MemoryFact with sensible defaults."""
    return MemoryFact(user_id=user_id, content=content, **kwargs)


async def test_upserted_fact_can_be_retrieved_by_id(store: FakeMemoryStore) -> None:
    fact = _make_fact()

    await store.upsert(fact)
    retrieved = await store.get(fact.id, "user-1")

    assert retrieved is not None
    assert retrieved.id == fact.id
    assert retrieved.content == "test fact"


async def test_get_returns_none_for_nonexistent_memory(store: FakeMemoryStore) -> None:
    result = await store.get("nonexistent-id", "user-1")

    assert result is None


async def test_get_returns_none_when_user_id_does_not_match(
    store: FakeMemoryStore,
) -> None:
    fact = _make_fact(user_id="user-1")
    await store.upsert(fact)

    result = await store.get(fact.id, "user-999")

    assert result is None


async def test_deleted_fact_cannot_be_retrieved(store: FakeMemoryStore) -> None:
    fact = _make_fact()
    await store.upsert(fact)

    await store.delete(fact.id, "user-1")
    result = await store.get(fact.id, "user-1")

    assert result is None


async def test_list_all_returns_only_facts_belonging_to_user(
    store: FakeMemoryStore,
) -> None:
    fact_a = _make_fact(user_id="user-1", content="fact A")
    fact_b = _make_fact(user_id="user-2", content="fact B")
    await store.upsert(fact_a)
    await store.upsert(fact_b)

    results = await store.list_all("user-1")

    assert len(results) == 1
    assert results[0].user_id == "user-1"


async def test_list_all_returns_facts_ordered_by_created_at_descending(
    store: FakeMemoryStore,
) -> None:
    older = _make_fact(
        content="older",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    newer = _make_fact(
        content="newer",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    await store.upsert(older)
    await store.upsert(newer)

    results = await store.list_all("user-1")

    assert results[0].content == "newer"
    assert results[1].content == "older"


async def test_search_returns_facts_for_correct_user_only(
    store: FakeMemoryStore,
) -> None:
    fact_a = _make_fact(user_id="user-1", content="fact A")
    fact_b = _make_fact(user_id="user-2", content="fact B")
    await store.upsert(fact_a)
    await store.upsert(fact_b)

    results = await store.search("user-1", query_embedding=[0.1, 0.2])

    assert len(results) == 1
    assert results[0].user_id == "user-1"


async def test_upsert_overwrites_existing_fact_with_same_id(
    store: FakeMemoryStore,
) -> None:
    fact = _make_fact(content="original")
    await store.upsert(fact)

    updated = fact.model_copy(update={"content": "updated"})
    await store.upsert(updated)
    retrieved = await store.get(fact.id, "user-1")

    assert retrieved is not None
    assert retrieved.content == "updated"
