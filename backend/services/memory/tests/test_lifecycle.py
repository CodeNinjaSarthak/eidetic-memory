"""Behavioral tests for the LifecycleManager."""

from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from memory.lifecycle import LifecycleManager
from memory.models.memory import MemoryFact

# ---------------------------------------------------------------------------
# Local fakes — no imports from storage/
# ---------------------------------------------------------------------------


class AbstractMemoryStore(ABC):
    """Minimal local stub matching the AbstractMemoryStore interface."""

    @abstractmethod
    async def upsert(self, fact: Any) -> None: ...

    @abstractmethod
    async def search(
        self, user_id: str, query_embedding: list[float], top_k: int = 10
    ) -> list[Any]: ...

    @abstractmethod
    async def get(self, memory_id: str, user_id: str) -> Any | None: ...

    @abstractmethod
    async def delete(self, memory_id: str, user_id: str) -> None: ...

    @abstractmethod
    async def list_all(self, user_id: str) -> list[Any]: ...


class FakeMemoryStore(AbstractMemoryStore):
    """In-memory store for lifecycle tests."""

    def __init__(self) -> None:
        self._facts: dict[str, Any] = {}

    async def upsert(self, fact: Any) -> None:
        """Store a fact by id."""
        self._facts[fact.id] = fact

    async def search(
        self, user_id: str, query_embedding: list[float], top_k: int = 10
    ) -> list[Any]:
        """Not used in lifecycle tests."""
        return []

    async def get(self, memory_id: str, user_id: str) -> Any | None:
        """Retrieve a fact by id, only if user_id matches."""
        fact = self._facts.get(memory_id)
        if fact is not None and fact.user_id == user_id:
            return fact
        return None

    async def delete(self, memory_id: str, user_id: str) -> None:
        """Remove a fact by id."""
        self._facts.pop(memory_id, None)

    async def list_all(self, user_id: str) -> list[Any]:
        """Return all facts for a given user."""
        return [f for f in self._facts.values() if f.user_id == user_id]

    @property
    def facts(self) -> dict[str, Any]:
        """Direct access to the internal store for assertions."""
        return self._facts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_ID = "u1"


def _make_fact(
    *,
    id: str = "fact-1",
    content: str = "User likes hiking",
    days_ago: float = 0.0,
    access_count: int = 0,
    user_id: str = _USER_ID,
) -> MemoryFact:
    """Build a MemoryFact with controllable age and access count."""
    now = datetime.now(UTC)
    ts = now - timedelta(days=days_ago)
    return MemoryFact(
        id=id,
        user_id=user_id,
        content=content,
        created_at=ts,
        updated_at=ts,
        metadata={"access_count": access_count},
    )


def _build_manager(
    store: FakeMemoryStore | None = None,
    recency_weight: float = 0.7,
    decay_days: float = 30.0,
    frequency_cap: int = 10,
) -> tuple[LifecycleManager, FakeMemoryStore]:
    """Construct a LifecycleManager with a fake store."""
    st = store or FakeMemoryStore()
    manager = LifecycleManager(
        store=st,
        recency_weight=recency_weight,
        decay_days=decay_days,
        frequency_cap=frequency_cap,
    )
    return manager, st


# ---------------------------------------------------------------------------
# Tests — Importance Scoring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recent_memory_scores_higher_than_old_memory() -> None:
    store = FakeMemoryStore()
    recent = _make_fact(id="recent", days_ago=1.0)
    old = _make_fact(id="old", days_ago=60.0)
    store._facts["recent"] = recent
    store._facts["old"] = old
    manager, _ = _build_manager(store=store)

    scored = await manager.score_memories(_USER_ID)
    by_id = {f.id: f for f in scored}

    assert by_id["recent"].importance_score > by_id["old"].importance_score


@pytest.mark.asyncio
async def test_frequently_accessed_memory_scores_higher() -> None:
    store = FakeMemoryStore()
    popular = _make_fact(id="popular", days_ago=10.0, access_count=8)
    ignored = _make_fact(id="ignored", days_ago=10.0, access_count=0)
    store._facts["popular"] = popular
    store._facts["ignored"] = ignored
    manager, _ = _build_manager(store=store)

    scored = await manager.score_memories(_USER_ID)
    by_id = {f.id: f for f in scored}

    assert by_id["popular"].importance_score > by_id["ignored"].importance_score


@pytest.mark.asyncio
async def test_importance_score_is_between_zero_and_one() -> None:
    store = FakeMemoryStore()
    store._facts["f1"] = _make_fact(id="f1", days_ago=0.0, access_count=10)
    store._facts["f2"] = _make_fact(id="f2", days_ago=365.0, access_count=0)
    manager, _ = _build_manager(store=store)

    scored = await manager.score_memories(_USER_ID)

    for fact in scored:
        assert 0.0 <= fact.importance_score <= 1.0


@pytest.mark.asyncio
async def test_score_memories_persists_scores_via_upsert() -> None:
    store = FakeMemoryStore()
    store._facts["f1"] = _make_fact(id="f1")
    manager, _ = _build_manager(store=store)

    await manager.score_memories(_USER_ID)

    assert store.facts["f1"].importance_score is not None


@pytest.mark.asyncio
async def test_frequency_score_is_capped() -> None:
    store = FakeMemoryStore()
    capped = _make_fact(id="capped", days_ago=10.0, access_count=100)
    at_cap = _make_fact(id="at-cap", days_ago=10.0, access_count=10)
    store._facts["capped"] = capped
    store._facts["at-cap"] = at_cap
    manager, _ = _build_manager(store=store, frequency_cap=10)

    scored = await manager.score_memories(_USER_ID)
    by_id = {f.id: f for f in scored}

    assert by_id["capped"].importance_score == pytest.approx(
        by_id["at-cap"].importance_score
    )


# ---------------------------------------------------------------------------
# Tests — Pruning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prune_removes_memories_below_min_importance() -> None:
    store = FakeMemoryStore()
    store._facts["old"] = _make_fact(id="old", days_ago=365.0, access_count=0)
    store._facts["fresh"] = _make_fact(id="fresh", days_ago=0.0, access_count=5)
    manager, _ = _build_manager(store=store)

    deleted = await manager.prune_memories(_USER_ID, min_importance=0.2)

    assert "old" in deleted
    assert "fresh" not in deleted
    assert "old" not in store.facts
    assert "fresh" in store.facts


@pytest.mark.asyncio
async def test_prune_respects_max_memories_limit() -> None:
    store = FakeMemoryStore()
    for i in range(5):
        store._facts[f"f{i}"] = _make_fact(
            id=f"f{i}", days_ago=float(i * 10), access_count=0
        )
    manager, _ = _build_manager(store=store)

    deleted = await manager.prune_memories(_USER_ID, min_importance=0.0, max_memories=2)

    assert len(store.facts) <= 2
    assert len(deleted) >= 3


@pytest.mark.asyncio
async def test_prune_returns_empty_list_when_all_memories_are_important() -> None:
    store = FakeMemoryStore()
    store._facts["f1"] = _make_fact(id="f1", days_ago=0.0, access_count=5)
    store._facts["f2"] = _make_fact(id="f2", days_ago=1.0, access_count=3)
    manager, _ = _build_manager(store=store)

    deleted = await manager.prune_memories(_USER_ID, min_importance=0.1)

    assert deleted == []
    assert len(store.facts) == 2


# ---------------------------------------------------------------------------
# Tests — Access Recording
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_access_increments_access_count() -> None:
    store = FakeMemoryStore()
    store._facts["f1"] = _make_fact(id="f1", access_count=3)
    manager, _ = _build_manager(store=store)

    await manager.record_access("f1", _USER_ID)

    assert store.facts["f1"].metadata["access_count"] == 4


@pytest.mark.asyncio
async def test_record_access_initializes_count_when_missing() -> None:
    store = FakeMemoryStore()
    fact = MemoryFact(id="f1", user_id=_USER_ID, content="test")
    store._facts["f1"] = fact
    manager, _ = _build_manager(store=store)

    await manager.record_access("f1", _USER_ID)

    assert store.facts["f1"].metadata["access_count"] == 1


@pytest.mark.asyncio
async def test_record_access_skips_gracefully_for_missing_memory() -> None:
    store = FakeMemoryStore()
    manager, _ = _build_manager(store=store)

    await manager.record_access("nonexistent", _USER_ID)

    assert len(store.facts) == 0


@pytest.mark.asyncio
async def test_record_access_returns_the_updated_fact() -> None:
    store = FakeMemoryStore()
    store._facts["f1"] = _make_fact(id="f1", access_count=0)
    manager, _ = _build_manager(store=store)

    result = await manager.record_access("f1", _USER_ID)

    assert result is not None
    assert result.metadata["access_count"] == 1
