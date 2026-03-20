"""Behavioral tests for the MemoryManager orchestrator."""

from abc import ABC, abstractmethod
from typing import Any

import pytest

from memory.manager import MemoryManager
from memory.models.conversation import ConversationPair, Message
from memory.models.memory import MemoryFact

# ---------------------------------------------------------------------------
# Local fakes — no imports from llm/ or storage/
# ---------------------------------------------------------------------------


class AbstractLLMService(ABC):
    """Minimal local stub matching the AbstractLLMService interface."""

    @abstractmethod
    async def complete(
        self, messages: list[dict[str, str]], system: str
    ) -> str: ...

    @abstractmethod
    async def complete_with_tool(
        self,
        messages: list[dict[str, str]],
        tool: dict[str, Any],
        system: str,
    ) -> dict[str, Any]: ...


class AbstractEmbeddingService(ABC):
    """Minimal local stub matching the AbstractEmbeddingService interface."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class AbstractMemoryStore(ABC):
    """Minimal local stub matching the AbstractMemoryStore interface."""

    @abstractmethod
    async def upsert(self, fact: MemoryFact) -> None: ...

    @abstractmethod
    async def search(
        self, user_id: str, query_embedding: list[float], top_k: int = 10
    ) -> list[MemoryFact]: ...

    @abstractmethod
    async def get(self, memory_id: str, user_id: str) -> MemoryFact | None: ...

    @abstractmethod
    async def delete(self, memory_id: str, user_id: str) -> None: ...

    @abstractmethod
    async def list_all(self, user_id: str) -> list[MemoryFact]: ...


class FakeLLMService(AbstractLLMService):
    """LLM service that sequences through extraction then evolution responses."""

    def __init__(
        self,
        extraction_response: dict[str, Any] | None = None,
        evolution_responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self._extraction_response = extraction_response or {"facts": []}
        self._evolution_responses = list(evolution_responses or [])
        self._evolution_call_index = 0

    async def complete(self, messages: list[dict[str, str]], system: str) -> str:
        """Not used in manager tests."""
        return ""

    async def complete_with_tool(
        self,
        messages: list[dict[str, str]],
        tool: dict[str, Any],
        system: str,
    ) -> dict[str, Any]:
        """Route to extraction or evolution response based on tool name."""
        if tool.get("name") == "extract_facts":
            return self._extraction_response
        # evolution call
        if self._evolution_call_index < len(self._evolution_responses):
            response = self._evolution_responses[self._evolution_call_index]
            self._evolution_call_index += 1
            return response
        return {"operation": "NOOP"}


class FakeEmbeddingService(AbstractEmbeddingService):
    """Embedding service that returns a deterministic vector and records calls."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector or [0.1, 0.2, 0.3]
        self.embed_calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        """Return a fixed vector and record the text."""
        self.embed_calls.append(text)
        return self._vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Not used in manager tests."""
        return [self._vector for _ in texts]


class FakeMemoryStore(AbstractMemoryStore):
    """In-memory store that supports upsert, get, search, and delete."""

    def __init__(self) -> None:
        self._facts: dict[str, MemoryFact] = {}
        self.search_results: list[MemoryFact] = []

    async def upsert(self, fact: MemoryFact) -> None:
        """Store a fact by id."""
        self._facts[fact.id] = fact

    async def search(
        self, user_id: str, query_embedding: list[float], top_k: int = 10
    ) -> list[MemoryFact]:
        """Return pre-configured search results."""
        return self.search_results[:top_k]

    async def get(self, memory_id: str, user_id: str) -> MemoryFact | None:
        """Retrieve a fact by id."""
        return self._facts.get(memory_id)

    async def delete(self, memory_id: str, user_id: str) -> None:
        """Remove a fact by id."""
        self._facts.pop(memory_id, None)

    async def list_all(self, user_id: str) -> list[MemoryFact]:
        """Return all stored facts."""
        return list(self._facts.values())

    @property
    def facts(self) -> dict[str, MemoryFact]:
        """Direct access to the internal store for assertions."""
        return self._facts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_ID = "u1"
_SESSION_ID = "session-1"


def _make_pair(
    user_content: str = "I work at Google",
    assistant_content: str = "Got it!",
) -> ConversationPair:
    """Build a ConversationPair with sensible defaults."""
    user_msg = Message(
        user_id=_USER_ID, session_id=_SESSION_ID, role="user", content=user_content
    )
    assistant_msg = Message(
        user_id=_USER_ID,
        session_id=_SESSION_ID,
        role="assistant",
        content=assistant_content,
    )
    return ConversationPair(current=assistant_msg, previous=user_msg)


def _build_manager(
    llm: FakeLLMService,
    embedding: FakeEmbeddingService | None = None,
    store: FakeMemoryStore | None = None,
) -> tuple[MemoryManager, FakeEmbeddingService, FakeMemoryStore]:
    """Construct a MemoryManager with fakes, returning all components."""
    emb = embedding or FakeEmbeddingService()
    st = store or FakeMemoryStore()
    manager = MemoryManager(
        store=st, embedding_service=emb, llm_service=llm
    )
    return manager, emb, st


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_extraction_returns_empty_result() -> None:
    llm = FakeLLMService(extraction_response={"facts": []})
    manager, _, _ = _build_manager(llm)

    results = await manager.add_memory(_make_pair(), user_id=_USER_ID)

    assert results == []


@pytest.mark.asyncio
async def test_add_operation_stores_fact_and_returns_it() -> None:
    llm = FakeLLMService(
        extraction_response={"facts": ["User works at Google"]},
        evolution_responses=[{"operation": "ADD"}],
    )
    manager, _, store = _build_manager(llm)

    results = await manager.add_memory(
        _make_pair(), user_id=_USER_ID, session_id=_SESSION_ID
    )

    assert len(results) == 1
    assert results[0].content == "User works at Google"
    assert results[0].user_id == _USER_ID
    assert results[0].id in store.facts


@pytest.mark.asyncio
async def test_update_operation_changes_existing_fact_content() -> None:
    store = FakeMemoryStore()
    existing = MemoryFact(id="fact-1", user_id=_USER_ID, content="User works at Meta")
    store._facts["fact-1"] = existing
    store.search_results = [existing]

    llm = FakeLLMService(
        extraction_response={"facts": ["User works at Google"]},
        evolution_responses=[
            {
                "operation": "UPDATE",
                "memory_id": "fact-1",
                "updated_content": "User works at Google",
            }
        ],
    )
    manager, _, _ = _build_manager(llm, store=store)

    results = await manager.add_memory(_make_pair(), user_id=_USER_ID)

    assert len(results) == 1
    assert store.facts["fact-1"].content == "User works at Google"


@pytest.mark.asyncio
async def test_delete_operation_removes_fact_from_store() -> None:
    store = FakeMemoryStore()
    existing = MemoryFact(
        id="fact-99", user_id=_USER_ID, content="User is vegetarian"
    )
    store._facts["fact-99"] = existing
    store.search_results = [existing]

    llm = FakeLLMService(
        extraction_response={"facts": ["User is no longer vegetarian"]},
        evolution_responses=[{"operation": "DELETE", "memory_id": "fact-99"}],
    )
    manager, _, _ = _build_manager(llm, store=store)

    results = await manager.add_memory(_make_pair(), user_id=_USER_ID)

    assert results == []
    assert "fact-99" not in store.facts


@pytest.mark.asyncio
async def test_noop_operation_makes_no_changes() -> None:
    store = FakeMemoryStore()
    existing = MemoryFact(id="fact-1", user_id=_USER_ID, content="User likes hiking")
    store._facts["fact-1"] = existing
    store.search_results = [existing]

    llm = FakeLLMService(
        extraction_response={"facts": ["User likes hiking"]},
        evolution_responses=[{"operation": "NOOP"}],
    )
    manager, _, _ = _build_manager(llm, store=store)

    results = await manager.add_memory(_make_pair(), user_id=_USER_ID)

    assert results == []
    assert store.facts["fact-1"].content == "User likes hiking"


@pytest.mark.asyncio
async def test_embed_called_for_each_candidate_fact() -> None:
    llm = FakeLLMService(
        extraction_response={"facts": ["Fact A", "Fact B", "Fact C"]},
        evolution_responses=[
            {"operation": "ADD"},
            {"operation": "ADD"},
            {"operation": "ADD"},
        ],
    )
    embedding = FakeEmbeddingService()
    manager, _, _ = _build_manager(llm, embedding=embedding)

    await manager.add_memory(_make_pair(), user_id=_USER_ID)

    assert embedding.embed_calls == ["Fact A", "Fact B", "Fact C"]


@pytest.mark.asyncio
async def test_search_memory_returns_relevant_memories() -> None:
    store = FakeMemoryStore()
    fact = MemoryFact(id="fact-1", user_id=_USER_ID, content="User likes hiking")
    store.search_results = [fact]

    llm = FakeLLMService()
    manager, embedding, _ = _build_manager(llm, store=store)

    results = await manager.search_memory("outdoor activities", user_id=_USER_ID)

    assert len(results) == 1
    assert results[0].content == "User likes hiking"
    assert "outdoor activities" in embedding.embed_calls


@pytest.mark.asyncio
async def test_update_with_missing_target_skips_gracefully() -> None:
    store = FakeMemoryStore()
    # No fact with id "ghost" exists in the store

    llm = FakeLLMService(
        extraction_response={"facts": ["User moved to NYC"]},
        evolution_responses=[
            {
                "operation": "UPDATE",
                "memory_id": "ghost",
                "updated_content": "User lives in NYC",
            }
        ],
    )
    manager, _, _ = _build_manager(llm, store=store)

    results = await manager.add_memory(_make_pair(), user_id=_USER_ID)

    assert results == []
    assert "ghost" not in store.facts


@pytest.mark.asyncio
async def test_update_operation_embeds_updated_content_not_candidate() -> None:
    """The stored embedding must match the updated content, not the candidate."""
    store = FakeMemoryStore()
    existing = MemoryFact(id="fact-1", user_id=_USER_ID, content="User works at Meta")
    store._facts["fact-1"] = existing
    store.search_results = [existing]

    embedding = FakeEmbeddingService()
    llm = FakeLLMService(
        extraction_response={"facts": ["User now works at Google"]},
        evolution_responses=[{
            "operation": "UPDATE",
            "memory_id": "fact-1",
            "updated_content": "User works at Google",
        }],
    )
    manager, _, _ = _build_manager(llm, embedding=embedding, store=store)

    await manager.add_memory(_make_pair(), user_id=_USER_ID)

    assert "User works at Google" in embedding.embed_calls
