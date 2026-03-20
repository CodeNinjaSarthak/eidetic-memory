"""Tests for the FastAPI memory API endpoints."""

from datetime import UTC, datetime

import pytest
from api.dependencies import get_memory_manager, get_memory_retriever
from api.main import app
from httpx import ASGITransport, AsyncClient

from storage.models import MemoryFact


class FakeMemoryManager:
    """Stub that returns a canned MemoryFact from add_memory."""

    async def add_memory(
        self,
        pair: object,
        user_id: str,
        session_id: str | None = None,
        conversation_summary: str | None = None,
        recent_messages: list | None = None,
    ) -> list[MemoryFact]:
        return [
            MemoryFact(
                id="fact-1",
                user_id=user_id,
                content="User likes jazz",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ]


class FakeMemoryRetriever:
    """Stub that returns a canned MemoryFact from retrieve."""

    async def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int | None = None,
    ) -> list[MemoryFact]:
        return [
            MemoryFact(
                id="fact-2",
                user_id=user_id,
                content="User works at Acme Corp",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ]


@pytest.fixture()
def client():
    """Async test client with fake dependencies injected."""
    app.dependency_overrides[get_memory_manager] = lambda: FakeMemoryManager()
    app.dependency_overrides[get_memory_retriever] = lambda: FakeMemoryRetriever()
    yield AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    )
    app.dependency_overrides.clear()


ADD_MEMORY_BODY = {
    "user_id": "u1",
    "session_id": "s1",
    "current_message": {
        "role": "user",
        "content": "I really love jazz music",
        "user_id": "u1",
        "session_id": "s1",
    },
    "previous_message": {
        "role": "assistant",
        "content": "What kind of music do you like?",
        "user_id": "u1",
        "session_id": "s1",
    },
}

SEARCH_BODY = {
    "query": "where does the user work",
    "user_id": "u1",
    "top_k": 5,
}


@pytest.mark.asyncio
async def test_health_check_returns_ok(client: AsyncClient):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_add_memory_returns_200_with_added_facts(client: AsyncClient):
    response = await client.post("/memories/", json=ADD_MEMORY_BODY)

    assert response.status_code == 200
    assert "added" in response.json()
    assert len(response.json()["added"]) == 1


@pytest.mark.asyncio
async def test_add_memory_response_contains_expected_fields(client: AsyncClient):
    response = await client.post("/memories/", json=ADD_MEMORY_BODY)

    fact = response.json()["added"][0]
    assert fact["id"] == "fact-1"
    assert fact["user_id"] == "u1"
    assert fact["content"] == "User likes jazz"
    assert "created_at" in fact
    assert "updated_at" in fact


@pytest.mark.asyncio
async def test_search_memories_returns_200_with_results(client: AsyncClient):
    response = await client.post("/memories/search", json=SEARCH_BODY)

    assert response.status_code == 200
    assert "memories" in response.json()
    assert len(response.json()["memories"]) == 1


@pytest.mark.asyncio
async def test_search_memories_response_contains_memory_content(client: AsyncClient):
    response = await client.post("/memories/search", json=SEARCH_BODY)

    memory = response.json()["memories"][0]
    assert memory["content"] == "User works at Acme Corp"
