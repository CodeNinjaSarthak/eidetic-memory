"""Integration tests for access control on the read-only demo endpoint."""

from datetime import UTC, datetime

import pytest
from api.dependencies import get_demo_retriever, get_llm_service, get_settings
from api.main import app, should_mount_write_routers
from httpx import ASGITransport, AsyncClient

from config.settings import Settings
from storage.models import MemoryFact

_BASE_SETTINGS_KWARGS = {
    "_env_file": None,
    "qdrant_url": "http://localhost:6333",
    "llm_provider": "claude",
    "anthropic_api_key": "sk-ant-test-key",
    "embedding_provider": "openai",
    "openai_api_key": "sk-openai-test-key",
}


class FakeDemoRetriever:
    """Stub retriever returning one canned memory regardless of the query."""

    async def retrieve(
        self, query: str, user_id: str, top_k: int | None = None
    ) -> list[MemoryFact]:
        return [
            MemoryFact(
                id="fact-1",
                user_id=user_id,
                content="Caroline mentioned she paints on weekends",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ]


class FakeDemoLLMService:
    """Stub LLM service that ignores prompt args and returns a canned reply."""

    async def complete(
        self,
        messages: list,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 200,
    ) -> str:
        return "Caroline paints on weekends."

    async def complete_with_tool(
        self, messages: list, tool: dict, system: str = ""
    ) -> dict:
        return {}


class ExplodingDemoLLMService:
    """LLM stub that fails the test if it is ever called."""

    async def complete(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("LLM must not be called for a rejected credential")

    async def complete_with_tool(self, *args: object, **kwargs: object) -> dict:
        raise AssertionError("LLM must not be called for a rejected credential")


class ExplodingDemoRetriever:
    """Retriever stub that fails the test if it is ever called."""

    async def retrieve(self, *args: object, **kwargs: object) -> list[MemoryFact]:
        raise AssertionError("Retrieval must not run for a rejected credential")


DEMO_QUERY_BODY = {
    "conversation_id": "conv-26",
    "question": "What does Caroline do on weekends?",
    "question_type": "factual",
}


@pytest.fixture()
def client_factory():
    """Build an async test client with demo dependencies overridden per test."""

    def _make(settings: Settings) -> AsyncClient:
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_demo_retriever] = lambda: FakeDemoRetriever()
        app.dependency_overrides[get_llm_service] = lambda: FakeDemoLLMService()
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    yield _make
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_demo_query_succeeds_without_credentials_when_unset_in_development(
    client_factory,
):
    settings = Settings(**_BASE_SETTINGS_KWARGS, api_env="development")

    async with client_factory(settings) as client:
        response = await client.post("/demo/query", json=DEMO_QUERY_BODY)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_demo_query_is_forbidden_without_credentials_when_unset_in_production(
    client_factory,
):
    settings = Settings(**_BASE_SETTINGS_KWARGS, api_env="production")

    async with client_factory(settings) as client:
        response = await client.post("/demo/query", json=DEMO_QUERY_BODY)

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_demo_query_returns_401_without_credentials_when_configured(
    client_factory,
):
    settings = Settings(
        **_BASE_SETTINGS_KWARGS, demo_username="reviewer", demo_password="s3cret"
    )

    async with client_factory(settings) as client:
        response = await client.post("/demo/query", json=DEMO_QUERY_BODY)

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_demo_query_returns_200_with_correct_credentials(client_factory):
    settings = Settings(
        **_BASE_SETTINGS_KWARGS, demo_username="reviewer", demo_password="s3cret"
    )

    async with client_factory(settings) as client:
        response = await client.post(
            "/demo/query", json=DEMO_QUERY_BODY, auth=("reviewer", "s3cret")
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Caroline paints on weekends."


@pytest.mark.asyncio
async def test_demo_query_returns_401_with_wrong_password(client_factory):
    settings = Settings(
        **_BASE_SETTINGS_KWARGS, demo_username="reviewer", demo_password="s3cret"
    )

    async with client_factory(settings) as client:
        response = await client.post(
            "/demo/query", json=DEMO_QUERY_BODY, auth=("reviewer", "wrong")
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_demo_query_rejects_wrong_credentials_before_calling_the_llm(
    client_factory,
):
    settings = Settings(
        **_BASE_SETTINGS_KWARGS, demo_username="reviewer", demo_password="s3cret"
    )

    async with client_factory(settings) as client:
        app.dependency_overrides[get_llm_service] = lambda: ExplodingDemoLLMService()
        app.dependency_overrides[get_demo_retriever] = lambda: ExplodingDemoRetriever()
        response = await client.post(
            "/demo/query", json=DEMO_QUERY_BODY, auth=("reviewer", "wrong")
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_check_returns_401_without_credentials_when_configured(
    client_factory,
):
    settings = Settings(
        **_BASE_SETTINGS_KWARGS, demo_username="reviewer", demo_password="s3cret"
    )

    async with client_factory(settings) as client:
        response = await client.get("/demo/auth-check")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_check_returns_401_with_wrong_credentials(client_factory):
    settings = Settings(
        **_BASE_SETTINGS_KWARGS, demo_username="reviewer", demo_password="s3cret"
    )

    async with client_factory(settings) as client:
        response = await client.get("/demo/auth-check", auth=("reviewer", "wrong"))

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_check_returns_200_ok_with_correct_credentials(client_factory):
    settings = Settings(
        **_BASE_SETTINGS_KWARGS, demo_username="reviewer", demo_password="s3cret"
    )

    async with client_factory(settings) as client:
        response = await client.get("/demo/auth-check", auth=("reviewer", "s3cret"))

    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.asyncio
async def test_auth_check_is_open_without_credentials_in_development(client_factory):
    settings = Settings(**_BASE_SETTINGS_KWARGS, api_env="development")

    async with client_factory(settings) as client:
        response = await client.get("/demo/auth-check")

    assert response.status_code == 200


def test_write_routers_are_not_mounted_in_demo_only_mode():
    settings = Settings(**_BASE_SETTINGS_KWARGS, demo_only_mode=True)

    assert should_mount_write_routers(settings) is False


def test_write_routers_are_mounted_by_default():
    settings = Settings(**_BASE_SETTINGS_KWARGS)

    assert should_mount_write_routers(settings) is True
