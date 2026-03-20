"""Behavioral tests for the memory evolution engine."""

from abc import ABC, abstractmethod
from typing import Any

import pytest

from memory.models.memory import MemoryFact, MemoryOperation
from memory.pipeline.update import EvolutionEngine


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


class FakeLLMService(AbstractLLMService):
    """In-memory LLM service that returns canned tool responses."""

    def __init__(self, tool_response: dict[str, Any] | None = None) -> None:
        self._tool_response = tool_response or {"operation": "NOOP"}

    async def complete(self, messages: list[dict[str, str]], system: str) -> str:
        """Return empty string — not used in evolution tests."""
        return ""

    async def complete_with_tool(
        self,
        messages: list[dict[str, str]],
        tool: dict[str, Any],
        system: str,
    ) -> dict[str, Any]:
        """Return the canned tool response."""
        return self._tool_response


class CapturingLLMService(AbstractLLMService):
    """LLM service that captures the messages it receives."""

    def __init__(self, tool_response: dict[str, Any] | None = None) -> None:
        self._tool_response = tool_response or {"operation": "NOOP"}
        self.captured_messages: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]], system: str) -> str:
        """Return empty string — not used in evolution tests."""
        return ""

    async def complete_with_tool(
        self,
        messages: list[dict[str, str]],
        tool: dict[str, Any],
        system: str,
    ) -> dict[str, Any]:
        """Capture messages and return canned response."""
        self.captured_messages.append(messages)
        return self._tool_response


def _make_fact(
    content: str = "User likes hiking",
    fact_id: str = "fact-1",
    user_id: str = "u1",
) -> MemoryFact:
    """Build a MemoryFact with sensible defaults."""
    return MemoryFact(id=fact_id, user_id=user_id, content=content)


@pytest.mark.asyncio
async def test_evolution_engine_returns_add_when_llm_says_add() -> None:
    service = FakeLLMService(
        tool_response={"operation": "ADD", "updated_content": "User works at Google"}
    )
    engine = EvolutionEngine(llm_service=service)

    result = await engine.decide("User works at Google", existing_memories=[])

    assert result.operation == MemoryOperation.ADD
    assert result.updated_content == "User works at Google"


@pytest.mark.asyncio
async def test_evolution_engine_returns_update_with_memory_id_and_content() -> None:
    existing = [_make_fact(content="User works at Meta", fact_id="fact-42")]
    service = FakeLLMService(
        tool_response={
            "operation": "UPDATE",
            "memory_id": "fact-42",
            "updated_content": "User works at Google",
        }
    )
    engine = EvolutionEngine(llm_service=service)

    result = await engine.decide("User works at Google", existing_memories=existing)

    assert result.operation == MemoryOperation.UPDATE
    assert result.memory_id == "fact-42"
    assert result.updated_content == "User works at Google"


@pytest.mark.asyncio
async def test_evolution_engine_returns_delete_with_memory_id() -> None:
    existing = [_make_fact(content="User is vegetarian", fact_id="fact-99")]
    service = FakeLLMService(
        tool_response={"operation": "DELETE", "memory_id": "fact-99"}
    )
    engine = EvolutionEngine(llm_service=service)

    result = await engine.decide(
        "User is no longer vegetarian", existing_memories=existing
    )

    assert result.operation == MemoryOperation.DELETE
    assert result.memory_id == "fact-99"


@pytest.mark.asyncio
async def test_evolution_engine_returns_noop_when_llm_says_noop() -> None:
    existing = [_make_fact(content="User likes hiking")]
    service = FakeLLMService(tool_response={"operation": "NOOP"})
    engine = EvolutionEngine(llm_service=service)

    result = await engine.decide("User likes hiking", existing_memories=existing)

    assert result.operation == MemoryOperation.NOOP


@pytest.mark.asyncio
async def test_evolution_engine_falls_back_to_noop_on_invalid_operation() -> None:
    service = FakeLLMService(tool_response={"operation": "MERGE"})
    engine = EvolutionEngine(llm_service=service)

    result = await engine.decide("User likes coffee", existing_memories=[])

    assert result.operation == MemoryOperation.NOOP


@pytest.mark.asyncio
async def test_evolution_engine_falls_back_to_noop_when_operation_missing() -> None:
    service = FakeLLMService(tool_response={"unrelated": "value"})
    engine = EvolutionEngine(llm_service=service)

    result = await engine.decide("User likes coffee", existing_memories=[])

    assert result.operation == MemoryOperation.NOOP


@pytest.mark.asyncio
async def test_evolution_engine_falls_back_to_noop_when_update_missing_memory_id() -> None:
    service = FakeLLMService(
        tool_response={"operation": "UPDATE", "updated_content": "User likes tea"}
    )
    engine = EvolutionEngine(llm_service=service)

    result = await engine.decide("User likes tea", existing_memories=[])

    assert result.operation == MemoryOperation.NOOP


@pytest.mark.asyncio
async def test_evolution_engine_falls_back_to_noop_when_delete_missing_memory_id() -> None:
    service = FakeLLMService(tool_response={"operation": "DELETE"})
    engine = EvolutionEngine(llm_service=service)

    result = await engine.decide("User is not vegetarian", existing_memories=[])

    assert result.operation == MemoryOperation.NOOP


@pytest.mark.asyncio
async def test_evolution_engine_falls_back_to_noop_when_update_missing_content() -> None:
    service = FakeLLMService(
        tool_response={"operation": "UPDATE", "memory_id": "fact-1"}
    )
    engine = EvolutionEngine(llm_service=service)

    result = await engine.decide("User likes tea", existing_memories=[])

    assert result.operation == MemoryOperation.NOOP


@pytest.mark.asyncio
async def test_evolution_engine_context_includes_candidate_fact() -> None:
    service = CapturingLLMService()
    engine = EvolutionEngine(llm_service=service)

    await engine.decide("User speaks French", existing_memories=[])

    user_content = service.captured_messages[0][0]["content"]
    assert "Candidate Fact:" in user_content
    assert "User speaks French" in user_content


@pytest.mark.asyncio
async def test_evolution_engine_context_includes_existing_memories() -> None:
    existing = [
        _make_fact(content="User speaks Spanish", fact_id="fact-1"),
        _make_fact(content="User lives in Paris", fact_id="fact-2"),
    ]
    service = CapturingLLMService()
    engine = EvolutionEngine(llm_service=service)

    await engine.decide("User speaks French", existing_memories=existing)

    user_content = service.captured_messages[0][0]["content"]
    assert "Existing Memories:" in user_content
    assert "[ID: fact-1] User speaks Spanish" in user_content
    assert "[ID: fact-2] User lives in Paris" in user_content


@pytest.mark.asyncio
async def test_evolution_engine_context_shows_none_when_no_existing_memories() -> None:
    service = CapturingLLMService()
    engine = EvolutionEngine(llm_service=service)

    await engine.decide("User likes sushi", existing_memories=[])

    user_content = service.captured_messages[0][0]["content"]
    assert "Existing Memories:\nNone" in user_content
