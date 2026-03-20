"""Behavioral tests for the fact extraction pipeline."""

from abc import ABC, abstractmethod
from typing import Any

import pytest

from memory.models.conversation import ConversationPair, Message
from memory.pipeline.extraction import ExtractionPipeline


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
        self._tool_response = tool_response or {"facts": []}

    async def complete(self, messages: list[dict[str, str]], system: str) -> str:
        """Return empty string — not used in extraction tests."""
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
        self._tool_response = tool_response or {"facts": []}
        self.captured_messages: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]], system: str) -> str:
        """Return empty string — not used in extraction tests."""
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


def _make_pair(
    user_content: str = "I love hiking",
    assistant_content: str = "That sounds fun!",
) -> ConversationPair:
    """Build a ConversationPair with sensible defaults."""
    return ConversationPair(
        previous=Message(
            user_id="u1", session_id="s1", role="user", content=user_content
        ),
        current=Message(
            user_id="u1", session_id="s1", role="assistant", content=assistant_content
        ),
    )


@pytest.mark.asyncio
async def test_extraction_pipeline_returns_facts_from_llm_response() -> None:
    facts = ["User enjoys hiking", "User prefers outdoor activities"]
    service = FakeLLMService(tool_response={"facts": facts})
    pipeline = ExtractionPipeline(llm_service=service)

    result = await pipeline.extract(pair=_make_pair())

    assert result == facts


@pytest.mark.asyncio
async def test_extraction_pipeline_returns_empty_list_when_no_facts() -> None:
    service = FakeLLMService(tool_response={"facts": []})
    pipeline = ExtractionPipeline(llm_service=service)

    result = await pipeline.extract(pair=_make_pair())

    assert result == []


@pytest.mark.asyncio
async def test_extraction_pipeline_includes_summary_in_context() -> None:
    service = CapturingLLMService()
    pipeline = ExtractionPipeline(llm_service=service)

    await pipeline.extract(
        pair=_make_pair(),
        conversation_summary="User has been discussing hobbies.",
    )

    user_content = service.captured_messages[0][0]["content"]
    assert "Conversation Summary:" in user_content
    assert "User has been discussing hobbies." in user_content


@pytest.mark.asyncio
async def test_extraction_pipeline_includes_recent_messages_in_context() -> None:
    service = CapturingLLMService()
    pipeline = ExtractionPipeline(llm_service=service)
    recent = [
        Message(user_id="u1", session_id="s1", role="user", content="I like pizza"),
        Message(
            user_id="u1", session_id="s1", role="assistant", content="Good choice!"
        ),
    ]

    await pipeline.extract(pair=_make_pair(), recent_messages=recent)

    user_content = service.captured_messages[0][0]["content"]
    assert "Recent Messages:" in user_content
    assert "I like pizza" in user_content
    assert "Good choice!" in user_content


@pytest.mark.asyncio
async def test_extraction_pipeline_always_includes_current_pair() -> None:
    service = CapturingLLMService()
    pipeline = ExtractionPipeline(llm_service=service)

    await pipeline.extract(pair=_make_pair(user_content="I work at Google"))

    user_content = service.captured_messages[0][0]["content"]
    assert "Current Conversation:" in user_content
    assert "I work at Google" in user_content


@pytest.mark.asyncio
async def test_extraction_pipeline_handles_missing_facts_key_gracefully() -> None:
    service = FakeLLMService(tool_response={"unrelated_key": "value"})
    pipeline = ExtractionPipeline(llm_service=service)

    result = await pipeline.extract(pair=_make_pair())

    assert result == []
