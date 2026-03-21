"""Tests for memory service data models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from memory.models.conversation import ConversationPair, Message
from memory.models.memory import MemoryFact, MemoryOperation, MemoryUpdate


def test_memory_fact_sets_defaults_when_created_with_required_fields_only() -> None:
    fact = MemoryFact(user_id="user-1", content="Likes Python")

    assert fact.user_id == "user-1"
    assert fact.content == "Likes Python"
    assert fact.id
    assert fact.session_id is None
    assert fact.embedding is None
    assert fact.importance_score is None
    assert fact.source_message_ids == []
    assert fact.metadata == {}
    assert isinstance(fact.created_at, datetime)
    assert isinstance(fact.updated_at, datetime)


def test_memory_fact_excludes_embedding_from_qdrant_payload() -> None:
    fact = MemoryFact(
        user_id="user-1",
        content="Likes Python",
        embedding=[0.1, 0.2, 0.3],
    )

    payload = fact.to_qdrant_payload()

    assert "embedding" not in payload
    assert payload["user_id"] == "user-1"
    assert payload["content"] == "Likes Python"


def test_memory_fact_update_content_changes_content_and_refreshes_timestamp() -> None:
    fact = MemoryFact(user_id="user-1", content="Old content")
    original_updated_at = fact.updated_at

    fact.update_content("New content")

    assert fact.content == "New content"
    assert fact.updated_at >= original_updated_at


def test_memory_fact_rejects_importance_score_above_one() -> None:
    with pytest.raises(ValidationError):
        MemoryFact(user_id="u", content="c", importance_score=1.5)


def test_memory_fact_rejects_importance_score_below_zero() -> None:
    with pytest.raises(ValidationError):
        MemoryFact(user_id="u", content="c", importance_score=-0.1)


def test_memory_fact_serializes_datetime_fields_as_iso_strings() -> None:
    fact = MemoryFact(user_id="u", content="c")

    data = fact.model_dump(mode="json")

    assert isinstance(data["created_at"], str)
    assert isinstance(data["updated_at"], str)


def test_memory_operation_enum_exposes_all_pipeline_operations() -> None:
    assert MemoryOperation.ADD == "ADD"
    assert MemoryOperation.UPDATE == "UPDATE"
    assert MemoryOperation.DELETE == "DELETE"
    assert MemoryOperation.NOOP == "NOOP"


def test_memory_update_accepts_valid_operation() -> None:
    update = MemoryUpdate(operation=MemoryOperation.ADD)

    assert update.operation == MemoryOperation.ADD
    assert update.memory_id is None


def test_conversation_pair_rejects_messages_from_different_sessions() -> None:
    user_message = Message(
        user_id="u1", session_id="session-A", role="user", content="hello"
    )
    assistant_message = Message(
        user_id="u1",
        session_id="session-B",
        role="assistant",
        content="hi",
    )

    with pytest.raises(ValidationError, match="same session"):
        ConversationPair(current=assistant_message, previous=user_message)


def test_conversation_pair_accepts_messages_from_same_session() -> None:
    user_message = Message(user_id="u1", session_id="s1", role="user", content="hello")
    assistant_message = Message(
        user_id="u1", session_id="s1", role="assistant", content="hi"
    )

    pair = ConversationPair(current=assistant_message, previous=user_message)

    assert pair.current.role == "assistant"
    assert pair.previous.role == "user"


def test_message_generates_id_and_timestamp_by_default() -> None:
    msg = Message(user_id="u1", session_id="s1", role="user", content="hi")

    assert msg.id
    assert isinstance(msg.timestamp, datetime)
