"""Conversation and message models for the memory extraction pipeline."""

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class Message(BaseModel):
    """A single message in a conversation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationPair(BaseModel):
    """A pair of consecutive messages used as input to the extraction phase."""

    current: Message
    previous: Message

    @model_validator(mode="after")
    def validate_same_session(self) -> "ConversationPair":
        """Ensure both messages belong to the same session."""
        if self.current.session_id != self.previous.session_id:
            msg = (
                f"Messages must belong to the same session. "
                f"Got current={self.current.session_id!r}, "
                f"previous={self.previous.session_id!r}"
            )
            raise ValueError(msg)
        return self
