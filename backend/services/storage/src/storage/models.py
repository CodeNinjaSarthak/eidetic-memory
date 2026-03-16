"""Data models for the storage layer."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class MemoryFact(BaseModel):
    """A single extracted memory fact stored in the memory system."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    session_id: str | None = None
    content: str
    embedding: list[float] | None = None
    importance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_message_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("created_at", "updated_at")
    @classmethod
    def serialize_datetime(cls, value: datetime) -> str:
        """Serialize datetime fields to ISO 8601 format."""
        return value.isoformat()

    def to_qdrant_payload(self) -> dict[str, Any]:
        """Return all fields except embedding as a dict for Qdrant point payload."""
        data = self.model_dump()
        data.pop("embedding", None)
        return data

    def update_content(self, new_content: str) -> None:
        """Update the memory content and refresh the updated_at timestamp."""
        self.content = new_content
        self.updated_at = datetime.now(UTC)


class SearchResult(BaseModel):
    """A memory fact paired with its similarity score from a search query."""

    fact: MemoryFact
    score: float
