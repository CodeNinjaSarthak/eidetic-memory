"""Request and response schemas for the API layer."""

from datetime import datetime

from pydantic import BaseModel


class MessageSchema(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str
    user_id: str
    session_id: str


class AddMemoryRequest(BaseModel):
    """Request body for adding a memory from a conversation pair."""

    user_id: str
    session_id: str
    current_message: MessageSchema
    previous_message: MessageSchema
    conversation_summary: str | None = None


class MemoryResponse(BaseModel):
    """A single memory fact returned to the client (no embedding)."""

    id: str
    user_id: str
    content: str
    importance_score: float | None = None
    created_at: datetime
    updated_at: datetime


class AddMemoryResponse(BaseModel):
    """Response body after adding memories."""

    added: list[MemoryResponse]


class SearchRequest(BaseModel):
    """Request body for searching memories."""

    query: str
    user_id: str
    top_k: int = 5


class SearchResponse(BaseModel):
    """Response body for a memory search."""

    memories: list[MemoryResponse]


class DeleteResponse(BaseModel):
    """Response body after deleting a memory."""

    deleted: bool
    memory_id: str
