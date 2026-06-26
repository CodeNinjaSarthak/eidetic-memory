"""Request and response schemas for the API layer."""

from datetime import datetime
from typing import Literal

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


class ChatRequest(BaseModel):
    """Request body for a memory-augmented chat turn."""

    user_id: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Response body for a chat turn."""

    reply: str
    facts_added: list[MemoryResponse]


class RankedMemory(BaseModel):
    """A memory fact with its rank before and after cross-encoder reranking."""

    content: str
    user_id: str
    rank_before: int
    rank_after: int | None = None


class DemoQueryRequest(BaseModel):
    """Request body for the read-only LoCoMo demo query endpoint."""

    conversation_id: str
    question: str
    question_type: Literal["factual", "open_domain"] = "factual"


class DemoQueryResponse(BaseModel):
    """Response body for a demo query, including before/after rerank views."""

    answer: str
    memories_top5_cosine: list[RankedMemory]
    memories_top5_reranked: list[RankedMemory]
    two_pass_would_fire: bool
    conversation_id: str
    speaker_a: str
    speaker_b: str
    question_type: str
