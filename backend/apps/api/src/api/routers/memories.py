"""Memory CRUD and search endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_memory_manager, get_memory_retriever, get_settings
from api.schemas import (
    AddMemoryRequest,
    AddMemoryResponse,
    DeleteResponse,
    MemoryResponse,
    SearchRequest,
    SearchResponse,
)
from config.settings import Settings
from memory.manager import MemoryManager
from memory.models.conversation import ConversationPair, Message
from retrieval.retriever import MemoryRetriever
from storage.models import MemoryFact
from storage.qdrant import QdrantMemoryStore

router = APIRouter(prefix="/memories", tags=["memories"])


def _to_response(fact: MemoryFact) -> MemoryResponse:
    """Convert a storage MemoryFact to an API MemoryResponse."""
    return MemoryResponse(
        id=fact.id,
        user_id=fact.user_id,
        content=fact.content,
        importance_score=fact.importance_score,
        created_at=fact.created_at,
        updated_at=fact.updated_at,
    )


@router.post("/", response_model=AddMemoryResponse)
async def add_memory(
    payload: AddMemoryRequest,
    manager: Annotated[MemoryManager, Depends(get_memory_manager)],
) -> AddMemoryResponse:
    """Extract and store memories from a conversation pair."""
    current = Message(
        user_id=payload.current_message.user_id,
        session_id=payload.current_message.session_id,
        role=payload.current_message.role,
        content=payload.current_message.content,
    )
    previous = Message(
        user_id=payload.previous_message.user_id,
        session_id=payload.previous_message.session_id,
        role=payload.previous_message.role,
        content=payload.previous_message.content,
    )
    pair = ConversationPair(current=current, previous=previous)

    facts = await manager.add_memory(
        pair=pair,
        user_id=payload.user_id,
        session_id=payload.session_id,
        conversation_summary=payload.conversation_summary,
    )

    return AddMemoryResponse(added=[_to_response(f) for f in facts])


@router.post("/search", response_model=SearchResponse)
async def search_memories(
    payload: SearchRequest,
    retriever: Annotated[MemoryRetriever, Depends(get_memory_retriever)],
) -> SearchResponse:
    """Search memories by semantic similarity."""
    facts = await retriever.retrieve(
        query=payload.query,
        user_id=payload.user_id,
        top_k=payload.top_k,
    )

    return SearchResponse(memories=[_to_response(f) for f in facts])


@router.get("/{user_id}", response_model=list[MemoryResponse])
async def list_memories(
    user_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[MemoryResponse]:
    """List all memories for a user."""
    store = QdrantMemoryStore.from_settings(settings)
    facts = await store.list_all(user_id=user_id)
    return [_to_response(f) for f in facts]


@router.delete("/{memory_id}", response_model=DeleteResponse)
async def delete_memory(
    memory_id: str,
    user_id: Annotated[str, Query(...)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DeleteResponse:
    """Delete a memory by id."""
    store = QdrantMemoryStore.from_settings(settings)
    await store.delete(memory_id=memory_id, user_id=user_id)
    return DeleteResponse(deleted=True, memory_id=memory_id)
