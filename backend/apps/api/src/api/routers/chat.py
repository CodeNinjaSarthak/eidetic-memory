"""Memory-augmented chat endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_llm_service, get_memory_manager, get_memory_retriever
from api.schemas import ChatRequest, ChatResponse, MemoryResponse
from llm.generation.base import AbstractLLMService
from memory.manager import MemoryManager
from memory.models.conversation import ConversationPair, Message
from retrieval.context import ContextBuilder
from retrieval.retriever import MemoryRetriever
from storage.models import MemoryFact

router = APIRouter(prefix="/chat", tags=["chat"])


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


@router.post("/", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    manager: Annotated[MemoryManager, Depends(get_memory_manager)],
    retriever: Annotated[MemoryRetriever, Depends(get_memory_retriever)],
    llm_service: Annotated[AbstractLLMService, Depends(get_llm_service)],
) -> ChatResponse:
    """Run a memory-augmented chat turn."""
    memories = await retriever.retrieve(
        query=payload.message, user_id=payload.user_id
    )

    system_prompt = ContextBuilder().build_system_prompt(
        base_prompt="You are a helpful assistant with memory of past conversations.",
        memories=memories,
    )

    reply = await llm_service.complete(
        messages=[{"role": "user", "content": payload.message}],
        system=system_prompt,
    )

    previous = Message(
        user_id=payload.user_id,
        session_id=payload.session_id,
        role="user",
        content=payload.message,
    )
    current = Message(
        user_id=payload.user_id,
        session_id=payload.session_id,
        role="assistant",
        content=reply,
    )
    pair = ConversationPair(current=current, previous=previous)

    facts = await manager.add_memory(
        pair=pair, user_id=payload.user_id, session_id=payload.session_id
    )

    return ChatResponse(reply=reply, facts_added=[_to_response(f) for f in facts])
