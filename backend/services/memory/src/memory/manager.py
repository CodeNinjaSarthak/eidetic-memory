"""MemoryManager — public orchestrator for the memory service.

Wires together extraction, evolution, embedding, and storage
into a single entry point for adding and searching memories.
"""

import logging

from llm.embeddings.base import AbstractEmbeddingService
from llm.generation.base import AbstractLLMService
from memory.models.conversation import ConversationPair, Message
from memory.models.memory import MemoryFact, MemoryOperation
from memory.pipeline.extraction import ExtractionPipeline
from memory.pipeline.update import EvolutionEngine
from storage.base import AbstractMemoryStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """Orchestrates extraction, evolution, embedding, and storage of memories.

    This is the public API for the memory service. It takes a conversation pair,
    extracts candidate facts, decides how each one evolves the memory store,
    and persists the results.
    """

    def __init__(
        self,
        store: AbstractMemoryStore,
        embedding_service: AbstractEmbeddingService,
        llm_service: AbstractLLMService,
        similarity_top_k: int = 10,
    ) -> None:
        """Initialize the memory manager.

        Args:
            store: Memory store backend for persistence and search.
            embedding_service: Service for converting text to vectors.
            llm_service: LLM service for extraction and evolution.
            similarity_top_k: Number of similar memories to retrieve per candidate.
        """
        self._store = store
        self._embedding_service = embedding_service
        self._extraction = ExtractionPipeline(llm_service)
        self._evolution = EvolutionEngine(llm_service)
        self._similarity_top_k = similarity_top_k

    async def add_memory(
        self,
        pair: ConversationPair,
        user_id: str,
        session_id: str | None = None,
        conversation_summary: str | None = None,
        recent_messages: list[Message] | None = None,
    ) -> list[MemoryFact]:
        """Extract facts from a conversation pair and evolve the memory store.

        Args:
            pair: The conversation pair to process.
            user_id: The user who owns these memories.
            session_id: Optional session identifier.
            conversation_summary: Optional summary of the conversation so far.
            recent_messages: Optional recent messages for context.

        Returns:
            List of MemoryFact instances that were added or updated.
        """
        candidates = await self._extraction.extract(
            pair, conversation_summary, recent_messages
        )

        if not candidates:
            return []

        results: list[MemoryFact] = []

        for candidate in candidates:
            embedding = await self._embedding_service.embed(candidate)

            similar = await self._store.search(
                user_id=user_id,
                query_embedding=embedding,
                top_k=self._similarity_top_k,
            )

            update = await self._evolution.decide(candidate, similar)

            if update.operation == MemoryOperation.ADD:
                fact = MemoryFact(
                    user_id=user_id,
                    session_id=session_id,
                    content=candidate,
                    embedding=embedding,
                )
                await self._store.upsert(fact)
                results.append(fact)

            elif update.operation == MemoryOperation.UPDATE:
                existing = await self._store.get(update.memory_id, user_id)
                if existing is None:
                    logger.warning(
                        "UPDATE target %s not found, skipping", update.memory_id
                    )
                    continue
                existing.update_content(update.updated_content)
                existing.embedding = await self._embedding_service.embed(
                    update.updated_content
                )
                await self._store.upsert(existing)
                results.append(existing)

            elif update.operation == MemoryOperation.DELETE:
                await self._store.delete(update.memory_id, user_id)

            # NOOP: do nothing

        return results

    async def search_memory(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> list[MemoryFact]:
        """Search for memories relevant to a query.

        Args:
            query: The search query text.
            user_id: The user whose memories to search.
            top_k: Maximum number of results to return.

        Returns:
            List of matching MemoryFact instances, ordered by relevance.
        """
        embedding = await self._embedding_service.embed(query)
        return await self._store.search(
            user_id=user_id,
            query_embedding=embedding,
            top_k=top_k,
        )
