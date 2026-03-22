"""MemoryManager — public orchestrator for the memory service.

Wires together extraction, evolution, embedding, and storage
into a single entry point for adding and searching memories.
"""

import asyncio
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

    async def _embed_and_search(
        self,
        candidate: str,
        user_id: str,
    ) -> tuple[str, list[float], list[MemoryFact]]:
        """Embed a candidate fact and search for similar existing memories.

        Args:
            candidate: The candidate fact text.
            user_id: The user who owns these memories.

        Returns:
            Tuple of (candidate text, embedding vector, similar memories).
        """
        embedding = await self._embedding_service.embed(candidate)
        similar = await self._store.search(
            user_id=user_id,
            query_embedding=embedding,
            top_k=self._similarity_top_k,
        )
        return candidate, embedding, similar

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

        # Phase 1: Embed + search in parallel for all candidates
        search_results = await asyncio.gather(
            *[self._embed_and_search(c, user_id) for c in candidates]
        )

        # Phase 2: Build embeddings dict and deduplicate existing memories
        embeddings: dict[str, list[float]] = {}
        all_existing: dict[str, MemoryFact] = {}
        for candidate, embedding, similar in search_results:
            embeddings[candidate] = embedding
            for mem in similar:
                all_existing[mem.id] = mem

        # Phase 3: Map UUIDs to sequential integers
        uuid_to_int: dict[str, str] = {
            mem_id: str(idx) for idx, mem_id in enumerate(all_existing)
        }
        int_to_uuid: dict[str, str] = {v: k for k, v in uuid_to_int.items()}

        mapped_memories = [
            {"id": uuid_to_int[mem_id], "text": mem.content}
            for mem_id, mem in all_existing.items()
        ]

        # Phase 4: ONE batch evolution call
        candidate_texts = [c for c, _, _ in search_results]
        batch_updates = await self._evolution.decide_batch(
            candidate_texts, mapped_memories
        )

        # Phase 5: Reverse-map integer IDs back to UUIDs
        for update in batch_updates:
            if update.memory_id and update.memory_id in int_to_uuid:
                update.memory_id = int_to_uuid[update.memory_id]

        # Phase 6: Execute operations
        results: list[MemoryFact] = []

        for i, update in enumerate(batch_updates):
            candidate = candidate_texts[i]
            embedding = embeddings[candidate]

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
