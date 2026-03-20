"""Semantic memory retrieval with optional importance-based re-ranking."""

import logging

from llm.embeddings.base import AbstractEmbeddingService
from storage.base import AbstractMemoryStore
from storage.models import MemoryFact

logger = logging.getLogger(__name__)


def _has_importance_scores(facts: list[MemoryFact]) -> bool:
    """Return True if every fact has a non-None importance_score."""
    return all(fact.importance_score is not None for fact in facts)


class MemoryRetriever:
    """Embeds a query, searches the vector store, and optionally re-ranks by importance."""

    def __init__(
        self,
        store: AbstractMemoryStore,
        embedding_service: AbstractEmbeddingService,
        top_k: int = 5,
        rerank_by_importance: bool = True,
    ) -> None:
        self._store = store
        self._embedding_service = embedding_service
        self._top_k = top_k
        self._rerank_by_importance = rerank_by_importance

    async def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int | None = None,
    ) -> list[MemoryFact]:
        """Embed the query, search for similar facts, and optionally re-rank.

        Args:
            query: The natural-language query to search for.
            user_id: Scope the search to this user's memories.
            top_k: Override the instance default for number of results.

        Returns:
            A list of MemoryFact instances ordered by relevance.
        """
        effective_top_k = top_k if top_k is not None else self._top_k

        query_embedding = await self._embedding_service.embed(query)

        facts = await self._store.search(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=effective_top_k,
        )

        if self._rerank_by_importance and _has_importance_scores(facts):
            facts.sort(key=lambda f: f.importance_score, reverse=True)  # type: ignore[arg-type]

        return facts[:effective_top_k]
