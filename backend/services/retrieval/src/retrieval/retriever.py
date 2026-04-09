"""Semantic memory retrieval with optional importance-based re-ranking."""

import asyncio
import logging

import httpx

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
        jina_api_key: str | None = None,
        reranker_fetch_multiplier: int = 3,
    ) -> None:
        self._store = store
        self._embedding_service = embedding_service
        self._top_k = top_k
        self._rerank_by_importance = rerank_by_importance
        self._jina_api_key = jina_api_key
        self._reranker_fetch_multiplier = reranker_fetch_multiplier

    async def _jina_rerank(
        self,
        query: str,
        facts: list[MemoryFact],
        top_k: int,
    ) -> list[MemoryFact]:
        """Rerank facts using Jina Reranker API, return top_k."""
        if not facts:
            return facts

        documents = [f.content for f in facts]
        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://api.jina.ai/v1/rerank",
                        headers={
                            "Authorization": f"Bearer {self._jina_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "jina-reranker-v2-base-multilingual",
                            "query": query,
                            "documents": documents,
                            "top_n": top_k,
                        },
                    )
                    if response.status_code == 429:
                        wait = base_delay * (2 ** attempt)
                        logger.warning(
                            "Jina rate limit hit, retrying in %.1fs (attempt %d/%d)",
                            wait, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(wait)
                        continue
                    response.raise_for_status()
                    data = response.json()
                    reranked = []
                    for result in data["results"]:
                        reranked.append(facts[result["index"]])
                    return reranked
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    logger.warning("Jina quota exhausted (403), falling back to vector order permanently")
                    self._jina_api_key = None
                    return facts[:top_k]
                if attempt == max_retries - 1:
                    logger.error("Jina reranker failed after %d attempts, falling back to vector order", max_retries)
                    return facts[:top_k]
                raise

        logger.error("Jina reranker exhausted retries, falling back to vector order")
        return facts[:top_k]

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
        fetch_k = (
            effective_top_k * self._reranker_fetch_multiplier
            if self._jina_api_key
            else effective_top_k
        )

        query_embedding = await self._embedding_service.embed(query)

        facts = await self._store.search(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=fetch_k,
        )

        if self._rerank_by_importance and _has_importance_scores(facts):
            facts.sort(key=lambda f: f.importance_score, reverse=True)  # type: ignore[arg-type]

        if self._jina_api_key:
            facts = await self._jina_rerank(query, facts, effective_top_k)

        return facts[:effective_top_k]
