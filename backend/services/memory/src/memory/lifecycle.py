"""Memory lifecycle management — importance scoring and pruning."""

import logging
import math
from datetime import UTC, datetime

from storage.base import AbstractMemoryStore
from storage.models import MemoryFact

logger = logging.getLogger(__name__)


class LifecycleManager:
    """Scores memories by recency and frequency, and prunes low-value ones.

    Importance is computed as a weighted combination of:
    - Recency: exponential decay based on days since last update
    - Frequency: log-scaled access count, capped at a configurable ceiling
    """

    def __init__(
        self,
        store: AbstractMemoryStore,
        recency_weight: float = 0.7,
        decay_days: float = 30.0,
        frequency_cap: int = 10,
    ) -> None:
        self._store = store
        self._recency_weight = recency_weight
        self._frequency_weight = 1.0 - recency_weight
        self._decay_days = decay_days
        self._frequency_cap = frequency_cap

    def _compute_importance(self, fact: MemoryFact) -> float:
        """Compute an importance score in [0, 1] for a single memory fact."""
        now = datetime.now(UTC)
        age_days = (now - fact.updated_at).total_seconds() / 86400.0
        recency_score = math.exp(-age_days / self._decay_days)

        access_count = fact.metadata.get("access_count", 0)
        capped = min(access_count, self._frequency_cap)
        frequency_score = math.log1p(capped) / math.log1p(self._frequency_cap)

        return (
            self._recency_weight * recency_score
            + self._frequency_weight * frequency_score
        )

    async def score_memories(self, user_id: str) -> list[MemoryFact]:
        """Score all memories for a user and persist the updated importance scores."""
        facts = await self._store.list_all(user_id)
        for fact in facts:
            fact.importance_score = self._compute_importance(fact)
            await self._store.upsert(fact)
        return facts

    async def prune_memories(
        self,
        user_id: str,
        min_importance: float = 0.2,
        max_memories: int | None = None,
    ) -> list[str]:
        """Delete low-scoring memories and return the IDs of deleted facts.

        Memories are first scored, then any below min_importance are pruned.
        If max_memories is set, the lowest-scoring facts beyond that limit
        are also removed.
        """
        facts = await self.score_memories(user_id)
        facts.sort(key=lambda f: f.importance_score or 0.0, reverse=True)

        deleted_ids: list[str] = []

        for fact in facts:
            if (fact.importance_score or 0.0) < min_importance:
                await self._store.delete(fact.id, user_id)
                deleted_ids.append(fact.id)

        if max_memories is not None:
            remaining = [f for f in facts if f.id not in set(deleted_ids)]
            for fact in remaining[max_memories:]:
                await self._store.delete(fact.id, user_id)
                deleted_ids.append(fact.id)

        logger.info(
            "Pruned %d memories for user %s", len(deleted_ids), user_id
        )
        return deleted_ids

    async def record_access(self, memory_id: str, user_id: str) -> MemoryFact | None:
        """Increment the access count for a memory on retrieval."""
        fact = await self._store.get(memory_id, user_id)
        if fact is None:
            logger.warning(
                "Cannot record access: memory %s not found for user %s",
                memory_id,
                user_id,
            )
            return None
        fact.increment_access_count()
        await self._store.upsert(fact)
        return fact
