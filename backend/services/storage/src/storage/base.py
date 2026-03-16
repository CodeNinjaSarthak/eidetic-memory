"""Abstract base class defining the memory store contract."""

from abc import ABC, abstractmethod

from storage.models import MemoryFact


class AbstractMemoryStore(ABC):
    """Interface for all memory storage backends.

    Implementations must provide async CRUD and search operations
    for MemoryFact instances, scoped by user_id.
    """

    @abstractmethod
    async def upsert(self, fact: MemoryFact) -> None:
        """Insert or update a memory fact.

        If a fact with the same id already exists, it is overwritten.
        """

    @abstractmethod
    async def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[MemoryFact]:
        """Return the top-k most similar facts for a user by embedding similarity."""

    @abstractmethod
    async def get(self, memory_id: str, user_id: str) -> MemoryFact | None:
        """Retrieve a single fact by id, or None if not found or user_id doesn't match."""

    @abstractmethod
    async def delete(self, memory_id: str, user_id: str) -> None:
        """Delete a fact by id. No-op if the fact does not exist."""

    @abstractmethod
    async def list_all(self, user_id: str) -> list[MemoryFact]:
        """Return all facts for a user, ordered by created_at descending."""
