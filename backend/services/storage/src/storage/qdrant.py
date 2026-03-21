"""Qdrant implementation of the memory store."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from storage.base import AbstractMemoryStore
from storage.models import MemoryFact

logger = logging.getLogger(__name__)


class QdrantMemoryStore(AbstractMemoryStore):
    """Memory store backed by a Qdrant vector database."""

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        collection_name: str = "memories",
        embedding_dimension: int = 1536,
    ) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key)
        self._collection_name = collection_name
        self._embedding_dimension = embedding_dimension

    @classmethod
    def from_settings(cls, settings: Settings) -> QdrantMemoryStore:
        """Construct a QdrantMemoryStore from application settings."""
        return cls(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None,
            collection_name=settings.qdrant_collection_name,
            embedding_dimension=settings.embedding_dimension,
        )

    async def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        collections = await self._client.get_collections()
        existing = {c.name for c in collections.collections}
        if self._collection_name not in existing:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s'", self._collection_name)
        await self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="user_id",
            field_schema="keyword",
        )

    async def upsert(self, fact: MemoryFact) -> None:
        """Insert or update a memory fact in Qdrant."""
        if fact.embedding is None:
            raise ValueError("Cannot upsert a MemoryFact without an embedding.")

        await self._ensure_collection()
        point = PointStruct(
            id=fact.id,
            vector=fact.embedding,
            payload=fact.to_qdrant_payload(),
        )
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[point],
        )

    async def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[MemoryFact]:
        """Search for similar facts belonging to a user."""
        await self._ensure_collection()
        results = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k,
        )
        return [MemoryFact(**point.payload) for point in results.points]

    async def get(self, memory_id: str, user_id: str) -> MemoryFact | None:
        """Retrieve a single fact by id if it belongs to the given user."""
        await self._ensure_collection()
        points = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=[memory_id],
        )
        if not points:
            return None
        payload = points[0].payload
        if payload.get("user_id") != user_id:
            return None
        return MemoryFact(**payload)

    async def delete(self, memory_id: str, user_id: str) -> None:
        """Delete a fact by id. No-op if the fact does not exist."""
        await self._ensure_collection()
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=PointIdsList(points=[memory_id]),
        )

    async def list_all(self, user_id: str) -> list[MemoryFact]:
        """Return all facts for a user, ordered by created_at descending."""
        await self._ensure_collection()
        facts: list[MemoryFact] = []
        offset = None
        while True:
            results, next_offset = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                ),
                offset=offset,
                limit=100,
            )
            facts.extend(MemoryFact(**point.payload) for point in results)
            if next_offset is None:
                break
            offset = next_offset
        facts.sort(key=lambda f: f.created_at, reverse=True)
        return facts
