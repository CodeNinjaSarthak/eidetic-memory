"""Dependency injection factories for FastAPI."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from config.settings import Settings
from llm.embeddings.openai import OpenAIEmbeddingService
from llm.generation.claude import ClaudeService
from memory.manager import MemoryManager
from retrieval.retriever import MemoryRetriever
from storage.qdrant import QdrantMemoryStore


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


def get_memory_manager(
    settings: Annotated[Settings, Depends(get_settings)],
) -> MemoryManager:
    """Build a MemoryManager wired to real services."""
    store = QdrantMemoryStore.from_settings(settings)
    embedding_service = OpenAIEmbeddingService(model=settings.embedding_model)
    llm_service = ClaudeService()

    return MemoryManager(
        store=store,
        embedding_service=embedding_service,
        llm_service=llm_service,
        similarity_top_k=settings.similarity_top_k,
    )


def get_memory_retriever(
    settings: Annotated[Settings, Depends(get_settings)],
) -> MemoryRetriever:
    """Build a MemoryRetriever wired to real services."""
    store = QdrantMemoryStore.from_settings(settings)
    embedding_service = OpenAIEmbeddingService(model=settings.embedding_model)

    return MemoryRetriever(
        store=store,
        embedding_service=embedding_service,
        top_k=settings.similarity_top_k,
    )
