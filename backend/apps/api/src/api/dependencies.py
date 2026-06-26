"""Dependency injection factories for FastAPI."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from config.settings import Settings
from llm.embeddings.azure import AzureEmbeddingService
from llm.embeddings.base import AbstractEmbeddingService
from llm.embeddings.openai import OpenAIEmbeddingService
from llm.generation.azure import AzureService
from llm.generation.base import AbstractLLMService
from llm.generation.claude import ClaudeService
from llm.generation.gemini import GeminiService
from llm.generation.groq import GroqService
from llm.generation.openai_service import OpenAIService
from memory.manager import MemoryManager
from retrieval.retriever import MemoryRetriever
from storage.qdrant import QdrantMemoryStore


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


def _build_embedding_service(settings: Settings) -> AbstractEmbeddingService:
    """Create the embedding service for the configured provider.

    Always uses text-embedding-3-small at 1536D — hardcoded to match the
    eidetic_memories collection. Never passes a dimensions= override.
    """
    match settings.embedding_provider:
        case "azure":
            return AzureEmbeddingService(
                api_key=settings.azure_openai_api_key.get_secret_value(),
                endpoint=settings.azure_openai_endpoint,
                deployment="text-embedding-3-small",
            )
        case "openai":
            return OpenAIEmbeddingService(
                api_key=settings.openai_api_key.get_secret_value(),
                model="text-embedding-3-small",
            )


def _build_llm_service(settings: Settings) -> AbstractLLMService:
    """Create the LLM service for the configured provider."""
    match settings.llm_provider:
        case "claude":
            return ClaudeService(
                api_key=settings.anthropic_api_key.get_secret_value(),
                model=settings.memory_extraction_model,
            )
        case "gemini":
            return GeminiService(
                api_key=settings.google_api_key.get_secret_value(),
                model=settings.memory_extraction_model,
            )
        case "azure":
            return AzureService(
                api_key=settings.azure_openai_api_key.get_secret_value(),
                endpoint=settings.azure_openai_endpoint,
                deployment=settings.azure_openai_deployment,
            )
        case "groq":
            return GroqService(
                api_key=settings.groq_api_key.get_secret_value(),
                model=settings.memory_extraction_model,
            )
        case "openai":
            return OpenAIService(
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.openai_model,
            )


_memory_manager: MemoryManager | None = None


def get_memory_manager(
    settings: Annotated[Settings, Depends(get_settings)],
) -> MemoryManager:
    """Return a cached MemoryManager wired to real services."""
    global _memory_manager
    if _memory_manager is None:
        store = QdrantMemoryStore.from_settings(settings)
        embedding_service = _build_embedding_service(settings)
        llm_service = _build_llm_service(settings)
        _memory_manager = MemoryManager(
            store=store,
            embedding_service=embedding_service,
            llm_service=llm_service,
            similarity_top_k=settings.similarity_top_k,
        )
    return _memory_manager


_memory_retriever: MemoryRetriever | None = None


def get_memory_retriever(
    settings: Annotated[Settings, Depends(get_settings)],
) -> MemoryRetriever:
    """Return a cached MemoryRetriever wired to real services."""
    global _memory_retriever
    if _memory_retriever is None:
        store = QdrantMemoryStore.from_settings(settings)
        embedding_service = _build_embedding_service(settings)
        _memory_retriever = MemoryRetriever(
            store=store,
            embedding_service=embedding_service,
            top_k=settings.similarity_top_k,
            reranker_fetch_multiplier=3,
        )
    return _memory_retriever


_llm_service: AbstractLLMService | None = None


def get_llm_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AbstractLLMService:
    """Return a cached LLM service for the configured provider."""
    global _llm_service
    if _llm_service is None:
        _llm_service = _build_llm_service(settings)
    return _llm_service


_demo_retriever: MemoryRetriever | None = None


def get_demo_retriever(
    settings: Annotated[Settings, Depends(get_settings)],
) -> MemoryRetriever:
    """Return a cached MemoryRetriever for the demo endpoint.

    Uses top_k=similarity_top_k (30), no Jina, fetch_multiplier=3 so
    the demo endpoint receives 90 candidates per speaker for local rerank.
    """
    global _demo_retriever
    if _demo_retriever is None:
        store = QdrantMemoryStore.from_settings(settings)
        embedding_service = _build_embedding_service(settings)
        _demo_retriever = MemoryRetriever(
            store=store,
            embedding_service=embedding_service,
            top_k=settings.similarity_top_k,
            jina_api_key=None,
            reranker_fetch_multiplier=3,
            rerank_by_importance=False,
        )
    return _demo_retriever
