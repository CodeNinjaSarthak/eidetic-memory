"""Dependency injection factories for FastAPI."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from config.settings import Settings
from llm.embeddings.gemini import GeminiEmbeddingService
from llm.generation.azure import AzureService
from llm.generation.base import AbstractLLMService
from llm.generation.claude import ClaudeService
from llm.generation.gemini import GeminiService
from llm.generation.groq import GroqService
from memory.manager import MemoryManager
from retrieval.retriever import MemoryRetriever
from storage.qdrant import QdrantMemoryStore


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


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


def get_memory_manager(
    settings: Annotated[Settings, Depends(get_settings)],
) -> MemoryManager:
    """Build a MemoryManager wired to real services."""
    store = QdrantMemoryStore.from_settings(settings)
    embedding_service = GeminiEmbeddingService(
        api_key=settings.google_api_key.get_secret_value()
        if settings.google_api_key
        else None,
        model=settings.embedding_model,
    )
    llm_service = _build_llm_service(settings)

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
    embedding_service = GeminiEmbeddingService(
        api_key=settings.google_api_key.get_secret_value()
        if settings.google_api_key
        else None,
        model=settings.embedding_model,
    )

    return MemoryRetriever(
        store=store,
        embedding_service=embedding_service,
        top_k=settings.similarity_top_k,
    )
