"""FastAPI application entry point."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_settings
from api.routers.chat import router as chat_router
from api.routers.demo import router as demo_router
from api.routers.memories import router as memories_router
from retrieval import reranker
from storage.qdrant import QdrantMemoryStore

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize resources on startup."""
    settings = get_settings()
    store = QdrantMemoryStore.from_settings(settings)
    # _ensure_collection creates eidetic_memories if absent; it already exists at 1536D
    # so this is a no-op that confirms the collection is reachable.
    await store._ensure_collection()
    reranker.load_reranker()
    yield


app = FastAPI(title="Eidetic Memory API", version="0.1.0", lifespan=lifespan)

_cors_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(memories_router)
app.include_router(demo_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Return a simple health check response."""
    return {"status": "ok"}
