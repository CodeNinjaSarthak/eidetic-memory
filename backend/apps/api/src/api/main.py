"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_settings
from api.routers.chat import router as chat_router
from api.routers.memories import router as memories_router
from storage.qdrant import QdrantMemoryStore

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize resources on startup."""
    settings = get_settings()
    store = QdrantMemoryStore.from_settings(settings)
    await store._ensure_collection()
    yield


app = FastAPI(title="Eidetic Memory API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(memories_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Return a simple health check response."""
    return {"status": "ok"}
