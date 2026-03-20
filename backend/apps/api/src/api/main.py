"""FastAPI application entry point."""

from fastapi import FastAPI

from api.routers.memories import router as memories_router

app = FastAPI(title="Eidetic Memory API", version="0.1.0")
app.include_router(memories_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Return a simple health check response."""
    return {"status": "ok"}
