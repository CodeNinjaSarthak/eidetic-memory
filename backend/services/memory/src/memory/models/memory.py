"""Memory operation models for the extraction/update pipeline."""

from enum import StrEnum

from pydantic import BaseModel

from storage.models import MemoryFact

__all__ = ["MemoryFact", "MemoryOperation", "MemoryUpdate"]


class MemoryOperation(StrEnum):
    """Operations that can be performed on a memory fact."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class MemoryUpdate(BaseModel):
    """An update operation produced by the update phase of the pipeline."""

    operation: MemoryOperation
    memory_id: str | None = None
    updated_content: str | None = None
