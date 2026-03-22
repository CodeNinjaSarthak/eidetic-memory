"""Memory evolution engine — decides how candidate facts update the memory store."""

import logging
from pathlib import Path
from typing import Any

from llm.generation.base import AbstractLLMService
from memory.models.memory import MemoryFact, MemoryOperation, MemoryUpdate

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_DECIDE_OPERATION_TOOL: dict[str, Any] = {
    "name": "decide_operation",
    "description": "Decide what operation to perform for the candidate fact given existing memories.",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["ADD", "UPDATE", "DELETE", "NOOP"],
                "description": "The operation to perform.",
            },
            "memory_id": {
                "type": "string",
                "description": "The ID of the existing memory to update or delete. Required for UPDATE and DELETE.",
            },
            "updated_content": {
                "type": "string",
                "description": "The new content for the memory. Required for UPDATE only. Omit for ADD, DELETE, and NOOP.",
            },
        },
        "required": ["operation"],
    },
}


def _build_user_content(
    candidate_fact: str,
    existing_memories: list[MemoryFact],
) -> str:
    """Assemble the user message content for the evolution LLM call.

    Args:
        candidate_fact: The new candidate fact to evaluate.
        existing_memories: Existing memories semantically similar to the candidate.

    Returns:
        Formatted string combining the candidate fact and existing memories for the LLM.
    """
    sections: list[str] = []

    sections.append(f"Candidate Fact:\n{candidate_fact}")

    if existing_memories:
        formatted = "\n".join(
            f"- [ID: {mem.id}] {mem.content}" for mem in existing_memories
        )
        sections.append(f"Existing Memories:\n{formatted}")
    else:
        sections.append("Existing Memories:\nNone")

    return "\n\n".join(sections)


def _parse_result(result: dict[str, Any]) -> MemoryUpdate:
    """Convert the raw LLM tool response into a MemoryUpdate.

    Falls back to NOOP if the response is malformed or contains an invalid operation.

    Args:
        result: Raw dict returned by the LLM tool call.

    Returns:
        A validated MemoryUpdate instance.
    """
    raw_operation = result.get("operation")
    if not isinstance(raw_operation, str):
        logger.warning("LLM returned non-string operation: %s", type(raw_operation))
        return MemoryUpdate(operation=MemoryOperation.NOOP)

    try:
        operation = MemoryOperation(raw_operation)
    except ValueError:
        logger.warning("LLM returned invalid operation: %s", raw_operation)
        return MemoryUpdate(operation=MemoryOperation.NOOP)

    memory_id = result.get("memory_id")
    updated_content = result.get("updated_content")

    if operation == MemoryOperation.UPDATE and not updated_content:
        logger.warning(
            "LLM returned UPDATE without updated_content, falling back to NOOP"
        )
        return MemoryUpdate(operation=MemoryOperation.NOOP)

    if operation in (MemoryOperation.UPDATE, MemoryOperation.DELETE) and not memory_id:
        logger.warning(
            "LLM returned %s without memory_id, falling back to NOOP", operation
        )
        return MemoryUpdate(operation=MemoryOperation.NOOP)

    return MemoryUpdate(
        operation=operation,
        memory_id=memory_id,
        updated_content=updated_content,
    )


_BATCH_DECIDE_TOOL: dict[str, Any] = {
    "name": "batch_decide_operations",
    "description": (
        "Decide operations for multiple candidate facts given existing memories."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "memory": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": (
                                "For UPDATE/DELETE: the integer ID of the existing "
                                "memory to act on. For ADD: leave as empty string."
                            ),
                        },
                        "text": {
                            "type": "string",
                            "description": (
                                "The memory content. For ADD: the new fact. "
                                "For UPDATE: the merged content."
                            ),
                        },
                        "event": {
                            "type": "string",
                            "enum": ["ADD", "UPDATE", "DELETE", "NONE"],
                            "description": "The operation to perform.",
                        },
                        "old_memory": {
                            "type": "string",
                            "description": (
                                "For UPDATE/DELETE: the previous text of the memory "
                                "being changed. For logging only."
                            ),
                        },
                    },
                    "required": ["event"],
                },
            },
        },
        "required": ["memory"],
    },
}


def _build_batch_user_content(
    candidates: list[str],
    existing_memories: list[dict[str, str]],
) -> str:
    """Assemble the user message content for a batched evolution LLM call.

    Args:
        candidates: The new candidate facts to evaluate.
        existing_memories: Existing memories with integer-mapped IDs.
            Each dict has "id" and "text" keys.

    Returns:
        Formatted string combining all candidates and existing memories for the LLM.
    """
    sections: list[str] = []

    numbered = "\n".join(
        f"{i + 1}. {candidate}" for i, candidate in enumerate(candidates)
    )
    sections.append(f"Candidate Facts:\n{numbered}")

    if existing_memories:
        formatted = "\n".join(
            f"- [ID: {mem['id']}] {mem['text']}" for mem in existing_memories
        )
        sections.append(f"Existing Memories:\n{formatted}")
    else:
        sections.append("Existing Memories:\nNone")

    return "\n\n".join(sections)


def _parse_batch_result(
    result: dict[str, Any],
    expected_count: int,
) -> list[MemoryUpdate]:
    """Convert the raw LLM batch tool response into a list of MemoryUpdate objects.

    Falls back to NOOP for individual items that are malformed. Pads with NOOP
    if fewer items than expected, truncates if more.

    Args:
        result: Raw dict returned by the LLM tool call.
        expected_count: The number of candidate facts that were sent.

    Returns:
        A list of validated MemoryUpdate instances, one per candidate.
    """
    noop = MemoryUpdate(operation=MemoryOperation.NOOP)
    raw_memory = result.get("memory")
    if not isinstance(raw_memory, list):
        logger.warning("Batch LLM response missing 'memory' array")
        return [noop] * expected_count

    updates: list[MemoryUpdate] = []
    for item in raw_memory[:expected_count]:
        raw_event = item.get("event") if isinstance(item, dict) else None
        if not isinstance(raw_event, str):
            logger.warning("Batch item missing 'event' field")
            updates.append(noop)
            continue

        # Map "NONE" from batch schema to internal NOOP
        if raw_event == "NONE":
            raw_event = "NOOP"

        try:
            operation = MemoryOperation(raw_event)
        except ValueError:
            logger.warning("Batch item has invalid event: %s", raw_event)
            updates.append(noop)
            continue

        memory_id = item.get("id") or None
        updated_content = item.get("text") or None

        if operation == MemoryOperation.UPDATE and not updated_content:
            logger.warning(
                "Batch UPDATE without text, falling back to NOOP"
            )
            updates.append(noop)
            continue

        if (
            operation in (MemoryOperation.UPDATE, MemoryOperation.DELETE)
            and not memory_id
        ):
            logger.warning(
                "Batch %s without id, falling back to NOOP", operation
            )
            updates.append(noop)
            continue

        updates.append(
            MemoryUpdate(
                operation=operation,
                memory_id=memory_id,
                updated_content=updated_content,
            )
        )

    # Pad with NOOP if fewer items than expected
    while len(updates) < expected_count:
        updates.append(noop)

    return updates


class EvolutionEngine:
    """Decides how a candidate fact should evolve the memory store.

    Takes a candidate fact string and a list of pre-retrieved similar memories,
    then uses an LLM tool call to decide whether to ADD, UPDATE, DELETE, or NOOP.
    """

    def __init__(self, llm_service: AbstractLLMService) -> None:
        """Initialize the evolution engine.

        Args:
            llm_service: The LLM service to use for deciding memory operations.
        """
        self._llm = llm_service
        self._system_prompt = (_PROMPTS_DIR / "update.txt").read_text()

    async def decide(
        self,
        candidate_fact: str,
        existing_memories: list[MemoryFact],
    ) -> MemoryUpdate:
        """Decide what operation to perform for a candidate fact.

        Args:
            candidate_fact: The new candidate fact to evaluate.
            existing_memories: Existing memories semantically similar to the candidate.

        Returns:
            A MemoryUpdate describing the operation to perform.
        """
        user_content = _build_user_content(candidate_fact, existing_memories)

        result = await self._llm.complete_with_tool(
            messages=[{"role": "user", "content": user_content}],
            tool=_DECIDE_OPERATION_TOOL,
            system=self._system_prompt,
        )

        return _parse_result(result)

    async def decide_batch(
        self,
        candidates: list[str],
        existing_memories: list[dict[str, str]],
    ) -> list[MemoryUpdate]:
        """Decide operations for multiple candidate facts in a single LLM call.

        Args:
            candidates: The new candidate facts to evaluate.
            existing_memories: Existing memories with integer-mapped IDs.
                Each dict has "id" and "text" keys.

        Returns:
            A list of MemoryUpdate objects, one per candidate.
        """
        user_content = _build_batch_user_content(candidates, existing_memories)

        result = await self._llm.complete_with_tool(
            messages=[{"role": "user", "content": user_content}],
            tool=_BATCH_DECIDE_TOOL,
            system=self._system_prompt,
        )

        return _parse_batch_result(result, len(candidates))
