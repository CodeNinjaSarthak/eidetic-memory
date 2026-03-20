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
