"""Fact extraction pipeline — extracts candidate memory facts from conversation pairs."""

import logging
from pathlib import Path
from typing import Any

from llm.generation.base import AbstractLLMService
from memory.models.conversation import ConversationPair, Message

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_EXTRACT_FACTS_TOOL: dict[str, Any] = {
    "name": "extract_facts",
    "description": "Extract a list of factual statements from the conversation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of factual statements extracted from the conversation.",
            }
        },
        "required": ["facts"],
    },
}


def _build_user_content(
    pair: ConversationPair,
    conversation_summary: str | None = None,
    recent_messages: list[Message] | None = None,
) -> str:
    """Assemble the user message content for the extraction LLM call.

    Args:
        pair: The current conversation pair to extract facts from.
        conversation_summary: Optional summary of the conversation so far.
        recent_messages: Optional list of recent messages for context.

    Returns:
        Formatted string combining all context for the LLM.
    """
    sections: list[str] = []

    if conversation_summary:
        sections.append(f"Conversation Summary:\n{conversation_summary}")

    if recent_messages:
        formatted = "\n".join(f"{msg.role}: {msg.content}" for msg in recent_messages)
        sections.append(f"Recent Messages:\n{formatted}")

    sections.append(
        f"Current Conversation:\n"
        f"{pair.previous.role}: {pair.previous.content}\n"
        f"{pair.current.role}: {pair.current.content}"
    )

    return "\n\n".join(sections)


class ExtractionPipeline:
    """Extracts candidate memory facts from a conversation pair using an LLM."""

    def __init__(self, llm_service: AbstractLLMService) -> None:
        """Initialize the extraction pipeline.

        Args:
            llm_service: The LLM service to use for fact extraction.
        """
        self._llm = llm_service
        self._system_prompt = (_PROMPTS_DIR / "extraction.txt").read_text()

    async def extract(
        self,
        pair: ConversationPair,
        conversation_summary: str | None = None,
        recent_messages: list[Message] | None = None,
    ) -> list[str]:
        """Extract candidate facts from a conversation pair.

        Args:
            pair: The conversation pair to extract facts from.
            conversation_summary: Optional summary of the conversation so far.
            recent_messages: Optional list of recent messages for context.

        Returns:
            A list of extracted fact strings.
        """
        user_content = _build_user_content(pair, conversation_summary, recent_messages)

        result = await self._llm.complete_with_tool(
            messages=[{"role": "user", "content": user_content}],
            tool=_EXTRACT_FACTS_TOOL,
            system=self._system_prompt,
        )

        facts = result.get("facts", [])
        if not isinstance(facts, list):
            logger.warning("LLM returned non-list facts value: %s", type(facts))
            return []

        return [str(f) for f in facts]
