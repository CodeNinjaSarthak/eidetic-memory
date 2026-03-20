"""Context builder for formatting retrieved memories into LLM prompts."""

import logging

from storage.models import MemoryFact

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Formats retrieved MemoryFacts into strings suitable for LLM prompt injection."""

    def __init__(
        self,
        header: str = "## Relevant Memories",
        max_memories: int | None = None,
        show_importance: bool = False,
    ) -> None:
        self._header = header
        self._max_memories = max_memories
        self._show_importance = show_importance

    def build(self, memories: list[MemoryFact]) -> str:
        """Format memories into a string with header and bullet lines.

        Returns empty string when no memories are provided.
        Truncates to max_memories if configured, preserving input order.
        """
        if not memories:
            return ""

        truncated = memories[: self._max_memories] if self._max_memories is not None else memories

        if len(truncated) < len(memories):
            logger.debug(
                "Truncated memories from %d to %d", len(memories), len(truncated)
            )

        lines = [self._header]
        for fact in truncated:
            line = f"- {fact.content}"
            if self._show_importance and fact.importance_score is not None:
                line += f" (importance: {fact.importance_score:.2f})"
            lines.append(line)

        return "\n".join(lines)

    def build_system_prompt(
        self, base_prompt: str, memories: list[MemoryFact]
    ) -> str:
        """Append formatted memory context to a base system prompt.

        Returns base_prompt unchanged when no memories are provided.
        Separates base prompt and memory context with a blank line.
        """
        context = self.build(memories)
        if not context:
            return base_prompt

        return f"{base_prompt}\n\n{context}"
