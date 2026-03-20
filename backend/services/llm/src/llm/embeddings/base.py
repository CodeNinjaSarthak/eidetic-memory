"""Abstract base class for embedding services."""

from abc import ABC, abstractmethod


class EmbeddingError(Exception):
    """Raised when an embedding request fails."""


class AbstractEmbeddingService(ABC):
    """Interface for converting text into vector embeddings."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingError: If the embedding request fails.
        """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in the same order as input.

        Raises:
            EmbeddingError: If the embedding request fails.
        """
