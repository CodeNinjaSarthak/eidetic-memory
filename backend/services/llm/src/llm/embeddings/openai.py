"""OpenAI-backed embedding service implementation."""

import os

from openai import AsyncOpenAI

from llm.embeddings.base import AbstractEmbeddingService, EmbeddingError


class OpenAIEmbeddingService(AbstractEmbeddingService):
    """Embedding service backed by OpenAI's embedding API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        """Initialize the OpenAI embedding service.

        Args:
            model: The OpenAI embedding model to use.
            api_key: OpenAI API key. If None, reads OPENAI_API_KEY from environment.

        Raises:
            ValueError: If no API key is provided or found in the environment.
        """
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key is required. Pass api_key= or set OPENAI_API_KEY."
            )
        self._client = AsyncOpenAI(api_key=resolved_key)
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingError: If the embedding request fails.
        """
        try:
            response = await self._client.embeddings.create(
                input=text,
                model=self._model,
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"Failed to embed text: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in the same order as input.

        Raises:
            EmbeddingError: If the embedding request fails.
        """
        try:
            response = await self._client.embeddings.create(
                input=texts,
                model=self._model,
            )
            return [
                item.embedding for item in sorted(response.data, key=lambda x: x.index)
            ]
        except Exception as e:
            raise EmbeddingError(f"Failed to embed batch: {e}") from e
