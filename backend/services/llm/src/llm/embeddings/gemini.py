"""Gemini-backed embedding service implementation."""

import math
import os

from google import genai

from llm.embeddings.base import AbstractEmbeddingService, EmbeddingError


class GeminiEmbeddingService(AbstractEmbeddingService):
    """Embedding service backed by Google's Gemini embedding API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-exp-03-07",
    ) -> None:
        """Initialize the Gemini embedding service.

        Args:
            api_key: Google API key. If None, reads GOOGLE_API_KEY from environment.
            model: The Gemini embedding model to use.

        Raises:
            ValueError: If no API key is provided or found in the environment.
        """
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Google API key is required. Pass api_key= or set GOOGLE_API_KEY."
            )
        self._client = genai.Client(api_key=resolved_key)
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string into an L2-normalized vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the L2-normalized embedding vector.

        Raises:
            EmbeddingError: If the embedding request fails.
        """
        try:
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=text,
            )
            values = response.embeddings[0].values

            norm = math.sqrt(sum(v * v for v in values))
            if norm > 0:
                values = [v / norm for v in values]

            return values
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to embed text: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts by calling embed() for each.

        Args:
            texts: List of texts to embed.

        Returns:
            List of L2-normalized embedding vectors in the same order as input.

        Raises:
            EmbeddingError: If any embedding request fails.
        """
        try:
            return [await self.embed(text) for text in texts]
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to embed batch: {e}") from e
