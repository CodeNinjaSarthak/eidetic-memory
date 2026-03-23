"""Azure OpenAI-backed embedding service implementation."""

import math
import os

from openai import AsyncAzureOpenAI

from llm.embeddings.base import AbstractEmbeddingService, EmbeddingError


class AzureEmbeddingService(AbstractEmbeddingService):
    """Embedding service backed by Azure OpenAI's embedding API."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment: str = "text-embedding-3-small",
    ) -> None:
        """Initialize the Azure OpenAI embedding service.

        Args:
            api_key: Azure OpenAI API key. If None, reads AZURE_OPENAI_API_KEY from environment.
            endpoint: Azure OpenAI endpoint. If None, reads AZURE_OPENAI_ENDPOINT from environment.
            deployment: The Azure deployment name for the embedding model.

        Raises:
            ValueError: If no API key or endpoint is provided or found in the environment.
        """
        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Azure OpenAI API key is required. Pass api_key= or set AZURE_OPENAI_API_KEY."
            )
        resolved_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not resolved_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint is required. Pass endpoint= or set AZURE_OPENAI_ENDPOINT."
            )
        self._client = AsyncAzureOpenAI(
            api_key=resolved_key,
            azure_endpoint=resolved_endpoint,
            api_version="2024-02-01",
        )
        self._deployment = deployment

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
            response = await self._client.embeddings.create(
                model=self._deployment,
                input=text,
            )
            values = list(response.data[0].embedding)

            norm = math.sqrt(sum(v * v for v in values))
            if norm > 0:
                values = [v / norm for v in values]

            return values
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to embed text: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of L2-normalized embedding vectors in the same order as input.

        Raises:
            EmbeddingError: If any embedding request fails.
        """
        try:
            response = await self._client.embeddings.create(
                model=self._deployment,
                input=texts,
            )
            results: list[list[float]] = []
            for item in response.data:
                values = list(item.embedding)
                norm = math.sqrt(sum(v * v for v in values))
                if norm > 0:
                    values = [v / norm for v in values]
                results.append(values)
            return results
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to embed batch: {e}") from e
