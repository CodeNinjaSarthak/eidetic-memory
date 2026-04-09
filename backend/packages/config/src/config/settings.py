"""Centralized application settings loaded from environment variables."""

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables.

    All services import their configuration from this single source of truth.
    Sensitive values use SecretStr to prevent accidental logging.
    """

    model_config = SettingsConfigDict(
        env_file=".env.development",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    # LLM Provider
    llm_provider: Literal["claude", "gemini", "azure", "groq"] = "claude"

    # Claude
    anthropic_api_key: SecretStr | None = None

    # Gemini
    google_api_key: SecretStr | None = None

    # Groq
    groq_api_key: SecretStr | None = None

    # Jina
    jina_api_key: SecretStr | None = None

    # Azure OpenAI
    azure_openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str | None = None

    # Qdrant
    qdrant_url: str
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = "eidetic_memories"

    # Embedding model
    embedding_model: str = "gemini-embedding-exp-03-07"
    embedding_dimension: int = Field(default=768, gt=0)

    # Memory pipeline
    memory_extraction_model: str = "gemini-2.0-flash"
    recency_window: int = Field(default=10, gt=0)
    similarity_top_k: int = Field(default=10, gt=0)

    # API
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, gt=0, lt=65536)
    api_env: Literal["development", "production", "test"] = "development"

    # Eval
    eval_llm_judge_model: str = "gemini-2.0-flash"

    @model_validator(mode="after")
    def _validate_provider_credentials(self) -> "Settings":
        """Ensure the selected LLM provider has required credentials."""
        if self.llm_provider == "claude" and not self.anthropic_api_key:
            raise ValueError(
                "anthropic_api_key is required when llm_provider is 'claude'"
            )

        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError("google_api_key is required when llm_provider is 'gemini'")

        if self.llm_provider == "azure":
            missing = [
                name
                for name, val in [
                    ("azure_openai_api_key", self.azure_openai_api_key),
                    ("azure_openai_endpoint", self.azure_openai_endpoint),
                    ("azure_openai_deployment", self.azure_openai_deployment),
                ]
                if not val
            ]
            if missing:
                raise ValueError(
                    f"azure provider requires all of: azure_openai_api_key, "
                    f"azure_openai_endpoint, azure_openai_deployment. "
                    f"Missing: {', '.join(missing)}"
                )

        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError("groq_api_key is required when llm_provider is 'groq'")

        return self
