"""Tests for the Settings configuration class."""

import pytest
from pydantic import ValidationError

from config.settings import Settings

# -- Shared kwargs to avoid reading the developer's local .env.development --
_MINIMUM_VALID = {
    "_env_file": None,
    "qdrant_url": "http://localhost:6333",
    "llm_provider": "claude",
    "anthropic_api_key": "sk-ant-test-key",
}


def test_settings_requires_qdrant_url():
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert "qdrant_url" in str(exc_info.value)


def test_settings_accepts_minimum_valid_configuration():
    settings = Settings(**_MINIMUM_VALID)

    assert settings.qdrant_url == "http://localhost:6333"


def test_settings_rejects_claude_provider_without_api_key():
    with pytest.raises(ValidationError, match="anthropic_api_key"):
        Settings(_env_file=None, qdrant_url="http://localhost:6333", llm_provider="claude")


def test_settings_rejects_gemini_provider_without_api_key():
    with pytest.raises(ValidationError, match="google_api_key"):
        Settings(_env_file=None, qdrant_url="http://localhost:6333", llm_provider="gemini")


def test_settings_rejects_azure_provider_with_partial_credentials():
    with pytest.raises(ValidationError, match="azure_openai_endpoint"):
        Settings(
            _env_file=None,
            qdrant_url="http://localhost:6333",
            llm_provider="azure",
            azure_openai_api_key="key",
        )


def test_settings_rejects_invalid_llm_provider():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qdrant_url="http://localhost:6333", llm_provider="openai")


def test_settings_rejects_non_positive_embedding_dimension():
    with pytest.raises(ValidationError):
        Settings(**_MINIMUM_VALID, embedding_dimension=0)


def test_settings_rejects_out_of_range_api_port():
    with pytest.raises(ValidationError):
        Settings(**_MINIMUM_VALID, api_port=99999)


def test_settings_api_key_does_not_appear_in_repr():
    settings = Settings(**_MINIMUM_VALID)

    representation = repr(settings)

    assert "sk-ant-test-key" not in representation


def test_qdrant_defaults_are_correct():
    settings = Settings(**_MINIMUM_VALID)

    assert settings.qdrant_collection_name == "eidetic_memories"


def test_memory_pipeline_defaults_are_correct():
    settings = Settings(**_MINIMUM_VALID)

    assert settings.recency_window == 10
    assert settings.similarity_top_k == 10


def test_embedding_defaults_are_correct():
    settings = Settings(**_MINIMUM_VALID)

    assert settings.embedding_dimension == 1536
    assert settings.embedding_model == "text-embedding-3-small"


def test_settings_rejects_invalid_api_env():
    with pytest.raises(ValidationError):
        Settings(**_MINIMUM_VALID, api_env="staging")


def test_api_defaults_are_correct():
    settings = Settings(**_MINIMUM_VALID)

    assert settings.api_port == 8000
    assert settings.api_env == "development"
