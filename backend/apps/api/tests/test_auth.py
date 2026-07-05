"""Tests for the demo endpoint's HTTP Basic Auth guard."""

import pytest
from api.auth import verify_demo_credentials
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

from config.settings import Settings

_BASE_SETTINGS = {
    "_env_file": None,
    "qdrant_url": "http://localhost:6333",
    "llm_provider": "claude",
    "anthropic_api_key": "sk-ant-test-key",
    "embedding_provider": "openai",
    "openai_api_key": "sk-openai-test-key",
}


def test_allows_request_when_credentials_unset_in_development():
    settings = Settings(**_BASE_SETTINGS, api_env="development")

    verify_demo_credentials(settings=settings, credentials=None)


def test_blocks_all_requests_when_credentials_unset_in_production():
    settings = Settings(**_BASE_SETTINGS, api_env="production")

    with pytest.raises(HTTPException) as exc_info:
        verify_demo_credentials(settings=settings, credentials=None)

    assert exc_info.value.status_code == 403


def test_rejects_missing_credentials_when_demo_credentials_are_configured():
    settings = Settings(
        **_BASE_SETTINGS, demo_username="reviewer", demo_password="s3cret"
    )

    with pytest.raises(HTTPException) as exc_info:
        verify_demo_credentials(settings=settings, credentials=None)

    assert exc_info.value.status_code == 401


def test_rejects_wrong_password_when_demo_credentials_are_configured():
    settings = Settings(
        **_BASE_SETTINGS, demo_username="reviewer", demo_password="s3cret"
    )
    credentials = HTTPBasicCredentials(username="reviewer", password="wrong")

    with pytest.raises(HTTPException) as exc_info:
        verify_demo_credentials(settings=settings, credentials=credentials)

    assert exc_info.value.status_code == 401


def test_accepts_matching_credentials_when_demo_credentials_are_configured():
    settings = Settings(
        **_BASE_SETTINGS, demo_username="reviewer", demo_password="s3cret"
    )
    credentials = HTTPBasicCredentials(username="reviewer", password="s3cret")

    verify_demo_credentials(settings=settings, credentials=credentials)
