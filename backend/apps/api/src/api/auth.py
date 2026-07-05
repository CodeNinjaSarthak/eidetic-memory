"""HTTP Basic Auth guard for the read-only demo surface."""

from hmac import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from api.dependencies import get_settings
from config.settings import Settings

_basic_auth = HTTPBasic(auto_error=False)


def verify_demo_credentials(
    settings: Annotated[Settings, Depends(get_settings)],
    credentials: Annotated[HTTPBasicCredentials | None, Depends(_basic_auth)],
) -> None:
    """Require DEMO_USERNAME/DEMO_PASSWORD via HTTP Basic Auth.

    If the credentials are unset: fail closed (403 all requests) in production,
    but allow requests through unauthenticated in development/test so local
    work doesn't require setting demo credentials.
    """
    if settings.demo_username is None or settings.demo_password is None:
        if settings.api_env == "production":
            raise HTTPException(
                status_code=403, detail="Demo access is not configured"
            )
        return

    expected_username = settings.demo_username.get_secret_value()
    expected_password = settings.demo_password.get_secret_value()
    is_valid = (
        credentials is not None
        and compare_digest(credentials.username, expected_username)
        and compare_digest(credentials.password, expected_password)
    )
    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid demo credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
