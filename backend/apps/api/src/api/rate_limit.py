"""In-memory request caps for the demo endpoint.

Protects against a leaked demo credential running up unbounded paid LLM
calls: a global daily cap plus a per-IP per-minute burst cap. Counters are
process-local — safe only when the API runs as a single instance.
"""

import time
from threading import Lock
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from api.dependencies import get_settings
from config.settings import Settings

_SECONDS_PER_DAY = 86400
_SECONDS_PER_MINUTE = 60


class DemoRateLimiter:
    """Tracks a global daily count and a per-IP per-minute count."""

    def __init__(self, daily_limit: int, per_ip_minute_limit: int) -> None:
        self._daily_limit = daily_limit
        self._per_ip_minute_limit = per_ip_minute_limit
        self._lock = Lock()
        self._day_bucket: int | None = None
        self._day_count = 0
        self._ip_buckets: dict[str, tuple[int, int]] = {}

    def check(self, client_ip: str, now: float | None = None) -> None:
        """Raise HTTPException(429) if the daily or per-IP cap is exceeded."""
        now = now if now is not None else time.time()
        current_day = int(now // _SECONDS_PER_DAY)
        current_minute = int(now // _SECONDS_PER_MINUTE)

        with self._lock:
            if self._day_bucket != current_day:
                self._day_bucket = current_day
                self._day_count = 0
            self._day_count += 1
            if self._day_count > self._daily_limit:
                raise HTTPException(
                    status_code=429, detail="Demo daily query limit reached"
                )

            bucket_minute, bucket_count = self._ip_buckets.get(client_ip, (current_minute, 0))
            if bucket_minute != current_minute:
                bucket_minute, bucket_count = current_minute, 0
            bucket_count += 1
            self._ip_buckets[client_ip] = (bucket_minute, bucket_count)
            if bucket_count > self._per_ip_minute_limit:
                raise HTTPException(
                    status_code=429, detail="Too many demo requests, slow down"
                )


_demo_rate_limiter: DemoRateLimiter | None = None


def get_demo_rate_limiter(settings: Settings) -> DemoRateLimiter:
    """Return the process-wide demo rate limiter, built from settings once."""
    global _demo_rate_limiter
    if _demo_rate_limiter is None:
        _demo_rate_limiter = DemoRateLimiter(
            daily_limit=settings.demo_daily_query_limit,
            per_ip_minute_limit=settings.demo_per_ip_minute_limit,
        )
    return _demo_rate_limiter


def _client_ip(request: Request) -> str:
    """Prefer the originating client from X-Forwarded-For behind a proxy."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_demo_rate_limit(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """FastAPI dependency enforcing the demo rate caps for the current request."""
    limiter = get_demo_rate_limiter(settings)
    limiter.check(_client_ip(request))
