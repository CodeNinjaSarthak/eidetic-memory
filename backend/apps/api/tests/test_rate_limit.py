"""Tests for the in-memory demo rate limiter."""

import pytest
from api.rate_limit import DemoRateLimiter
from fastapi import HTTPException

_DAY_START = 1_700_000_000.0  # arbitrary fixed epoch second, day-aligned enough for tests


def test_allows_requests_under_the_daily_limit():
    limiter = DemoRateLimiter(daily_limit=5, per_ip_minute_limit=100)

    for _ in range(5):
        limiter.check("1.2.3.4", now=_DAY_START)


def test_raises_429_once_daily_limit_is_exceeded():
    limiter = DemoRateLimiter(daily_limit=2, per_ip_minute_limit=100)
    limiter.check("1.2.3.4", now=_DAY_START)
    limiter.check("5.6.7.8", now=_DAY_START)

    with pytest.raises(HTTPException) as exc_info:
        limiter.check("9.9.9.9", now=_DAY_START)

    assert exc_info.value.status_code == 429


def test_daily_limit_resets_on_a_new_day():
    limiter = DemoRateLimiter(daily_limit=1, per_ip_minute_limit=100)
    limiter.check("1.2.3.4", now=_DAY_START)

    limiter.check("1.2.3.4", now=_DAY_START + 86400)


def test_raises_429_once_per_ip_minute_limit_is_exceeded():
    limiter = DemoRateLimiter(daily_limit=1000, per_ip_minute_limit=2)
    limiter.check("1.2.3.4", now=_DAY_START)
    limiter.check("1.2.3.4", now=_DAY_START + 1)

    with pytest.raises(HTTPException) as exc_info:
        limiter.check("1.2.3.4", now=_DAY_START + 2)

    assert exc_info.value.status_code == 429


def test_per_ip_minute_limit_is_tracked_separately_per_ip():
    limiter = DemoRateLimiter(daily_limit=1000, per_ip_minute_limit=1)
    limiter.check("1.2.3.4", now=_DAY_START)

    limiter.check("5.6.7.8", now=_DAY_START)


def test_per_ip_minute_limit_resets_on_a_new_minute():
    limiter = DemoRateLimiter(daily_limit=1000, per_ip_minute_limit=1)
    limiter.check("1.2.3.4", now=_DAY_START)

    limiter.check("1.2.3.4", now=_DAY_START + 60)
