"""Unit tests for common.http.config.HttpConfig (§8)."""

from __future__ import annotations

from typing import Any

import pytest

from common.http import HttpConfig


def test_defaults_are_valid() -> None:
    cfg = HttpConfig()
    assert cfg.max_retries == 4
    assert cfg.min_interval_s == 0.0
    assert cfg.user_agent is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_timeout_s": 0.0},
        {"max_retries": 0},
        {"retry_backoff_base": -1.0},
        {"retry_backoff_max": -1.0},
        {"min_interval_s": -0.1},
    ],
)
def test_invalid_values_raise(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        HttpConfig(**kwargs)


def test_zero_backoff_allowed_for_tests() -> None:
    cfg = HttpConfig(retry_backoff_base=0.0, retry_backoff_max=0.0)
    assert cfg.retry_backoff_base == 0.0
