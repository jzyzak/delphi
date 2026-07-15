"""Unit tests for common.llm.config.LLMConfig (§8)."""

from __future__ import annotations

from typing import Any

import pytest

from common.llm import LLMConfig


def test_defaults_are_valid() -> None:
    cfg = LLMConfig()
    assert cfg.temperature == pytest.approx(1.0)
    assert cfg.max_concurrency == 8
    assert cfg.max_retries == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"temperature": 2.5},
        {"max_tokens": 0},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"max_concurrency": 0},
        {"request_timeout_s": 0.0},
        {"max_retries": 0},
        {"retry_backoff_base": -1.0},
        {"thinking": "extended"},
        {"thinking": ""},
        {"effort": "ultra"},
        {"effort": ""},
    ],
)
def test_invalid_values_raise(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LLMConfig(**kwargs)


def test_zero_backoff_is_allowed_for_tests() -> None:
    cfg = LLMConfig(retry_backoff_base=0.0, retry_backoff_max=0.0)
    assert cfg.retry_backoff_base == 0.0


def test_thinking_and_effort_default_to_none() -> None:
    cfg = LLMConfig()
    assert cfg.thinking is None
    assert cfg.effort is None


def test_adaptive_thinking_is_valid() -> None:
    assert LLMConfig(thinking="adaptive").thinking == "adaptive"


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_valid_effort_levels(effort: str) -> None:
    assert LLMConfig(effort=effort).effort == effort
