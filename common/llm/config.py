"""Typed configuration for the LLM transport layer.

The ``thinking`` and ``effort`` fields are Anthropic-transport-only for now:
the direct Anthropic API transport honors them; other transports (Bedrock)
ignore them.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LLMConfig"]

# Allowed values for the Anthropic-only knobs (None = feature off).
_VALID_THINKING = frozenset({None, "adaptive"})
_VALID_EFFORT = frozenset({None, "low", "medium", "high", "xhigh", "max"})


@dataclass(frozen=True)
class LLMConfig:
    """Inference + transport parameters for a Bedrock structured client.

    ``temperature`` is intentionally > 0 by default: the forecast ensemble (18)
    relies on independent, varied draws. Reproducibility is provided by the
    content-addressed ensemble cache (the data of record, CLAUDE.md section 6),
    not by seeding the model.

    Retry backoff is parameterized so tests can drive it to zero (no real
    sleeps) while production uses exponential backoff.

    ``thinking`` and ``effort`` are honored by the Anthropic transport only for
    now (Bedrock ignores them). ``thinking="adaptive"`` enables adaptive
    thinking (and drops sampling parameters, which adaptive-thinking models
    reject); ``effort`` maps to the Messages API ``output_config.effort``.
    """

    temperature: float = 1.0
    max_tokens: int = 1024
    top_p: float = 0.95
    max_concurrency: int = 8
    request_timeout_s: float = 60.0
    max_retries: int = 4
    retry_backoff_base: float = 0.5
    retry_backoff_max: float = 8.0
    thinking: str | None = None
    effort: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            msg = f"temperature must be in [0, 2], got {self.temperature!r}"
            raise ValueError(msg)
        if self.max_tokens <= 0:
            msg = f"max_tokens must be positive, got {self.max_tokens!r}"
            raise ValueError(msg)
        if not 0.0 < self.top_p <= 1.0:
            msg = f"top_p must be in (0, 1], got {self.top_p!r}"
            raise ValueError(msg)
        if self.max_concurrency < 1:
            msg = f"max_concurrency must be >= 1, got {self.max_concurrency!r}"
            raise ValueError(msg)
        if self.request_timeout_s <= 0:
            msg = f"request_timeout_s must be positive, got {self.request_timeout_s!r}"
            raise ValueError(msg)
        if self.max_retries < 1:
            msg = f"max_retries must be >= 1, got {self.max_retries!r}"
            raise ValueError(msg)
        if self.retry_backoff_base < 0.0 or self.retry_backoff_max < 0.0:
            msg = "retry backoff parameters must be non-negative"
            raise ValueError(msg)
        if self.thinking not in _VALID_THINKING:
            msg = f"thinking must be None or 'adaptive', got {self.thinking!r}"
            raise ValueError(msg)
        if self.effort not in _VALID_EFFORT:
            allowed = sorted(v for v in _VALID_EFFORT if v is not None)
            msg = f"effort must be None or one of {allowed}, got {self.effort!r}"
            raise ValueError(msg)
