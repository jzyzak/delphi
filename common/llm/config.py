"""Typed configuration for the LLM transport layer."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LLMConfig"]


@dataclass(frozen=True)
class LLMConfig:
    """Inference + transport parameters for a Bedrock structured client.

    ``temperature`` is intentionally > 0 by default: the forecast ensemble (18)
    relies on independent, varied draws. Reproducibility is provided by the
    content-addressed ensemble cache (the data of record, CLAUDE.md section 6),
    not by seeding the model.

    Retry backoff is parameterized so tests can drive it to zero (no real
    sleeps) while production uses exponential backoff.
    """

    temperature: float = 1.0
    max_tokens: int = 1024
    top_p: float = 0.95
    max_concurrency: int = 8
    request_timeout_s: float = 60.0
    max_retries: int = 4
    retry_backoff_base: float = 0.5
    retry_backoff_max: float = 8.0

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
