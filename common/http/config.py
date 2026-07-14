"""Typed configuration for the HTTP transport layer."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HttpConfig"]


@dataclass(frozen=True)
class HttpConfig:
    """Transport parameters for an :class:`~common.http.client.HttpClient`.

    ``min_interval_s`` enforces a minimum spacing between requests (politeness /
    rate limiting), e.g. SEC EDGAR asks for <= 10 req/s. ``user_agent`` is sent
    on every request; some sources (EDGAR) require a descriptive one.

    Retry backoff is parameterized so tests can drive it to zero (no real
    sleeps) while production uses exponential backoff.
    """

    request_timeout_s: float = 30.0
    max_retries: int = 4
    retry_backoff_base: float = 0.5
    retry_backoff_max: float = 8.0
    min_interval_s: float = 0.0
    user_agent: str | None = None

    def __post_init__(self) -> None:
        if self.request_timeout_s <= 0:
            msg = f"request_timeout_s must be positive, got {self.request_timeout_s!r}"
            raise ValueError(msg)
        if self.max_retries < 1:
            msg = f"max_retries must be >= 1, got {self.max_retries!r}"
            raise ValueError(msg)
        if self.retry_backoff_base < 0.0 or self.retry_backoff_max < 0.0:
            msg = "retry backoff parameters must be non-negative"
            raise ValueError(msg)
        if self.min_interval_s < 0.0:
            msg = f"min_interval_s must be non-negative, got {self.min_interval_s!r}"
            raise ValueError(msg)
