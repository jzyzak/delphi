"""Typed errors for the HTTP transport layer."""

from __future__ import annotations

__all__ = [
    "HttpError",
    "HttpNotFound",
    "HttpRateLimited",
]


class HttpError(RuntimeError):
    """Base class for HTTP transport failures."""


class HttpRateLimited(HttpError):
    """Raised on HTTP 429. Retryable with backoff."""


class HttpNotFound(HttpError):
    """Raised on HTTP 404. Not retryable."""


class _TransientHttpError(HttpError):
    """Internal marker for retryable 5xx responses.

    Subclasses :class:`HttpError` so callers that catch ``HttpError`` after the
    retry budget is exhausted still handle it; kept private because callers
    should not branch on transient-vs-permanent server errors.
    """
