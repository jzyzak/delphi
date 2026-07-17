"""Typed errors for the HTTP transport layer."""

from __future__ import annotations

__all__ = [
    "HttpError",
    "HttpNotFound",
    "HttpRateLimited",
    "HttpTransportFailure",
]


class HttpError(RuntimeError):
    """Base class for HTTP transport failures."""


class HttpTransportFailure(HttpError):
    """Network-level failure (connect/handshake/read) after the retry budget.

    Wraps ``httpx.TransportError`` so callers see ONE exception taxonomy: a
    provider dying on an SSL handshake must be skippable by the composite
    searcher exactly like a 429, never a raw httpx crash that kills a run.
    """


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
