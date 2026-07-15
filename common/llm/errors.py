"""Typed errors for the LLM transport layer."""

from __future__ import annotations

__all__ = [
    "LLMError",
    "LLMThrottledError",
    "MalformedLLMOutput",
]


class LLMError(RuntimeError):
    """Base class for LLM transport failures."""


class LLMThrottledError(LLMError):
    """Raised when the provider throttles or is transiently unavailable.

    Retryable: the transport retries this with exponential backoff.
    """


class LLMRefusedError(LLMError):
    """The provider's safety layer declined the request (stop_reason=refusal).

    Deliberately NOT retryable: the same input will be declined again, so
    retrying only burns budget. Callers treat it like a per-question refusal.
    """


class MalformedLLMOutput(LLMError):
    """Raised when provider output cannot be parsed into the expected structure.

    Retryable: a fresh sample may produce well-formed output, so the transport
    retries this up to ``LLMConfig.max_retries`` before propagating.
    """
