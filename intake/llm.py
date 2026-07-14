"""Structured-LLM seam for intake.

Intake reasons about a question (typing, normalization, resolvability) via an
injected structured-output client. The seam is a narrow Protocol so the real
transport (the direct Anthropic API, or Bedrock) and a deterministic test
fixture are interchangeable — unit tests never touch the network (CLAUDE.md §2.8).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "FixtureStructuredLLM",
    "StructuredLLM",
]


@runtime_checkable
class StructuredLLM(Protocol):
    """Minimal structured-output seam (satisfied by the LLM transports)."""

    def invoke_structured(self, *, system: str, user: str) -> dict[str, Any]:
        """Return the parsed JSON object produced for a ``(system, user)`` prompt."""
        ...


class FixtureStructuredLLM:
    """Deterministic structured LLM for tests.

    Returns queued responses in call order; once the queue is exhausted it
    returns an empty object (which downstream parsers treat as "unknown"). All
    calls are recorded for assertions. No network, no randomness.
    """

    def __init__(
        self,
        responses: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ) -> None:
        if responses is None:
            queue: list[dict[str, Any]] = []
        elif isinstance(responses, Mapping):
            queue = [dict(responses)]
        else:
            queue = [dict(r) for r in responses]
        self._queue = queue
        self.calls: list[tuple[str, str]] = []

    def invoke_structured(self, *, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        if not self._queue:
            return {}
        return dict(self._queue.pop(0))
