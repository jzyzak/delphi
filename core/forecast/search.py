"""Domain-agnostic as-of search seam for forecast-layer resolution.

Specialized layers (e.g. the evidence providers in ``sources/``) implement
``AsOfSearcher`` and inject it into the supervisor. The core never imports
specialized search backends.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.pit.models import ensure_utc


class Evidence(BaseModel):
    """One retrieved snippet pinned at or before the forecast as-of."""

    model_config = ConfigDict(frozen=True)

    snippet: str
    source: str
    source_id: str
    knowledge_time: datetime
    score: float = Field(ge=0.0)
    query: str = ""

    @field_validator("knowledge_time")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


@runtime_checkable
class AsOfSearcher(Protocol):
    """Point-in-time search pinned at ``as_of`` — the forecast-layer seam."""

    def as_of_search(self, query: str, *, as_of: datetime) -> Sequence[Evidence]:
        """Return evidence with ``knowledge_time <= as_of`` only."""
        ...


class FixtureAsOfSearch:
    """Deterministic as-of search for tests (no network, no PIT store)."""

    def __init__(
        self,
        responses: Mapping[str, Sequence[Evidence]] | None = None,
        *,
        default: Sequence[Evidence] = (),
    ) -> None:
        self._responses = {k: tuple(v) for k, v in (responses or {}).items()}
        self._default = tuple(default)
        self.call_count = 0
        self.queries: list[tuple[str, datetime]] = []

    def as_of_search(self, query: str, *, as_of: datetime) -> Sequence[Evidence]:
        self.call_count += 1
        as_of = ensure_utc(as_of)
        self.queries.append((query, as_of))
        key = query.strip().lower()
        if key in self._responses:
            return self._responses[key]
        return self._default


__all__ = [
    "AsOfSearcher",
    "Evidence",
    "FixtureAsOfSearch",
]
