"""Content-addressed ensemble cache — the data of record for forecast ensembles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.forecast.ensemble import EnsembleForecast


@dataclass(frozen=True)
class EnsembleCacheKey:
    """Content-addressed ensemble cache key."""

    content_hash: str
    model_version: str
    prompt_version: str
    ensemble_config: str


class EnsembleCache(ABC):
    """Append-only ensemble cache — identical keys are idempotent."""

    @abstractmethod
    def get(self, key: EnsembleCacheKey) -> EnsembleForecast | None:
        """Return cached ensemble for an addressing key, or None on miss."""

    @abstractmethod
    def put(self, key: EnsembleCacheKey, forecast: EnsembleForecast) -> None:
        """Append cache entry; identical keys are idempotent."""


class InMemoryEnsembleCache(EnsembleCache):
    """In-memory ensemble cache for tests and local development."""

    def __init__(self) -> None:
        self._entries: dict[EnsembleCacheKey, EnsembleForecast] = {}

    def get(self, key: EnsembleCacheKey) -> EnsembleForecast | None:
        return self._entries.get(key)

    def put(self, key: EnsembleCacheKey, forecast: EnsembleForecast) -> None:
        if key in self._entries:
            return
        self._entries[key] = forecast

    @property
    def keys(self) -> tuple[EnsembleCacheKey, ...]:
        return tuple(self._entries.keys())
