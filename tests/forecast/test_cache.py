"""Unit tests for ensemble cache (§8)."""

from __future__ import annotations

from datetime import UTC, datetime

from core.forecast.cache import EnsembleCacheKey, InMemoryEnsembleCache
from core.forecast.ensemble import EnsembleForecast, build_ensemble
from core.forecast.llm import ForecastDraw


def _sample_ensemble() -> EnsembleForecast:
    draws = (
        ForecastDraw(
            probability=0.4,
            run_index=0,
            model_version="m1",
            prompt_version="p1",
        ),
        ForecastDraw(
            probability=0.6,
            run_index=1,
            model_version="m1",
            prompt_version="p1",
        ),
    )
    return build_ensemble(
        draws,
        aggregator="median",
        knowledge_time=datetime(2024, 1, 1, tzinfo=UTC),
    )


class TestInMemoryEnsembleCache:
    def test_happy_path_put_and_get(self) -> None:
        cache = InMemoryEnsembleCache()
        key = EnsembleCacheKey("h1", "m1", "p1", "n=2|agg=median|trim=0.1|spread=std")
        ensemble = _sample_ensemble()
        cache.put(key, ensemble)
        assert cache.get(key) is ensemble

    def test_boundary_idempotent_put(self) -> None:
        cache = InMemoryEnsembleCache()
        key = EnsembleCacheKey("h1", "m1", "p1", "cfg")
        ensemble = _sample_ensemble()
        cache.put(key, ensemble)
        cache.put(key, ensemble)
        assert len(cache.keys) == 1

    def test_failure_miss_returns_none(self) -> None:
        cache = InMemoryEnsembleCache()
        key = EnsembleCacheKey("missing", "m1", "p1", "cfg")
        assert cache.get(key) is None
