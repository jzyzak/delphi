"""Unit tests for FixtureForecastLLM (§8)."""

from __future__ import annotations

import pytest

from core.forecast.llm import FixtureForecastLLM, ForecastRequest


def _request(content_hash: str, run_index: int = 0) -> ForecastRequest:
    return ForecastRequest(
        content="doc body",
        content_hash=content_hash,
        run_index=run_index,
        prompt="forecast",
    )


class TestFixtureForecastLLM:
    def test_happy_path_explicit_responses(self) -> None:
        llm = FixtureForecastLLM({"abc": (0.3, 0.7)})
        draws = llm.forecast_batch([_request("abc", 0), _request("abc", 1)])
        assert llm.batch_call_count == 1
        assert llm.request_count == 2
        assert draws[0].probability == pytest.approx(0.3)
        assert draws[1].probability == pytest.approx(0.7)

    def test_boundary_default_response(self) -> None:
        llm = FixtureForecastLLM(default_response=0.55)
        draws = llm.forecast_batch([_request("missing", 0)])
        assert draws[0].probability == pytest.approx(0.55)

    def test_boundary_base_plus_noise_is_clipped(self) -> None:
        llm = FixtureForecastLLM(base_probability=0.5, noise_std=0.01, seed=1)
        draws = llm.forecast_batch([_request("h1", i) for i in range(5)])
        assert all(0.0 <= d.probability <= 1.0 for d in draws)

    def test_failure_empty_batch_returns_empty(self) -> None:
        llm = FixtureForecastLLM()
        assert llm.forecast_batch([]) == ()
        assert llm.batch_call_count == 0

    def test_provenance_carried_on_draws(self) -> None:
        llm = FixtureForecastLLM(default_response=0.5)
        draws = llm.forecast_batch([_request("hash", 3)])
        assert draws[0].provenance["run_index"] == 3
        assert draws[0].provenance["content_hash"] == "hash"
