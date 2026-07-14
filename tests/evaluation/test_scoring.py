"""Unit + properness tests for proper scorers (C6.1)."""

from __future__ import annotations

import math

import pytest

from evaluation.scoring import (
    BrierScorer,
    CRPSScorer,
    LogScorer,
    ScoredRecord,
    brier_score,
    crps_from_quantiles,
    log_score,
    mean_score,
)


class TestBrier:
    def test_reference_values(self) -> None:
        assert brier_score(0.7, 1.0) == pytest.approx(0.09)
        assert brier_score(0.3, 0.0) == pytest.approx(0.09)
        assert brier_score(1.0, 1.0) == 0.0

    def test_properness(self) -> None:
        # Expected Brier at true p is minimized by reporting p.
        true_p = 0.7

        def expected(report: float) -> float:
            return true_p * brier_score(report, 1.0) + (1 - true_p) * brier_score(report, 0.0)

        assert expected(0.7) < expected(0.5)
        assert expected(0.7) < expected(0.9)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            brier_score(1.5, 1.0)
        with pytest.raises(ValueError, match="binary outcome"):
            brier_score(0.5, 0.5)


class TestLog:
    def test_reference_values(self) -> None:
        assert log_score(0.5, 1.0) == pytest.approx(math.log(2))
        assert log_score(1.0, 1.0) == pytest.approx(0.0, abs=1e-9)

    def test_properness(self) -> None:
        true_p = 0.7

        def expected(report: float) -> float:
            return true_p * log_score(report, 1.0) + (1 - true_p) * log_score(report, 0.0)

        assert expected(0.7) < expected(0.5)
        assert expected(0.7) < expected(0.95)


class TestCRPS:
    def test_point_forecast_at_outcome_is_zero(self) -> None:
        quantiles = [(0.25, 5.0), (0.5, 5.0), (0.75, 5.0)]
        assert crps_from_quantiles(quantiles, 5.0) == pytest.approx(0.0)

    def test_closer_is_better(self) -> None:
        q_close = [(0.5, 4.0)]
        q_far = [(0.5, 1.0)]
        assert crps_from_quantiles(q_close, 5.0) < crps_from_quantiles(q_far, 5.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            crps_from_quantiles([], 1.0)

    def test_bad_level_raises(self) -> None:
        with pytest.raises(ValueError, match="quantile level"):
            crps_from_quantiles([(1.5, 1.0)], 1.0)


class TestScorerInterface:
    def test_names_and_scores(self) -> None:
        assert BrierScorer().name == "brier"
        assert LogScorer().name == "log"
        assert CRPSScorer().name == "crps"
        assert BrierScorer().score(0.7, 1.0) == pytest.approx(0.09)
        assert CRPSScorer().score([(0.5, 5.0)], 5.0) == pytest.approx(0.0)

    def test_mean_score(self) -> None:
        assert mean_score(BrierScorer(), [0.7, 0.3], [1.0, 0.0]) == pytest.approx(0.09)

    def test_mean_score_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            mean_score(BrierScorer(), [0.5], [1.0, 0.0])

    def test_mean_score_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            mean_score(BrierScorer(), [], [])


def test_scored_record_defaults() -> None:
    r = ScoredRecord(question_id="q", domain="d", probability=0.5, outcome=1.0)
    assert r.baselines == {}
