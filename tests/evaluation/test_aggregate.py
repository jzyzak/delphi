"""Tests for bootstrap CIs + per-domain aggregation + baseline deltas (C6.4)."""

from __future__ import annotations

import pytest

from evaluation.aggregate import (
    baseline_delta,
    bootstrap_ci,
    per_domain_summary,
    summarize_scores,
)
from evaluation.baselines import Baseline
from evaluation.scoring import BrierScorer, ScoredRecord


def _record(qid: str, domain: str, p: float, o: float) -> ScoredRecord:
    return ScoredRecord(question_id=qid, domain=domain, probability=p, outcome=o)


class TestBootstrapCI:
    def test_interval_brackets_mean(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        lo, hi = bootstrap_ci(values, n_boot=500, seed=1)
        assert lo <= sum(values) / len(values) <= hi

    def test_deterministic_with_seed(self) -> None:
        values = [0.1, 0.9, 0.2, 0.8]
        assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            bootstrap_ci([])
        with pytest.raises(ValueError, match="alpha"):
            bootstrap_ci([0.5], alpha=1.5)


class TestSummarize:
    def test_summary_fields(self) -> None:
        records = [_record("q1", "d", 0.7, 1.0), _record("q2", "d", 0.3, 0.0)]
        summary = summarize_scores(BrierScorer(), records, n_boot=200, seed=0)
        assert summary.scorer == "brier"
        assert summary.mean == pytest.approx(0.09)
        assert summary.n == 2
        assert summary.ci_low <= summary.mean <= summary.ci_high

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            summarize_scores(BrierScorer(), [])


def test_per_domain_summary_groups_sorted() -> None:
    records = [
        _record("q1", "econ", 0.5, 1.0),
        _record("q2", "geo", 0.5, 0.0),
        _record("q3", "econ", 0.5, 0.0),
    ]
    summary = per_domain_summary(BrierScorer(), records, n_boot=100, seed=0)
    assert list(summary) == ["econ", "geo"]
    assert summary["econ"].n == 2


class TestBaselineDelta:
    def test_model_beats_baseline_negative(self) -> None:
        records = [_record("q1", "d", 0.9, 1.0), _record("q2", "d", 0.1, 0.0)]
        baseline = Baseline(name="weak", predictions={"q1": 0.5, "q2": 0.5})
        delta = baseline_delta(BrierScorer(), records, baseline)
        assert delta is not None
        assert delta < 0.0

    def test_no_coverage_returns_none(self) -> None:
        records = [_record("q1", "d", 0.5, 1.0)]
        baseline = Baseline(name="none", predictions={"other": 0.5})
        assert baseline_delta(BrierScorer(), records, baseline) is None
