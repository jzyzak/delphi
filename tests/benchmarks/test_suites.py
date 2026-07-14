"""Tests for the retrospective evaluation suite loader."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from benchmarks.metaculus import MetaculusAdapter
from benchmarks.suites import (
    QuestionForecast,
    build_eval_context,
    forecaster_fn,
    records_baseline,
)
from core.forecast.leakage_judge import FixtureLeakageJudgeLLM, LeakageJudge
from core.orchestration.budget import InMemoryBudgetLedger
from evaluation.baselines import Baseline
from evaluation.calibration_split import assign_calibration_split
from evaluation.harness import EvalHarness


def _adapter(n: int = 6) -> MetaculusAdapter:
    records: list[dict[str, Any]] = []
    for i in range(n):
        records.append(
            {
                "id": i,
                "title": f"Will event {i} happen?",
                "as_of": "2026-01-01T00:00:00Z",
                "domain": "econ" if i % 2 == 0 else "geo",
                "community": 0.5,
                "resolution": float(i % 2),
                "resolved_at": "2026-06-01T00:00:00Z",
            }
        )
    return MetaculusAdapter.from_records(records)


def _harness() -> EvalHarness:
    return EvalHarness(budget_ledger=InMemoryBudgetLedger(cap=1000, trials_count=lambda: 0))


def _forecast_fn(prob: float = 0.6) -> Any:
    def _fn(_text: str, _as_of: datetime) -> QuestionForecast:
        return QuestionForecast(accepted=True, raw_probability=prob)

    return _fn


class TestBuildEvalContext:
    def test_scores_only_disjoint_test_split(self) -> None:
        adapter = _adapter(6)
        ctx = build_eval_context(
            adapter, _forecast_fn(), harness=_harness(), calibration_fraction=0.5, seed=0
        )
        accepted = sorted(q.question_id for q in adapter.questions())
        calibration = assign_calibration_split(accepted, fraction=0.5, seed=0)
        expected_test = {qid for qid in accepted if qid not in calibration}
        scored = {r.question_id for r in ctx.inputs.records}
        assert scored == expected_test
        # Calibration questions must never appear in the scored set (§2.5).
        assert not (scored & calibration)

    def test_baselines_and_judge_passthrough(self) -> None:
        adapter = _adapter(6)
        baseline = Baseline(name="market", predictions={"metaculus:0": 0.5})
        judge = LeakageJudge(FixtureLeakageJudgeLLM())
        ctx = build_eval_context(
            adapter,
            _forecast_fn(),
            harness=_harness(),
            judge=judge,
            extra_baselines=(baseline,),
        )
        assert ctx.inputs.baselines == (baseline,)
        assert ctx.judge is judge

    def test_identity_calibration_scores_all_accepted(self) -> None:
        adapter = _adapter(4)
        ctx = build_eval_context(
            adapter, _forecast_fn(), harness=_harness(), calibration_fraction=0.0
        )
        scored = {r.question_id for r in ctx.inputs.records}
        assert scored == {q.question_id for q in adapter.questions()}

    def test_refused_questions_excluded(self) -> None:
        adapter = _adapter(4)

        def _fn(_text: str, _as_of: datetime) -> QuestionForecast:
            return QuestionForecast(accepted=False, raw_probability=None)

        with pytest.raises(ValueError, match="no accepted forecasts"):
            build_eval_context(adapter, _fn, harness=_harness())

    def test_full_calibration_split_raises(self) -> None:
        adapter = _adapter(4)
        with pytest.raises(ValueError, match="lower calibration_fraction"):
            build_eval_context(
                adapter, _forecast_fn(), harness=_harness(), calibration_fraction=1.0
            )

    def test_report_renders_from_context(self) -> None:
        from evaluation.report import render_report

        adapter = _adapter(6)
        ctx = build_eval_context(adapter, _forecast_fn(), harness=_harness())
        rendered = render_report(ctx.inputs, harness=ctx.harness)
        assert "Proper scores" in rendered


class TestRecordsBaseline:
    def test_builds_predictions_from_records(self) -> None:
        records = [
            {"id": "abc", "freeze_value": 0.4},
            {"id": "def", "freeze_value": 0.9},
        ]
        baseline = records_baseline(records, source="forecastbench")
        assert baseline.predictions == {
            "forecastbench:abc": 0.4,
            "forecastbench:def": 0.9,
        }

    def test_skips_missing_or_unparseable_values(self) -> None:
        records = [
            {"id": "a"},  # no freeze_value
            {"freeze_value": 0.5},  # no id
            {"id": "b", "freeze_value": "n/a"},  # unparseable
            {"id": "c", "freeze_value": 0.7},
        ]
        baseline = records_baseline(records, source="forecastbench")
        assert baseline.predictions == {"forecastbench:c": 0.7}


class _FakeCalibrated:
    def __init__(self, raw: float) -> None:
        self.raw_probability = raw


class _FakeEvidence:
    def __init__(self) -> None:
        self.snippet = "as-of evidence text"
        self.knowledge_time = datetime(2025, 12, 1, tzinfo=UTC)
        self.source = "news"


class _FakeResult:
    def __init__(self, *, accepted: bool, raw: float) -> None:
        self.accepted = accepted
        self.question_id = "metaculus:0"
        self.calibrated = _FakeCalibrated(raw) if accepted else None
        self.evidence = (_FakeEvidence(),)
        self.rationale = "because reasons"


class _FakeForecaster:
    def __init__(self, *, accepted: bool = True, raw: float = 0.7) -> None:
        self._accepted = accepted
        self._raw = raw

    def forecast(self, _text: str, *, as_of: datetime) -> _FakeResult:
        return _FakeResult(accepted=self._accepted, raw=self._raw)


class TestForecasterFn:
    def test_maps_accepted_result_with_traces(self) -> None:
        fn = forecaster_fn(_FakeForecaster(raw=0.7))  # type: ignore[arg-type]
        out = fn("q", datetime(2026, 1, 1, tzinfo=UTC))
        assert out.accepted
        assert out.raw_probability == 0.7
        # One search trace (evidence) + one supervisor trace (rationale).
        assert len(out.traces) == 2

    def test_refused_result_has_no_probability(self) -> None:
        fn = forecaster_fn(_FakeForecaster(accepted=False))  # type: ignore[arg-type]
        out = fn("q", datetime(2026, 1, 1, tzinfo=UTC))
        assert not out.accepted
        assert out.raw_probability is None

    def test_skips_undated_evidence_and_empty_rationale(self) -> None:
        class _Undated:
            snippet = "no date here"
            knowledge_time = None  # undated -> skipped as a trace
            source = "news"

        class _Result:
            accepted = True
            question_id = "metaculus:0"
            calibrated = _FakeCalibrated(0.5)
            evidence = (_Undated(),)
            rationale = ""  # empty -> no supervisor trace

        class _Forecaster:
            def forecast(self, _text: str, *, as_of: datetime) -> _Result:
                return _Result()

        fn = forecaster_fn(_Forecaster())  # type: ignore[arg-type]
        out = fn("q", datetime(2026, 1, 1, tzinfo=UTC))
        assert out.accepted
        assert out.traces == ()
