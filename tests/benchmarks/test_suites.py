"""Tests for the retrospective evaluation suite loader."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from benchmarks.metaculus import MetaculusAdapter
from benchmarks.suites import (
    QuestionForecast,
    build_eval_context,
    constant_baseline,
    filter_records_by_source,
    forecaster_fn,
    records_baseline,
    sample_records,
)
from core.forecast.calibration import FrozenCalibration, calibrate
from core.forecast.leakage_judge import FixtureLeakageJudgeLLM, LeakageJudge
from core.orchestration.budget import InMemoryBudgetLedger
from evaluation.baselines import Baseline
from evaluation.calibration_split import assign_calibration_split, question_fingerprint
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
    def _fn(_text: str, _as_of: datetime, _metadata: Any = None) -> QuestionForecast:
        return QuestionForecast(accepted=True, raw_probability=prob)

    return _fn


class TestBuildEvalContext:
    def test_threads_benchmark_id_metadata_to_forecast_fn(self) -> None:
        from resolution.benchmark_source import BENCHMARK_QUESTION_ID_KEY

        seen: dict[str, Any] = {}

        def _fn(_text: str, _as_of: datetime, metadata: Any = None) -> QuestionForecast:
            seen[metadata[BENCHMARK_QUESTION_ID_KEY]] = metadata
            return QuestionForecast(accepted=True, raw_probability=0.6)

        adapter = _adapter(4)
        build_eval_context(adapter, _fn, harness=_harness())
        # Every forecast carried the benchmark identity, so its registry record
        # is later resolvable (and can become calibration corpus).
        assert set(seen) == {q.question_id for q in adapter.questions()}
        for qid, metadata in seen.items():
            assert metadata["benchmark_source"] == "metaculus"
            assert f"metaculus:{metadata['benchmark_external_id']}" == qid
            # The crowd value at the freeze rides along as the market anchor.
            assert metadata["market_freeze_value"] == 0.5
            # No close_time on these records: the scheduled resolution date
            # falls back to the benchmark resolution's date (the estimator's
            # horizon; part of the question's own definition, so as-of-safe).
            assert metadata["resolution_date"] == "2026-06-01T00:00:00+00:00"

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

        def _fn(_text: str, _as_of: datetime, _metadata: Any = None) -> QuestionForecast:
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

    def test_one_failing_question_does_not_kill_the_run(self) -> None:
        # A transient failure (network drop, provider outage) on one question
        # must drop that question, not the whole eval.
        calls = {"n": 0}

        def _fn(_text: str, _as_of: datetime, _metadata: Any = None) -> QuestionForecast:
            calls["n"] += 1
            if calls["n"] == 2:
                raise ConnectionError("network dropped")
            return QuestionForecast(accepted=True, raw_probability=0.6)

        adapter = _adapter(6)
        ctx = build_eval_context(
            adapter, _fn, harness=_harness(), calibration=_identity_calibration()
        )
        assert calls["n"] == 6
        assert len(ctx.inputs.records) == 5  # the failed question is dropped

    def test_all_questions_failing_raises(self) -> None:
        def _fn(_text: str, _as_of: datetime, _metadata: Any = None) -> QuestionForecast:
            raise ConnectionError("network gone")

        with pytest.raises(ValueError, match="no accepted forecasts"):
            build_eval_context(_adapter(4), _fn, harness=_harness())

    def test_within_run_mode_reports_calibration_provenance(self) -> None:
        ctx = build_eval_context(_adapter(6), _forecast_fn(), harness=_harness())
        prov = ctx.inputs.calibration_provenance
        assert prov is not None
        assert prov["source"] == "within-run split"
        # A 3-point within-run fit is below the trust threshold: labeled fallback.
        assert prov["fallback"] is True


class TestFilterRecordsBySource:
    _RECORDS = (
        {"id": "fred-DFF"},
        {"id": "dbnomics-a_b_c.D"},
        {"id": "acled-abc123"},
        {"id": "yfinance-MMM"},
        {"id": ""},
    )

    def test_keeps_only_matching_prefixes(self) -> None:
        kept = filter_records_by_source(list(self._RECORDS), ["fred", "dbnomics"])
        assert [r["id"] for r in kept] == ["fred-DFF", "dbnomics-a_b_c.D"]

    def test_matching_is_case_and_whitespace_tolerant(self) -> None:
        kept = filter_records_by_source(list(self._RECORDS), [" FRED "])
        assert [r["id"] for r in kept] == ["fred-DFF"]

    def test_unknown_source_keeps_nothing(self) -> None:
        assert filter_records_by_source(list(self._RECORDS), ["polymarket"]) == []

    def test_empty_sources_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one source"):
            filter_records_by_source(list(self._RECORDS), [" ", ""])


def _identity_calibration(**overrides: Any) -> FrozenCalibration:
    kwargs: dict[str, Any] = {
        "method": "platt",
        "alpha": 1.0,
        "a": 1.0,
        "b": 0.0,
        "n": 40,
        "artifact_hash": "abc123",
    }
    kwargs.update(overrides)
    return FrozenCalibration(**kwargs)


class TestArtifactModeEvalContext:
    def test_scores_all_accepted_questions(self) -> None:
        adapter = _adapter(6)
        ctx = build_eval_context(
            adapter, _forecast_fn(), harness=_harness(), calibration=_identity_calibration()
        )
        scored = {r.question_id for r in ctx.inputs.records}
        assert scored == {q.question_id for q in adapter.questions()}

    def test_applies_the_full_composed_map(self) -> None:
        # alpha=2 with an identity recalibrator: raw 0.6 -> extremized ~0.6923
        # (inside the 0.2 floor band), so the artifact's numbers are provably
        # applied rather than the within-run fit.
        calibration = _identity_calibration(alpha=2.0, floor=0.2)
        ctx = build_eval_context(
            _adapter(4), _forecast_fn(0.6), harness=_harness(), calibration=calibration
        )
        expected = calibrate(0.6, alpha=2.0)
        for record in ctx.inputs.records:
            assert record.probability == pytest.approx(expected, abs=1e-6)

    def test_floor_clamps_in_artifact_mode(self) -> None:
        calibration = _identity_calibration(alpha=1.0, floor=0.45)
        ctx = build_eval_context(
            _adapter(4), _forecast_fn(0.9), harness=_harness(), calibration=calibration
        )
        for record in ctx.inputs.records:
            assert record.probability == pytest.approx(0.55, abs=1e-6)

    def test_provenance_carries_artifact_identity(self) -> None:
        ctx = build_eval_context(
            _adapter(4), _forecast_fn(), harness=_harness(), calibration=_identity_calibration()
        )
        prov = ctx.inputs.calibration_provenance
        assert prov is not None
        assert prov["artifact_hash"] == "abc123"

    def test_fit_set_overlap_by_id_raises(self) -> None:
        adapter = _adapter(4)
        some_qid = next(iter(q.question_id for q in adapter.questions()))
        calibration = _identity_calibration(
            fitted_meta={"question_fingerprints": [question_fingerprint(some_qid)]}
        )
        with pytest.raises(ValueError, match="fit set"):
            build_eval_context(adapter, _forecast_fn(), harness=_harness(), calibration=calibration)

    def test_fit_set_overlap_by_text_raises(self) -> None:
        adapter = _adapter(4)
        calibration = _identity_calibration(
            fitted_meta={"question_fingerprints": [question_fingerprint("Will event 1 happen?")]}
        )
        with pytest.raises(ValueError, match="fit set"):
            build_eval_context(adapter, _forecast_fn(), harness=_harness(), calibration=calibration)

    def test_exclude_fit_questions_drops_overlap_before_forecasting(self) -> None:
        adapter = _adapter(4)
        fit_qid = sorted(q.question_id for q in adapter.questions())[0]
        calibration = _identity_calibration(
            fitted_meta={"question_fingerprints": [question_fingerprint(fit_qid)]}
        )
        forecast_calls: list[str] = []

        def _fn(text: str, _as_of: datetime, _metadata: Any = None) -> QuestionForecast:
            forecast_calls.append(text)
            return QuestionForecast(accepted=True, raw_probability=0.6)

        ctx = build_eval_context(
            adapter,
            _fn,
            harness=_harness(),
            calibration=calibration,
            exclude_fit_questions=True,
        )
        scored = {r.question_id for r in ctx.inputs.records}
        assert fit_qid not in scored
        assert len(scored) == 3
        # Excluded BEFORE forecasting: no LLM/trials cost for the spent question.
        assert len(forecast_calls) == 3
        prov = ctx.inputs.calibration_provenance
        assert prov is not None
        assert prov["excluded_fit_overlap"] == 1

    def test_exclude_fit_questions_with_no_overlap_reports_nothing(self) -> None:
        calibration = _identity_calibration(
            fitted_meta={"question_fingerprints": [question_fingerprint("unrelated")]}
        )
        ctx = build_eval_context(
            _adapter(4),
            _forecast_fn(),
            harness=_harness(),
            calibration=calibration,
            exclude_fit_questions=True,
        )
        prov = ctx.inputs.calibration_provenance
        assert prov is not None
        assert "excluded_fit_overlap" not in prov

    def test_exclude_fit_questions_everything_overlapping_raises(self) -> None:
        adapter = _adapter(4)
        fingerprints = [question_fingerprint(q.question_id) for q in adapter.questions()]
        calibration = _identity_calibration(fitted_meta={"question_fingerprints": fingerprints})
        with pytest.raises(ValueError, match="every resolved question"):
            build_eval_context(
                adapter,
                _forecast_fn(),
                harness=_harness(),
                calibration=calibration,
                exclude_fit_questions=True,
            )

    def test_disjoint_fit_set_passes(self) -> None:
        calibration = _identity_calibration(
            fitted_meta={"question_fingerprints": [question_fingerprint("unrelated question")]}
        )
        ctx = build_eval_context(
            _adapter(4), _forecast_fn(), harness=_harness(), calibration=calibration
        )
        assert len(ctx.inputs.records) == 4


def _fb_records(counts: dict[str, int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source, count in counts.items():
        for i in range(count):
            records.append({"id": f"{source}-{i:03d}", "question": f"{source} q{i}"})
    return records


class TestSampleRecords:
    def test_deterministic_for_same_seed(self) -> None:
        records = _fb_records({"acled": 20, "fred": 20, "manifold": 10})
        first = sample_records(records, n=10, seed=7)
        second = sample_records(records, n=10, seed=7)
        assert [r["id"] for r in first] == [r["id"] for r in second]

    def test_different_seed_differs(self) -> None:
        records = _fb_records({"acled": 30, "fred": 30})
        a = [r["id"] for r in sample_records(records, n=10, seed=1)]
        b = [r["id"] for r in sample_records(records, n=10, seed=2)]
        assert a != b

    def test_stratified_proportional_allocation(self) -> None:
        records = _fb_records({"acled": 30, "fred": 30, "manifold": 30, "polymarket": 10})
        sampled = sample_records(records, n=10, seed=0)
        by_source: dict[str, int] = {}
        for record in sampled:
            source = str(record["id"]).split("-", 1)[0]
            by_source[source] = by_source.get(source, 0) + 1
        assert sum(by_source.values()) == 10
        assert by_source["acled"] == 3
        assert by_source["fred"] == 3
        assert by_source["manifold"] == 3
        assert by_source["polymarket"] == 1

    def test_n_at_least_total_returns_all(self) -> None:
        records = _fb_records({"acled": 3})
        assert sample_records(records, n=3, seed=0) == records
        assert sample_records(records, n=99, seed=0) == records

    def test_tiny_stratum_never_overflows(self) -> None:
        records = _fb_records({"acled": 1, "fred": 50})
        sampled = sample_records(records, n=10, seed=0)
        assert len(sampled) == 10
        acled = [r for r in sampled if str(r["id"]).startswith("acled")]
        assert len(acled) <= 1

    def test_exact_sample_size_honored(self) -> None:
        records = _fb_records({"a": 7, "b": 5, "c": 3})
        for n in (1, 5, 10, 14):
            assert len(sample_records(records, n=n, seed=0)) == n

    def test_invalid_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            sample_records(_fb_records({"a": 2}), n=0)


class TestConstantBaseline:
    def test_builds_uninformed_baseline(self) -> None:
        records = _fb_records({"acled": 2})
        baseline = constant_baseline(records, source="forecastbench")
        assert baseline.name == "uninformed_0.25"
        assert baseline.predictions == {
            "forecastbench:acled-000": 0.25,
            "forecastbench:acled-001": 0.25,
        }

    def test_skips_records_without_ids(self) -> None:
        baseline = constant_baseline(
            [{"id": "a-1"}, {"question": "no id"}], source="forecastbench", value=0.5
        )
        assert baseline.predictions == {"forecastbench:a-1": 0.5}


class TestResume:
    def test_interrupted_run_resumes_without_repaying(self, tmp_path) -> None:
        resume_file = tmp_path / "run.jsonl"
        calls = {"n": 0}

        def _crashing_fn(_text: str, _as_of: datetime, _metadata: Any = None) -> QuestionForecast:
            calls["n"] += 1
            if calls["n"] == 4:
                raise ConnectionError("machine slept")
            return QuestionForecast(accepted=True, raw_probability=0.6)

        adapter = _adapter(6)
        # First run: 3 succeed, 1 fails transiently, 2 more succeed.
        build_eval_context(
            adapter,
            _crashing_fn,
            harness=_harness(),
            calibration=_identity_calibration(),
            resume_path=resume_file,
        )
        assert calls["n"] == 6

        # Second run: only the failed question is re-forecast.
        rerun_calls = {"n": 0}

        def _fn(_text: str, _as_of: datetime, _metadata: Any = None) -> QuestionForecast:
            rerun_calls["n"] += 1
            return QuestionForecast(accepted=True, raw_probability=0.7)

        ctx = build_eval_context(
            adapter,
            _fn,
            harness=_harness(),
            calibration=_identity_calibration(),
            resume_path=resume_file,
        )
        assert rerun_calls["n"] == 1  # only the previously failed question
        assert len(ctx.inputs.records) == 6

    def test_refusals_are_persisted_and_not_retried(self, tmp_path) -> None:
        resume_file = tmp_path / "run.jsonl"
        adapter = _adapter(4)

        calls = {"n": 0}

        def _refusing_fn(_text: str, _as_of: datetime, _metadata: Any = None) -> QuestionForecast:
            calls["n"] += 1
            return QuestionForecast(accepted=False, raw_probability=None)

        with pytest.raises(ValueError, match="no accepted forecasts"):
            build_eval_context(adapter, _refusing_fn, harness=_harness(), resume_path=resume_file)
        assert calls["n"] == 4
        with pytest.raises(ValueError, match="no accepted forecasts"):
            build_eval_context(adapter, _refusing_fn, harness=_harness(), resume_path=resume_file)
        assert calls["n"] == 4  # nothing re-forecast on resume

    def test_traces_survive_resume(self, tmp_path) -> None:
        from core.forecast.leakage_judge import Trace, TraceComponent

        resume_file = tmp_path / "run.jsonl"
        trace = Trace(
            component=TraceComponent.SEARCH,
            as_of=datetime(2026, 1, 1, tzinfo=UTC),
            text="evidence snippet",
            forecast_id="f1",
        )

        def _fn(_text: str, _as_of: datetime, _metadata: Any = None) -> QuestionForecast:
            return QuestionForecast(accepted=True, raw_probability=0.6, traces=(trace,))

        adapter = _adapter(4)
        build_eval_context(
            adapter,
            _fn,
            harness=_harness(),
            calibration=_identity_calibration(),
            resume_path=resume_file,
        )
        ctx = build_eval_context(
            adapter,
            lambda _t, _a, _m=None: (_ for _ in ()).throw(AssertionError("must not be called")),
            harness=_harness(),
            calibration=_identity_calibration(),
            resume_path=resume_file,
        )
        assert len(ctx.inputs.traces) == 4
        assert ctx.inputs.traces[0].text == "evidence snippet"

    def test_different_run_key_refuses_to_resume(self, tmp_path) -> None:
        resume_file = tmp_path / "run.jsonl"
        build_eval_context(
            _adapter(4),
            _forecast_fn(),
            harness=_harness(),
            calibration=_identity_calibration(),
            resume_path=resume_file,
        )
        # Different question set (6 vs 4) -> different run key -> hard refusal.
        with pytest.raises(ValueError, match="different run"):
            build_eval_context(
                _adapter(6),
                _forecast_fn(),
                harness=_harness(),
                calibration=_identity_calibration(),
                resume_path=resume_file,
            )

    def test_different_tag_refuses_to_resume(self, tmp_path) -> None:
        resume_file = tmp_path / "run.jsonl"
        build_eval_context(
            _adapter(4),
            _forecast_fn(),
            harness=_harness(),
            calibration=_identity_calibration(),
            resume_path=resume_file,
            resume_tag="no-search",
        )
        with pytest.raises(ValueError, match="different run"):
            build_eval_context(
                _adapter(4),
                _forecast_fn(),
                harness=_harness(),
                calibration=_identity_calibration(),
                resume_path=resume_file,
                resume_tag="with-search",
            )

    def test_torn_tail_line_is_dropped_and_retried(self, tmp_path) -> None:
        resume_file = tmp_path / "run.jsonl"
        adapter = _adapter(4)
        build_eval_context(
            adapter,
            _forecast_fn(),
            harness=_harness(),
            calibration=_identity_calibration(),
            resume_path=resume_file,
        )
        # Simulate an interruption mid-write: truncate the last line.
        content = resume_file.read_text(encoding="utf-8")
        resume_file.write_text(content[:-20], encoding="utf-8")
        calls = {"n": 0}

        def _fn(_text: str, _as_of: datetime, _metadata: Any = None) -> QuestionForecast:
            calls["n"] += 1
            return QuestionForecast(accepted=True, raw_probability=0.5)

        ctx = build_eval_context(
            adapter,
            _fn,
            harness=_harness(),
            calibration=_identity_calibration(),
            resume_path=resume_file,
        )
        assert calls["n"] == 1  # the torn entry is re-forecast
        assert len(ctx.inputs.records) == 4

    def test_corrupt_interior_line_raises(self, tmp_path) -> None:
        from benchmarks.suites import ResumeStore, run_key_for

        resume_file = tmp_path / "run.jsonl"
        key = run_key_for(["a", "b"])
        ResumeStore(resume_file, run_key=key)
        lines = resume_file.read_text(encoding="utf-8")
        resume_file.write_text(
            lines
            + "not json\n"
            + '{"kind": "forecast", "question_id": "a", "accepted": true,'
            + ' "raw_probability": 0.5}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="corrupt at line"):
            ResumeStore(resume_file, run_key=key)

    def test_corrupt_header_raises(self, tmp_path) -> None:
        from benchmarks.suites import ResumeStore

        resume_file = tmp_path / "run.jsonl"
        resume_file.write_text("garbage\n", encoding="utf-8")
        with pytest.raises(ValueError, match="corrupt header"):
            ResumeStore(resume_file, run_key="k")

    def test_run_key_is_order_insensitive_but_tag_sensitive(self) -> None:
        from benchmarks.suites import run_key_for

        assert run_key_for(["a", "b"]) == run_key_for(["b", "a"])
        assert run_key_for(["a", "b"], tag="x") != run_key_for(["a", "b"], tag="y")
        assert run_key_for(["a", "b"]) != run_key_for(["a", "c"])


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

    def forecast(self, _text: str, *, as_of: datetime, metadata: Any = None) -> _FakeResult:
        self.seen_metadata = metadata
        return _FakeResult(accepted=self._accepted, raw=self._raw)


class TestForecasterFn:
    def test_maps_accepted_result_with_traces(self) -> None:
        fn = forecaster_fn(_FakeForecaster(raw=0.7))  # type: ignore[arg-type]
        out = fn("q", datetime(2026, 1, 1, tzinfo=UTC))
        assert out.accepted
        assert out.raw_probability == 0.7
        # One search trace (evidence) + one supervisor trace (rationale).
        assert len(out.traces) == 2

    def test_threads_metadata_to_forecaster(self) -> None:
        forecaster = _FakeForecaster(raw=0.7)
        fn = forecaster_fn(forecaster)  # type: ignore[arg-type]
        fn("q", datetime(2026, 1, 1, tzinfo=UTC), {"benchmark_question_id": "m:1"})
        assert forecaster.seen_metadata == {"benchmark_question_id": "m:1"}

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
            def forecast(self, _text: str, *, as_of: datetime, metadata: Any = None) -> _Result:
                return _Result()

        fn = forecaster_fn(_Forecaster())  # type: ignore[arg-type]
        out = fn("q", datetime(2026, 1, 1, tzinfo=UTC))
        assert out.accepted
        assert out.traces == ()


