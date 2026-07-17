"""Unit tests for the delphi CLI (intake + forecast wired; others stubbed)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from common.cli import _evidence_provider_names, build_parser, main
from core.forecast.leakage_judge import FixtureLeakageJudgeLLM, LeakageJudge
from core.forecast.llm import FixtureForecastLLM
from core.forecast.search import Evidence, FixtureAsOfSearch
from core.forecast.supervisor import FixtureSupervisorLLM
from core.registry.store import InMemoryRegistryStore
from evaluation.report import EvalContext
from forecaster.chain import Forecaster
from intake.llm import FixtureStructuredLLM
from intake.service import IntakeService

_CLASSIFY_BINARY = {"question_type": "binary", "entities": ["X"]}
_NORMALIZE_OK = {
    "canonical_text": "Will X reach GA before 2025-01-01?",
    "domain": "tech",
    "resolution_criteria": "Resolves YES if GA before 2025-01-01.",
    "close_time": "2025-01-01T00:00:00+00:00",
    "resolvable": True,
}


class TestIntakeCommand:
    def test_accepted_prints_and_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        llm = FixtureStructuredLLM([_CLASSIFY_BINARY, _NORMALIZE_OK])
        store = InMemoryRegistryStore()
        code = main(["intake", "Will X ship?"], llm=llm, store=store)
        out = capsys.readouterr().out
        assert code == 0
        assert "ACCEPTED question_id=" in out
        assert "type: binary" in out
        assert "domain: tech" in out
        assert "close_time: 2025-01-01T00:00:00+00:00" in out

    def test_accepted_without_close_time(self, capsys: pytest.CaptureFixture[str]) -> None:
        normalize = {k: v for k, v in _NORMALIZE_OK.items() if k != "close_time"}
        llm = FixtureStructuredLLM([_CLASSIFY_BINARY, normalize])
        code = main(["intake", "Will X ship?"], llm=llm, store=InMemoryRegistryStore())
        out = capsys.readouterr().out
        assert code == 0
        assert "close_time:" not in out

    def test_refused_prints_and_returns_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        llm = FixtureStructuredLLM([{"question_type": "unknown"}])
        code = main(["intake", "Mmm?"], llm=llm, store=InMemoryRegistryStore())
        out = capsys.readouterr().out
        assert code == 1
        assert "REFUSED reason=unknown_type" in out
        assert "detail:" in out

    def test_as_of_triggers_already_resolved(self, capsys: pytest.CaptureFixture[str]) -> None:
        normalize_past = {**_NORMALIZE_OK, "close_time": "2024-01-01T00:00:00+00:00"}
        llm = FixtureStructuredLLM([_CLASSIFY_BINARY, normalize_past])
        code = main(
            ["intake", "Did X ship?", "--as-of", "2024-06-01T00:00:00"],
            llm=llm,
            store=InMemoryRegistryStore(),
        )
        assert code == 1
        assert "already_resolved" in capsys.readouterr().out


_FORECAST_NORMALIZE = {
    "canonical_text": "Will X ship by 2025?",
    "domain": "tech",
    "resolution_criteria": "Resolves YES on GA announcement.",
    "close_time": "2025-06-01T00:00:00+00:00",
    "resolvable": True,
}


def _forecaster(
    store: InMemoryRegistryStore,
    *,
    classify: dict = _CLASSIFY_BINARY,
    normalize: dict = _FORECAST_NORMALIZE,
) -> Forecaster:
    intake = IntakeService(llm=FixtureStructuredLLM([classify, normalize]), store=store)
    return Forecaster(
        intake=intake,
        searcher=FixtureAsOfSearch(
            default=(
                Evidence(
                    snippet="signal",
                    source="hosted",
                    source_id="http://a",
                    knowledge_time=datetime(2024, 1, 1, tzinfo=UTC),
                    score=0.5,
                ),
            )
        ),
        reasoning_llm=FixtureStructuredLLM(
            [{"reference_class": "rc", "base_rate": 0.4}, {"rule": "none"}]
        ),
        forecast_llm=FixtureForecastLLM(default_response=0.6),
        supervisor_llm=FixtureSupervisorLLM(),
        leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM()),
        registry_store=store,
    )


class TestForecastCommand:
    def test_accepted_prints_probability(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            ["forecast", "Will X ship?", "--as-of", "2024-06-01T00:00:00"],
            forecaster=_forecaster(InMemoryRegistryStore()),
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "FORECAST question_id=" in out
        assert "probability:" in out
        assert "band: [" in out
        assert "rationale:" in out

    def test_refused_returns_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            ["forecast", "Mmm?", "--as-of", "2024-06-01T00:00:00"],
            forecaster=_forecaster(InMemoryRegistryStore(), classify={"question_type": "unknown"}),
        )
        assert code == 1
        assert "REFUSED reason=" in capsys.readouterr().out

    def test_as_of_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["forecast", "Will X ship?"])


class TestResolveCommand:
    def test_resolves_and_reports(self, capsys: pytest.CaptureFixture[str]) -> None:
        from core.registry.models import QuestionInput
        from resolution.service import ResolutionService
        from resolution.sources import MappingResolutionSource, ResolvedOutcome

        store = InMemoryRegistryStore()
        qid = store.record_question(
            QuestionInput(
                text="Will X win?",
                question_type="binary",
                domain="politics",
                resolution_criteria="Official result.",
            )
        )
        service = ResolutionService(
            store=store,
            source=MappingResolutionSource(
                {
                    qid: ResolvedOutcome(
                        resolved_value=1.0,
                        resolved_at=datetime(2025, 1, 1, tzinfo=UTC),
                        source="gov",
                    )
                }
            ),
        )
        code = main(["resolve", "--since", "2024-01-01T00:00:00"], resolution_service=service)
        out = capsys.readouterr().out
        assert code == 0
        assert "RESOLVED 1 question(s)" in out
        assert "resolution_id:" in out

    def test_answers_file_drives_resolution(self, capsys: pytest.CaptureFixture[str]) -> None:
        import json

        from core.registry.models import QuestionInput
        from resolution.service import ResolutionService
        from resolution.sources import load_mapping_source

        store = InMemoryRegistryStore()
        qid = store.record_question(
            QuestionInput(
                text="Will X win?",
                question_type="binary",
                domain="politics",
                resolution_criteria="Official result.",
            )
        )
        answers = {qid: {"value": 1.0, "resolved_at": "2025-01-01T00:00:00Z"}}
        path = _tmp_answers(json.dumps(answers))
        service = ResolutionService(store=store, source=load_mapping_source(path))
        code = main(["resolve", "--answers", str(path)], resolution_service=service)
        out = capsys.readouterr().out
        assert code == 0
        assert "RESOLVED 1 question(s)" in out


def _tmp_answers(contents: str):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "answers.json"
    path.write_text(contents, encoding="utf-8")
    return path


def _eval_context() -> EvalContext:
    from core.forecast.leakage_judge import Trace, TraceComponent
    from core.orchestration.budget import InMemoryBudgetLedger
    from evaluation.harness import EvalHarness
    from evaluation.report import EvalContext, EvalInputs
    from evaluation.scoring import ScoredRecord

    records = (
        ScoredRecord(question_id="q1", domain="econ", probability=0.8, outcome=1.0),
        ScoredRecord(question_id="q2", domain="geo", probability=0.2, outcome=0.0),
    )
    traces = (
        Trace(
            component=TraceComponent.SEARCH,
            as_of=datetime(2026, 1, 1, tzinfo=UTC),
            text="clean",
        ),
    )
    harness = EvalHarness(budget_ledger=InMemoryBudgetLedger(cap=100, trials_count=lambda: 0))
    return EvalContext(
        inputs=EvalInputs(records=records, traces=traces),
        harness=harness,
        judge=LeakageJudge(FixtureLeakageJudgeLLM()),
    )


class TestEvalCommand:
    def test_scores_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["eval"], eval_context=_eval_context())
        out = capsys.readouterr().out
        assert code == 0
        assert "Proper scores" in out
        assert "ECE=" in out

    def test_scored_report_carries_leakage_audit_by_default(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Leakage-first (§2.6): the default scored report must include the
        # audit — it is never a separate opt-in.
        code = main(["eval"], eval_context=_eval_context())
        out = capsys.readouterr().out
        assert code == 0
        assert "## Leakage audit (§2.6)" in out
        assert "leakage_rate" in out
        assert "## Trials ledger (§2.4)" in out

    def test_leakage_audit(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["eval", "--leakage-audit"], eval_context=_eval_context())
        out = capsys.readouterr().out
        assert code == 0
        assert "leakage_rate" in out

    def test_dump_forecasts_writes_scored_probabilities(
        self, capsys: pytest.CaptureFixture[str], tmp_path
    ) -> None:
        import json

        target = tmp_path / "run.json"
        code = main(
            ["eval", "--dump-forecasts", str(target)],
            eval_context=_eval_context(),
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "Dumped" in out
        dumped = json.loads(target.read_text(encoding="utf-8"))
        assert dumped and all(0.0 <= p <= 1.0 for p in dumped.values())

    def test_extra_baseline_appears_in_report(
        self, capsys: pytest.CaptureFixture[str], tmp_path
    ) -> None:
        import json

        ctx = _eval_context()
        baseline_file = tmp_path / "nosearch.json"
        predictions = {r.question_id: 0.5 for r in ctx.inputs.records}
        baseline_file.write_text(json.dumps(predictions), encoding="utf-8")
        code = main(
            ["eval", "--extra-baseline", f"no_search={baseline_file}"],
            eval_context=ctx,
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "vs no_search:" in out

    def test_malformed_extra_baseline_fails_loudly(
        self, capsys: pytest.CaptureFixture[str], tmp_path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("[1, 2, 3]", encoding="utf-8")
        code = main(
            ["eval", "--extra-baseline", f"x={bad}"],
            eval_context=_eval_context(),
        )
        assert code == 1
        assert "Cannot load extra baseline" in capsys.readouterr().out

    def test_extra_baseline_without_equals_fails_loudly(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            ["eval", "--extra-baseline", "just-a-path.json"],
            eval_context=_eval_context(),
        )
        assert code == 1
        assert "NAME=PATH" in capsys.readouterr().out

    def test_missing_extra_baseline_file_fails_loudly(
        self, capsys: pytest.CaptureFixture[str], tmp_path
    ) -> None:
        code = main(
            ["eval", "--extra-baseline", f"x={tmp_path / 'nope.json'}"],
            eval_context=_eval_context(),
        )
        assert code == 1
        assert "Cannot load extra baseline" in capsys.readouterr().out

    def test_ablation_flags_parse(self) -> None:
        args = build_parser().parse_args(
            [
                "eval",
                "--suite",
                "forecastbench",
                "--no-search",
                "--sample",
                "50",
                "--sample-seed",
                "7",
            ]
        )
        assert args.no_search is True
        assert args.sample == 50
        assert args.sample_seed == 7

    def test_ablation_flags_default_off(self) -> None:
        args = build_parser().parse_args(["eval"])
        assert args.no_search is False
        assert args.sample is None
        assert args.dump_forecasts is None
        assert args.extra_baselines == []

    def test_leakage_audit_without_judge(self, capsys: pytest.CaptureFixture[str]) -> None:
        from core.orchestration.budget import InMemoryBudgetLedger
        from evaluation.harness import EvalHarness
        from evaluation.report import EvalContext, EvalInputs
        from evaluation.scoring import ScoredRecord

        ctx = EvalContext(
            inputs=EvalInputs(
                records=(ScoredRecord(question_id="q1", domain="d", probability=0.5, outcome=1.0),)
            ),
            harness=EvalHarness(budget_ledger=InMemoryBudgetLedger(cap=10, trials_count=lambda: 0)),
            judge=None,
        )
        code = main(["eval", "--leakage-audit"], eval_context=ctx)
        assert code == 1
        assert "No leakage judge" in capsys.readouterr().out


class TestConductorCommand:
    def test_accepted_prints_route(self, capsys: pytest.CaptureFixture[str]) -> None:
        from conductor.heuristic import HeuristicConductor

        conductor = HeuristicConductor(forecaster=_forecaster(InMemoryRegistryStore()))
        code = main(
            ["conductor", "Will X ship?", "--as-of", "2024-06-01T00:00:00"],
            conductor=conductor,
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "CONDUCTOR question_id=" in out
        assert "route:" in out
        assert "red-team:" in out

    def test_refused_returns_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        from conductor.heuristic import HeuristicConductor

        conductor = HeuristicConductor(
            forecaster=_forecaster(InMemoryRegistryStore(), classify={"question_type": "unknown"})
        )
        code = main(["conductor", "Mmm?", "--as-of", "2024-06-01T00:00:00"], conductor=conductor)
        assert code == 1
        assert "REFUSED reason=" in capsys.readouterr().out


def _live_context(store: InMemoryRegistryStore):
    from benchmarks.live import LiveHarvestAdapter
    from benchmarks.live_loop.harvest import HarvestJob
    from benchmarks.live_loop.score import ScoreJob
    from common.cli import LiveContext
    from conductor.heuristic import HeuristicConductor
    from core.orchestration.run_state import InMemoryRunStateStore
    from resolution.service import ResolutionService
    from resolution.sources import MappingResolutionSource

    open_normalize = {k: v for k, v in _FORECAST_NORMALIZE.items() if k != "close_time"}
    conductor = HeuristicConductor(forecaster=_forecaster(store, normalize=open_normalize))
    adapter = LiveHarvestAdapter.harvest(
        [{"id": "a", "question": "Will X ship?"}],
        harvest_time=datetime(2026, 7, 1, tzinfo=UTC),
    )
    service = ResolutionService(store=store, source=MappingResolutionSource({}))
    return LiveContext(
        harvest_job=HarvestJob(conductor=conductor),
        score_job=ScoreJob(store=store, resolution_service=service),
        adapter=adapter,
        run_state=InMemoryRunStateStore(),
    )


class TestBenchLiveCommand:
    def test_harvest_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        ctx = _live_context(InMemoryRegistryStore())
        code = main(
            ["bench", "live", "--harvest", "--tick", "2026-07-01T00:00:00"], live_context=ctx
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "HARVEST pending=1" in out

    def test_score_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        ctx = _live_context(InMemoryRegistryStore())
        code = main(["bench", "live", "--score", "--tick", "2026-07-01T00:00:00"], live_context=ctx)
        out = capsys.readouterr().out
        assert code == 0
        assert "SCORE resolved=" in out

    def test_repeat_tick_is_skipped(self, capsys: pytest.CaptureFixture[str]) -> None:
        ctx = _live_context(InMemoryRegistryStore())
        main(["bench", "live", "--harvest", "--tick", "2026-07-01T00:00:00"], live_context=ctx)
        capsys.readouterr()
        code = main(
            ["bench", "live", "--harvest", "--tick", "2026-07-01T00:00:00"], live_context=ctx
        )
        assert code == 0
        assert "SKIPPED" in capsys.readouterr().out


class TestServeCommand:
    def test_check_runs_health_roundtrip(self, capsys: pytest.CaptureFixture[str]) -> None:
        from api.routes import ForecastService
        from api.server import DelphiApp
        from conductor.heuristic import HeuristicConductor

        store = InMemoryRegistryStore()
        forecaster = _forecaster(store)
        service = ForecastService(
            forecaster=forecaster,
            conductor=HeuristicConductor(forecaster=forecaster),
            store=store,
            intake=IntakeService(llm=FixtureStructuredLLM(), store=store),
        )
        code = main(["serve", "--check", "--port", "9001"], api_app=DelphiApp(service))
        out = capsys.readouterr().out
        assert code == 0
        assert "DELPHI API health=200" in out
        assert "port=9001" in out


class TestDoctorCommand:
    def test_all_pass_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        checks = [
            ("postgres", lambda: "connected"),
            ("bedrock", lambda: "reachable"),
        ]
        code = main(["doctor"], doctor_checks=checks)
        out = capsys.readouterr().out
        assert code == 0
        assert "[PASS] postgres: connected" in out
        assert "DOCTOR ok" in out

    def test_any_failure_returns_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        def boom() -> str:
            raise RuntimeError("DELPHI_PG_DSN not set")

        checks = [("postgres", boom), ("bedrock", lambda: "reachable")]
        code = main(["doctor"], doctor_checks=checks)
        out = capsys.readouterr().out
        assert code == 1
        assert "[FAIL] postgres: RuntimeError: DELPHI_PG_DSN not set" in out
        assert "DOCTOR failed" in out


class TestCalibrationFitCommand:
    @staticmethod
    def _store_with_resolved_forecast() -> tuple[InMemoryRegistryStore, str]:
        from core.registry.models import ResolutionInput

        store = InMemoryRegistryStore()
        result = _forecaster(store).forecast("Will X ship?", as_of=datetime(2024, 6, 1, tzinfo=UTC))
        assert result.question_id is not None
        store.record_resolution(
            ResolutionInput(
                question_id=result.question_id,
                resolved_value=1.0,
                resolved_at=datetime(2025, 7, 1, tzinfo=UTC),
                source="test",
            )
        )
        return store, result.question_id

    @staticmethod
    def _clock() -> datetime:
        return datetime(2026, 7, 15, 12, 0, tzinfo=UTC)

    def test_fits_and_writes_versioned_artifact(
        self, capsys: pytest.CaptureFixture[str], tmp_path
    ) -> None:
        from evaluation.calibration_split import question_fingerprint
        from forecaster.calibration_artifact import load_calibration_artifact

        store, question_id = self._store_with_resolved_forecast()
        code = main(
            ["calibration", "fit", "--output", str(tmp_path)],
            store=store,
            clock=self._clock,
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "Fitted platt recalibrator on 1 resolved question(s)" in out
        # One fit point is below the trust threshold: labeled identity fallback.
        assert "IDENTITY FALLBACK" in out
        artifacts = list(tmp_path.glob("calibration-20260715-*.json"))
        assert len(artifacts) == 1
        learned = load_calibration_artifact(artifacts[0])
        assert learned.fitted_meta["fitted_at"] == "2026-07-15T12:00:00+00:00"
        assert learned.fitted_meta["n"] == 1
        assert learned.fitted_meta["fallback"] is True
        # §2.5 disjointness fingerprints: registry id + normalized question text.
        fingerprints = learned.fitted_meta["question_fingerprints"]
        assert question_fingerprint(question_id) in fingerprints
        question = store.get_question(question_id)
        assert question_fingerprint(question.text) in fingerprints

    def test_fingerprints_include_resolution_label(
        self, capsys: pytest.CaptureFixture[str], tmp_path
    ) -> None:
        # A backfilled resolution carries the benchmark id in resolved_label
        # (the question predates benchmark-id metadata); the artifact must
        # fingerprint it or a later artifact-mode eval could re-score the fit
        # question undetected (§2.5).
        from core.registry.models import ResolutionInput
        from evaluation.calibration_split import question_fingerprint
        from forecaster.calibration_artifact import load_calibration_artifact

        store = InMemoryRegistryStore()
        result = _forecaster(store).forecast("Will X ship?", as_of=datetime(2024, 6, 1, tzinfo=UTC))
        assert result.question_id is not None
        store.record_resolution(
            ResolutionInput(
                question_id=result.question_id,
                resolved_value=1.0,
                resolved_at=datetime(2025, 7, 1, tzinfo=UTC),
                source="forecastbench",
                resolved_label="forecastbench:acled-abc123",
            )
        )
        code = main(
            ["calibration", "fit", "--output", str(tmp_path)], store=store, clock=self._clock
        )
        assert code == 0
        artifact = load_calibration_artifact(next(iter(tmp_path.glob("*.json"))))
        fingerprints = artifact.fitted_meta["question_fingerprints"]
        assert question_fingerprint("forecastbench:acled-abc123") in fingerprints

    def test_explicit_output_file_path(self, capsys: pytest.CaptureFixture[str], tmp_path) -> None:
        store, _ = self._store_with_resolved_forecast()
        target = tmp_path / "my-artifact.json"
        code = main(
            ["calibration", "fit", "--output", str(target)],
            store=store,
            clock=self._clock,
        )
        assert code == 0
        assert target.exists()
        assert str(target) in capsys.readouterr().out

    def test_empty_registry_returns_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["calibration", "fit"], store=InMemoryRegistryStore(), clock=self._clock)
        assert code == 1
        assert "nothing to fit" in capsys.readouterr().out

    def test_holdout_ids_are_excluded(self, capsys: pytest.CaptureFixture[str], tmp_path) -> None:
        import json as jsonlib

        store, question_id = self._store_with_resolved_forecast()
        holdout = tmp_path / "holdout.json"
        holdout.write_text(jsonlib.dumps([question_id]), encoding="utf-8")
        code = main(
            ["calibration", "fit", "--output", str(tmp_path), "--holdout-file", str(holdout)],
            store=store,
            clock=self._clock,
        )
        assert code == 1
        assert "nothing to fit" in capsys.readouterr().out

    def test_invalid_holdout_file_returns_one(
        self, capsys: pytest.CaptureFixture[str], tmp_path
    ) -> None:
        holdout = tmp_path / "holdout.json"
        holdout.write_text('{"not": "a list"}', encoding="utf-8")
        code = main(
            ["calibration", "fit", "--holdout-file", str(holdout)],
            store=InMemoryRegistryStore(),
            clock=self._clock,
        )
        assert code == 1
        assert "Cannot read holdout file" in capsys.readouterr().out

    def test_missing_holdout_file_returns_one(
        self, capsys: pytest.CaptureFixture[str], tmp_path
    ) -> None:
        code = main(
            ["calibration", "fit", "--holdout-file", str(tmp_path / "nope.json")],
            store=InMemoryRegistryStore(),
            clock=self._clock,
        )
        assert code == 1
        assert "Cannot read holdout file" in capsys.readouterr().out

    def test_resolved_question_without_forecast_is_skipped(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from core.registry.models import QuestionInput, ResolutionInput

        store = InMemoryRegistryStore()
        qid = store.record_question(
            QuestionInput(
                text="Will Y happen?",
                question_type="binary",
                domain="tech",
                resolution_criteria="Official announcement.",
            )
        )
        store.record_resolution(
            ResolutionInput(
                question_id=qid,
                resolved_value=0.0,
                resolved_at=datetime(2025, 7, 1, tzinfo=UTC),
                source="test",
            )
        )
        code = main(["calibration", "fit"], store=store, clock=self._clock)
        assert code == 1
        assert "nothing to fit" in capsys.readouterr().out

    def test_forecast_without_probability_is_skipped(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from core.registry.models import (
            ForecastInput,
            Quantile,
            QuestionInput,
            ResolutionInput,
        )

        store = InMemoryRegistryStore()
        qid = store.record_question(
            QuestionInput(
                text="How many units by 2026?",
                question_type="binary",
                domain="tech",
                resolution_criteria="Official count.",
            )
        )
        store.record_forecast(
            ForecastInput(
                question_id=qid,
                as_of=datetime(2024, 6, 1, tzinfo=UTC),
                quantiles=(Quantile(level=0.5, value=10.0),),
                rationale="distributional",
                model_provenance={"forecast_llm": {"model_version": "m"}},
                repro_handle={"as_of": "2024-06-01T00:00:00+00:00"},
            )
        )
        store.record_resolution(
            ResolutionInput(
                question_id=qid,
                resolved_value=1.0,
                resolved_at=datetime(2025, 7, 1, tzinfo=UTC),
                source="test",
            )
        )
        code = main(["calibration", "fit"], store=store, clock=self._clock)
        assert code == 1
        assert "nothing to fit" in capsys.readouterr().out

    def test_unresolved_forecasts_are_skipped(self, capsys: pytest.CaptureFixture[str]) -> None:
        store = InMemoryRegistryStore()
        _forecaster(store).forecast("Will X ship?", as_of=datetime(2024, 6, 1, tzinfo=UTC))
        code = main(["calibration", "fit"], store=store, clock=self._clock)
        assert code == 1
        assert "nothing to fit" in capsys.readouterr().out

    def test_default_clock_is_used_when_not_injected(self, tmp_path) -> None:
        store, _ = self._store_with_resolved_forecast()
        code = main(["calibration", "fit", "--output", str(tmp_path)], store=store)
        assert code == 0
        assert list(tmp_path.glob("calibration-*.json"))

    def test_fit_subcommand_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["calibration"])

    def test_benchmark_metadata_id_is_fingerprinted(self, tmp_path) -> None:
        from core.registry.models import ResolutionInput
        from evaluation.calibration_split import question_fingerprint
        from forecaster.calibration_artifact import load_calibration_artifact
        from resolution.benchmark_source import BENCHMARK_QUESTION_ID_KEY

        store = InMemoryRegistryStore()
        # Thread the benchmark origin through intake metadata, as the live loop does.
        result = _forecaster(store).forecast(
            "Will X ship?",
            as_of=datetime(2024, 6, 1, tzinfo=UTC),
            metadata={BENCHMARK_QUESTION_ID_KEY: "forecastbench:abc"},
        )
        assert result.question_id is not None
        store.record_resolution(
            ResolutionInput(
                question_id=result.question_id,
                resolved_value=1.0,
                resolved_at=datetime(2025, 7, 1, tzinfo=UTC),
                source="test",
            )
        )
        code = main(
            ["calibration", "fit", "--output", str(tmp_path)], store=store, clock=self._clock
        )
        assert code == 0
        learned = load_calibration_artifact(next(iter(tmp_path.glob("*.json"))))
        assert (
            question_fingerprint("forecastbench:abc")
            in learned.fitted_meta["question_fingerprints"]
        )


class TestEvalCalibrationArtifactFlag:
    def test_flag_parses(self) -> None:
        args = build_parser().parse_args(
            ["eval", "--suite", "forecastbench", "--calibration-artifact", "/tmp/a.json"]
        )
        assert args.calibration_artifact == "/tmp/a.json"

    def test_flag_defaults_to_none(self) -> None:
        args = build_parser().parse_args(["eval"])
        assert args.calibration_artifact is None


class TestEvidenceProviderNames:
    def test_default_is_all_providers(self) -> None:
        assert _evidence_provider_names({}) == ("tavily", "gdelt", "wikipedia")

    def test_env_selects_subset(self) -> None:
        env = {"DELPHI_EVIDENCE_PROVIDERS": "gdelt,wikipedia"}
        assert _evidence_provider_names(env) == ("gdelt", "wikipedia")

    def test_whitespace_and_case_tolerated(self) -> None:
        env = {"DELPHI_EVIDENCE_PROVIDERS": " Tavily , GDELT "}
        assert _evidence_provider_names(env) == ("tavily", "gdelt")

    def test_duplicates_collapse_order_preserving(self) -> None:
        env = {"DELPHI_EVIDENCE_PROVIDERS": "wikipedia,tavily,wikipedia"}
        assert _evidence_provider_names(env) == ("wikipedia", "tavily")

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown evidence provider"):
            _evidence_provider_names({"DELPHI_EVIDENCE_PROVIDERS": "bing"})

    def test_only_commas_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one provider"):
            _evidence_provider_names({"DELPHI_EVIDENCE_PROVIDERS": " , ,"})

    def test_none_alone_is_valid(self) -> None:
        assert _evidence_provider_names({"DELPHI_EVIDENCE_PROVIDERS": "none"}) == ("none",)

    def test_none_combined_with_providers_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be combined"):
            _evidence_provider_names({"DELPHI_EVIDENCE_PROVIDERS": "none,tavily"})


class TestParser:
    def test_missing_command_errors(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestResolveSourceSelection:
    def test_no_source_prints_hint_and_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["resolve"])
        out = capsys.readouterr().out
        assert code == 2
        assert "Nothing to resolve from" in out
        assert "--answers" in out
        assert "--suite" in out

    def test_answers_and_suite_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["resolve", "--answers", "x.json", "--suite", "metaculus"])

    def test_unknown_suite_rejected_by_parser(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["resolve", "--suite", "not-a-suite"])

    def test_suite_flag_reaches_injected_service(self, capsys: pytest.CaptureFixture[str]) -> None:
        from resolution.service import ResolutionService
        from resolution.sources import MappingResolutionSource

        service = ResolutionService(
            store=InMemoryRegistryStore(), source=MappingResolutionSource({})
        )
        code = main(["resolve", "--suite", "metaculus"], resolution_service=service)
        out = capsys.readouterr().out
        assert code == 0
        assert "RESOLVED 0 question(s)" in out


class TestHoldoutGovernorFromEnv:
    def test_unconfigured_returns_none(self) -> None:
        from common.cli import _holdout_governor_from_env

        assert _holdout_governor_from_env(env={}) is None

    def test_partial_config_returns_none(self, tmp_path) -> None:
        from common.cli import _holdout_governor_from_env

        payload = tmp_path / "holdout.json"
        payload.write_text('{"brier": 0.12}', encoding="utf-8")
        assert _holdout_governor_from_env(env={"DELPHI_HOLDOUT_FILE": str(payload)}) is None
        assert _holdout_governor_from_env(env={"DELPHI_HOLDOUT_BUDGET": "3"}) is None

    def test_configured_builds_budgeted_logged_governor(self, tmp_path) -> None:
        from common.cli import _holdout_governor_from_env

        payload = tmp_path / "holdout.json"
        payload.write_text('{"brier": 0.12}', encoding="utf-8")
        governor = _holdout_governor_from_env(
            env={"DELPHI_HOLDOUT_FILE": str(payload), "DELPHI_HOLDOUT_BUDGET": "2"}
        )
        assert governor is not None
        assert governor.remaining_budget() == 2
        view = governor.access_holdout(reason="wiring test")
        assert view.payload == {"brier": 0.12}
        assert governor.remaining_budget() == 1
        assert governor.verify_chain().ok

    def test_non_integer_budget_raises(self, tmp_path) -> None:
        from common.cli import _holdout_governor_from_env

        payload = tmp_path / "holdout.json"
        payload.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="must be an integer"):
            _holdout_governor_from_env(
                env={"DELPHI_HOLDOUT_FILE": str(payload), "DELPHI_HOLDOUT_BUDGET": "many"}
            )

    def test_non_object_payload_raises(self, tmp_path) -> None:
        from common.cli import _holdout_governor_from_env

        payload = tmp_path / "holdout.json"
        payload.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            _holdout_governor_from_env(
                env={"DELPHI_HOLDOUT_FILE": str(payload), "DELPHI_HOLDOUT_BUDGET": "1"}
            )


class TestEvalTrialsLedger:
    def test_no_dsn_refuses_by_default(self) -> None:
        from common.cli import _eval_trials_ledger

        with pytest.raises(RuntimeError, match="No durable trials ledger"):
            _eval_trials_ledger(pg_dsn=None, cap=100, env={})

    def test_opt_in_must_be_exactly_one(self) -> None:
        from common.cli import _eval_trials_ledger

        with pytest.raises(RuntimeError, match="No durable trials ledger"):
            _eval_trials_ledger(
                pg_dsn=None, cap=100, env={"DELPHI_ALLOW_EPHEMERAL_TRIALS_LEDGER": "true"}
            )

    def test_explicit_opt_in_returns_ephemeral_ledger_with_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from common.cli import _eval_trials_ledger
        from core.orchestration.budget import InMemoryBudgetLedger

        ledger = _eval_trials_ledger(
            pg_dsn=None, cap=100, env={"DELPHI_ALLOW_EPHEMERAL_TRIALS_LEDGER": "1"}
        )
        assert isinstance(ledger, InMemoryBudgetLedger)
        assert ledger.durable is False
        out = capsys.readouterr().out
        assert "EPHEMERAL" in out
        assert "§2.4" in out

    def test_dsn_builds_durable_postgres_ledger(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from common.cli import _eval_trials_ledger
        from core.orchestration.budget import PostgresBudgetLedger

        seen: dict[str, object] = {}

        def fake_connect(dsn: str, *, cap: int) -> str:
            seen["dsn"] = dsn
            seen["cap"] = cap
            return "durable-ledger"

        monkeypatch.setattr(PostgresBudgetLedger, "connect", fake_connect)
        ledger = _eval_trials_ledger(pg_dsn="postgresql://test", cap=42, env={})
        assert ledger == "durable-ledger"
        assert seen == {"dsn": "postgresql://test", "cap": 42}


class TestServeAuthWarning:
    def _service(self):
        from api.routes import ForecastService
        from conductor.heuristic import HeuristicConductor

        store = InMemoryRegistryStore()
        forecaster = _forecaster(store)
        # The /v1/classify + /v1/formalize surface needs its own intake seam.
        intake = IntakeService(
            llm=FixtureStructuredLLM([_CLASSIFY_BINARY, _FORECAST_NORMALIZE]), store=store
        )
        return ForecastService(
            forecaster=forecaster,
            conductor=HeuristicConductor(forecaster=forecaster),
            store=store,
            intake=intake,
        )

    def test_warns_when_endpoint_is_open(self, capsys: pytest.CaptureFixture[str]) -> None:
        from api.server import DelphiApp

        app = DelphiApp(self._service())
        assert app.auth_enabled is False
        code = main(["serve", "--check"], api_app=app)
        out = capsys.readouterr().out
        assert code == 0
        assert "UNAUTHENTICATED" in out

    def test_silent_when_token_configured(self, capsys: pytest.CaptureFixture[str]) -> None:
        from api.server import DelphiApp

        app = DelphiApp(self._service(), auth_token="secret-token")
        assert app.auth_enabled is True
        code = main(["serve", "--check"], api_app=app)
        out = capsys.readouterr().out
        assert code == 0
        assert "UNAUTHENTICATED" not in out


class TestEvalRecordLimits:
    def test_defaults_when_unset(self) -> None:
        from common.cli import _eval_record_limits

        assert _eval_record_limits(env={}) == (None, None, 5)

    def test_reads_all_three_knobs(self) -> None:
        from common.cli import _eval_record_limits

        resolved_after, max_questions, max_pages = _eval_record_limits(
            env={
                "DELPHI_EVAL_RESOLVED_AFTER": "2026-02-01",
                "DELPHI_EVAL_MAX_QUESTIONS": "250",
                "DELPHI_EVAL_MAX_PAGES": "10",
            }
        )
        assert resolved_after == "2026-02-01"
        assert max_questions == 250
        assert max_pages == 10

    def test_empty_strings_fall_back(self) -> None:
        from common.cli import _eval_record_limits

        assert _eval_record_limits(
            env={"DELPHI_EVAL_RESOLVED_AFTER": "", "DELPHI_EVAL_MAX_QUESTIONS": ""}
        ) == (None, None, 5)

    def test_non_integer_raises(self) -> None:
        from common.cli import _eval_record_limits

        with pytest.raises(ValueError, match="must be integers"):
            _eval_record_limits(env={"DELPHI_EVAL_MAX_QUESTIONS": "many"})


class TestFilterEvalRecords:
    _RECORDS = [
        {"id": "old", "resolved_at": "2025-06-01T00:00:00Z"},
        {"id": "fresh", "resolved_at": "2026-06-01T00:00:00Z"},
        {"id": "mid", "resolved_at": "2026-03-01T00:00:00Z"},
        {"id": "unresolved"},
    ]

    def test_no_knobs_is_identity(self) -> None:
        from common.cli import _filter_eval_records

        out = _filter_eval_records(self._RECORDS, resolved_after=None, max_questions=None)
        assert out == self._RECORDS

    def test_resolved_after_drops_old_and_unresolved(self) -> None:
        from common.cli import _filter_eval_records

        out = _filter_eval_records(self._RECORDS, resolved_after="2026-02-01", max_questions=None)
        assert [r["id"] for r in out] == ["fresh", "mid"]

    def test_cutoff_is_strict(self) -> None:
        from common.cli import _filter_eval_records

        out = _filter_eval_records(
            [{"id": "at", "resolved_at": "2026-02-01T00:00:00Z"}],
            resolved_after="2026-02-01",
            max_questions=None,
        )
        assert out == []

    def test_max_questions_keeps_freshest_first(self) -> None:
        from common.cli import _filter_eval_records

        out = _filter_eval_records(self._RECORDS, resolved_after=None, max_questions=2)
        assert [r["id"] for r in out] == ["fresh", "mid"]

    def test_unresolved_sorts_last_under_cap(self) -> None:
        from common.cli import _filter_eval_records

        out = _filter_eval_records(self._RECORDS, resolved_after=None, max_questions=4)
        assert [r["id"] for r in out] == ["fresh", "mid", "old", "unresolved"]

    def test_filter_then_cap_compose(self) -> None:
        from common.cli import _filter_eval_records

        out = _filter_eval_records(self._RECORDS, resolved_after="2026-01-01", max_questions=1)
        assert [r["id"] for r in out] == ["fresh"]

    def test_does_not_mutate_input(self) -> None:
        from common.cli import _filter_eval_records

        snapshot = [dict(r) for r in self._RECORDS]
        _filter_eval_records(self._RECORDS, resolved_after=None, max_questions=1)
        assert snapshot == self._RECORDS


class TestEvalWithLeakageAudit:
    def test_flags_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["eval", "--leakage-audit", "--with-leakage-audit"])

    def test_with_audit_renders_report_and_audit(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            ["eval", "--with-leakage-audit"],
            eval_context=_eval_context(),  # type: ignore[arg-type]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "Proper scores" in out
        assert "leakage_rate" in out

    def test_without_judge_scored_report_warns_not_run(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # §2.6: the scored report still renders, but must carry the explicit
        # NOT-RUN leakage warning instead of silently omitting the section.
        from core.orchestration.budget import InMemoryBudgetLedger
        from evaluation.harness import EvalHarness
        from evaluation.report import EvalContext, EvalInputs
        from evaluation.scoring import ScoredRecord

        ctx = EvalContext(
            inputs=EvalInputs(
                records=(ScoredRecord(question_id="q1", domain="d", probability=0.5, outcome=1.0),)
            ),
            harness=EvalHarness(budget_ledger=InMemoryBudgetLedger(cap=10, trials_count=lambda: 0)),
            judge=None,
        )
        code = main(["eval", "--with-leakage-audit"], eval_context=ctx)
        out = capsys.readouterr().out
        assert code == 0
        assert "NOT RUN (no leakage judge configured)" in out
