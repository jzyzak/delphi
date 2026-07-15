"""Unit tests for the delphi CLI (intake + forecast wired; others stubbed)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from common.cli import build_parser, main
from core.forecast.leakage_judge import FixtureLeakageJudgeLLM, LeakageJudge
from core.forecast.llm import FixtureForecastLLM
from core.forecast.search import Evidence, FixtureAsOfSearch
from core.forecast.supervisor import FixtureSupervisorLLM
from core.registry.store import InMemoryRegistryStore
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


def _eval_context() -> object:
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
        code = main(["eval"], eval_context=_eval_context())  # type: ignore[arg-type]
        out = capsys.readouterr().out
        assert code == 0
        assert "Proper scores" in out
        assert "ECE=" in out

    def test_leakage_audit(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["eval", "--leakage-audit"], eval_context=_eval_context())  # type: ignore[arg-type]
        out = capsys.readouterr().out
        assert code == 0
        assert "leakage_rate" in out

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


class TestServeAuthWarning:
    def _service(self):
        from api.routes import ForecastService
        from conductor.heuristic import HeuristicConductor

        store = InMemoryRegistryStore()
        forecaster = _forecaster(store)
        return ForecastService(
            forecaster=forecaster,
            conductor=HeuristicConductor(forecaster=forecaster),
            store=store,
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

    def test_with_audit_requires_judge(self, capsys: pytest.CaptureFixture[str]) -> None:
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
        assert code == 1
        assert "No leakage judge" in out
