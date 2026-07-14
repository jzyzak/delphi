"""Tests for the live benchmark loop (Phase 9)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmarks.live import LiveHarvestAdapter
from benchmarks.live_loop import ClaimOutcome, claim_and_run, live_cadence
from benchmarks.live_loop.harvest import HarvestJob
from benchmarks.live_loop.score import LiveMetrics, ScoreJob, collect_scored_records
from conductor.heuristic import HeuristicConductor
from core.forecast.leakage_judge import FixtureLeakageJudgeLLM, LeakageJudge
from core.forecast.llm import FixtureForecastLLM
from core.forecast.search import Evidence, FixtureAsOfSearch
from core.forecast.supervisor import FixtureSupervisorLLM
from core.orchestration.run_state import InMemoryRunStateStore
from core.registry.models import ForecastInput, QuestionInput, ResolutionInput
from core.registry.store import InMemoryRegistryStore
from intake.llm import FixtureStructuredLLM
from intake.service import IntakeService
from resolution.service import ResolutionService
from resolution.sources import MappingResolutionSource, ResolvedOutcome

_HARVEST_TIME = datetime(2026, 7, 1, tzinfo=UTC)

_CLASSIFY = {"question_type": "binary", "entities": ["X"]}
_NORMALIZE = {
    "canonical_text": "Will X ship?",
    "domain": "tech",
    "resolution_criteria": "Resolves YES on GA.",
    "resolvable": True,
}


def _conductor(store: InMemoryRegistryStore) -> HeuristicConductor:
    forecaster_llms = FixtureStructuredLLM(
        [_CLASSIFY, _NORMALIZE, _CLASSIFY, _NORMALIZE, _CLASSIFY, _NORMALIZE]
    )
    from forecaster.chain import Forecaster

    forecaster = Forecaster(
        intake=IntakeService(llm=forecaster_llms, store=store),
        searcher=FixtureAsOfSearch(
            default=(
                Evidence(
                    snippet="signal",
                    source="hosted",
                    source_id="http://a",
                    knowledge_time=datetime(2026, 1, 1, tzinfo=UTC),
                    score=0.5,
                ),
            )
        ),
        reasoning_llm=FixtureStructuredLLM(
            [
                {"reference_class": "rc", "base_rate": 0.4},
                {"rule": "none"},
                {"reference_class": "rc", "base_rate": 0.4},
                {"rule": "none"},
            ]
        ),
        forecast_llm=FixtureForecastLLM(default_response=0.6),
        supervisor_llm=FixtureSupervisorLLM(),
        leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM()),
        registry_store=store,
    )
    return HeuristicConductor(forecaster=forecaster)


def test_live_cadence_is_nightly() -> None:
    assert live_cadence().schedule_expression == "rate(1 day)"


def _refusing_conductor(store: InMemoryRegistryStore) -> HeuristicConductor:
    from forecaster.chain import Forecaster

    forecaster = Forecaster(
        intake=IntakeService(llm=FixtureStructuredLLM([{"question_type": "unknown"}]), store=store),
        searcher=FixtureAsOfSearch(default=()),
        reasoning_llm=FixtureStructuredLLM([]),
        forecast_llm=FixtureForecastLLM(default_response=0.5),
        supervisor_llm=FixtureSupervisorLLM(),
        leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM()),
        registry_store=store,
    )
    return HeuristicConductor(forecaster=forecaster)


class TestHarvestJob:
    def test_harvests_and_forecasts_open_questions(self) -> None:
        store = InMemoryRegistryStore()
        adapter = LiveHarvestAdapter.harvest(
            [{"id": "a", "question": "Will X ship?"}], harvest_time=_HARVEST_TIME
        )
        run = HarvestJob(conductor=_conductor(store)).run(adapter)
        assert run.count == 1
        assert run.refused == ()
        # A pending forecast was written to the registry.
        assert len(store.all_questions()) == 1

    def test_refused_question_counted_as_refused(self) -> None:
        store = InMemoryRegistryStore()
        adapter = LiveHarvestAdapter.harvest(
            [{"id": "a", "question": "Mmm?"}], harvest_time=_HARVEST_TIME
        )
        run = HarvestJob(conductor=_refusing_conductor(store)).run(adapter)
        assert run.count == 0
        assert run.refused == ("live:a",)

    def test_harvest_threads_benchmark_id_into_metadata(self) -> None:
        from resolution.benchmark_source import BENCHMARK_QUESTION_ID_KEY

        store = InMemoryRegistryStore()
        adapter = LiveHarvestAdapter.harvest(
            [{"id": "a", "question": "Will X ship?"}], harvest_time=_HARVEST_TIME
        )
        HarvestJob(conductor=_conductor(store)).run(adapter)
        question = store.all_questions()[0]
        # The benchmark id is recorded so the score job can resolve it later.
        assert question.metadata[BENCHMARK_QUESTION_ID_KEY] == "live:a"
        assert question.metadata["benchmark_source"] == "live"
        assert question.metadata["benchmark_external_id"] == "a"


class TestScoreCollection:
    def _seed(self, store: InMemoryRegistryStore, *, probability: float, outcome: float) -> str:
        qid = store.record_question(
            QuestionInput(
                text="Will X ship?",
                question_type="binary",
                domain="tech",
                resolution_criteria="GA.",
            )
        )
        store.record_forecast(
            ForecastInput(
                question_id=qid,
                as_of=_HARVEST_TIME,
                probability=probability,
                rationale="r",
                model_provenance={"m": "v"},
                repro_handle={"as_of": _HARVEST_TIME.isoformat()},
            )
        )
        store.record_resolution(
            ResolutionInput(
                question_id=qid,
                resolved_value=outcome,
                resolved_at=datetime(2026, 12, 1, tzinfo=UTC),
                source="gov",
            )
        )
        return qid

    def test_collect_scored_records(self) -> None:
        store = InMemoryRegistryStore()
        self._seed(store, probability=0.8, outcome=1.0)
        records = collect_scored_records(store)
        assert len(records) == 1
        assert records[0].domain == "tech"

    def test_skips_unresolved_and_non_binary(self) -> None:
        store = InMemoryRegistryStore()
        # Question with a forecast but no resolution -> skipped.
        unresolved = store.record_question(
            QuestionInput(
                text="Open one?",
                question_type="binary",
                domain="tech",
                resolution_criteria="GA.",
            )
        )
        store.record_forecast(
            ForecastInput(
                question_id=unresolved,
                as_of=_HARVEST_TIME,
                probability=0.5,
                rationale="r",
                model_provenance={"m": "v"},
                repro_handle={"as_of": _HARVEST_TIME.isoformat()},
            )
        )
        # Question with a non-binary resolution -> skipped.
        self._seed(store, probability=0.5, outcome=0.5)
        assert collect_scored_records(store) == ()

    def test_metrics_from_records(self) -> None:
        store = InMemoryRegistryStore()
        self._seed(store, probability=0.8, outcome=1.0)
        metrics = LiveMetrics.from_records(collect_scored_records(store))
        assert metrics.n == 1
        assert metrics.brier == pytest.approx((0.8 - 1.0) ** 2)
        assert metrics.log is not None

    def test_empty_metrics(self) -> None:
        metrics = LiveMetrics.from_records([])
        assert metrics.n == 0
        assert metrics.brier is None


class TestScoreJob:
    def test_resolves_then_scores(self) -> None:
        store = InMemoryRegistryStore()
        qid = store.record_question(
            QuestionInput(
                text="Will X win?",
                question_type="binary",
                domain="politics",
                resolution_criteria="Official.",
            )
        )
        store.record_forecast(
            ForecastInput(
                question_id=qid,
                as_of=_HARVEST_TIME,
                probability=0.7,
                rationale="r",
                model_provenance={"m": "v"},
                repro_handle={"as_of": _HARVEST_TIME.isoformat()},
            )
        )
        service = ResolutionService(
            store=store,
            source=MappingResolutionSource(
                {
                    qid: ResolvedOutcome(
                        resolved_value=1.0,
                        resolved_at=datetime(2026, 12, 1, tzinfo=UTC),
                        source="gov",
                    )
                }
            ),
        )
        run = ScoreJob(store=store, resolution_service=service).run()
        assert len(run.resolved) == 1
        assert run.metrics.n == 1
        assert run.metrics.brier == pytest.approx((0.7 - 1.0) ** 2)


class TestClaimAndRun:
    def test_runs_then_skips_on_repeat(self) -> None:
        run_state = InMemoryRunStateStore()
        calls: list[int] = []

        def action() -> str:
            calls.append(1)
            return "done"

        outcome1, result1 = claim_and_run(
            run_state, step_id="s1", tick_at=_HARVEST_TIME, action=action
        )
        assert outcome1 == ClaimOutcome.RAN
        assert result1 == "done"

        outcome2, result2 = claim_and_run(
            run_state, step_id="s1", tick_at=_HARVEST_TIME, action=action
        )
        assert outcome2 == ClaimOutcome.SKIPPED
        assert result2 is None
        assert len(calls) == 1  # action not re-run

    def test_failure_marked_and_reraised(self) -> None:
        run_state = InMemoryRunStateStore()

        def boom() -> str:
            msg = "kaboom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="kaboom"):
            claim_and_run(run_state, step_id="s2", tick_at=_HARVEST_TIME, action=boom)
        record = run_state.get_step("s2")
        assert record is not None and record.error_message == "kaboom"
