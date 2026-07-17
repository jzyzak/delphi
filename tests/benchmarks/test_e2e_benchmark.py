"""End-to-end benchmark exercises (hermetic: network + LLM mocked).

Runs the two benchmark paths built in C7/C9 all the way through, stubbing only
the external transports (HTTP via ``httpx.MockTransport``; the LLM/search via the
deterministic fixtures) as CLAUDE.md §2.8 requires. This is the closest a
credential-free environment can get to a real ``delphi eval`` / ``delphi bench
live`` run, and it guards the whole wiring against regressions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from benchmarks.fetchers.metaculus_api import MetaculusFetcher
from benchmarks.live_loop.harvest import HarvestJob
from benchmarks.live_loop.score import ScoreJob
from benchmarks.market_consensus import consensus_baseline
from benchmarks.metaculus import MetaculusAdapter
from benchmarks.suites import QuestionForecast, build_eval_context
from common.http.client import HttpClient
from conductor.heuristic import HeuristicConductor
from core.forecast.leakage_judge import FixtureLeakageJudgeLLM, LeakageJudge, Trace, TraceComponent
from core.forecast.llm import FixtureForecastLLM
from core.forecast.search import Evidence, FixtureAsOfSearch
from core.forecast.supervisor import FixtureSupervisorLLM
from core.orchestration.budget import InMemoryBudgetLedger
from core.registry.store import InMemoryRegistryStore
from evaluation.harness import EvalHarness
from evaluation.report import render_leakage_audit, render_report
from forecaster.chain import Forecaster
from intake.llm import FixtureStructuredLLM
from intake.service import IntakeService
from resolution.benchmark_source import BenchmarkResolutionSource
from resolution.service import ResolutionService

_HARVEST_TIME = datetime(2026, 7, 1, tzinfo=UTC)


def _http(handler: Any) -> HttpClient:
    return HttpClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _resolved_post(idx: int) -> dict[str, Any]:
    return {
        "id": idx,
        "title": f"Will resolved event {idx} occur?",
        "projects": {"category": [{"slug": "economy"}]},
        "question": {
            "id": 1000 + idx,
            "type": "binary",
            "description": "criteria",
            "open_time": "2026-01-01T00:00:00Z",
            "actual_resolve_time": "2026-06-01T00:00:00Z",
            "resolution": "yes" if idx % 2 == 0 else "no",
            "aggregations": {"recency_weighted": {"latest": {"centers": [0.55]}}},
        },
    }


class TestRetrospectiveEndToEnd:
    def test_fetch_map_calibrate_score_and_audit(self) -> None:
        posts = [_resolved_post(i) for i in range(6)]

        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": posts, "next": None})

        records = MetaculusFetcher(http=_http(handler)).fetch()
        adapter = MetaculusAdapter.from_records(records)

        # Deterministic raw forecasts (the real forecaster_fn is unit-tested
        # separately; here we drive the assembly/calibration/scoring end to end).
        def forecast_fn(_text: str, _as_of: datetime, _metadata: Any = None) -> QuestionForecast:
            trace = Trace(
                component=TraceComponent.SEARCH,
                as_of=datetime(2026, 1, 1, tzinfo=UTC),
                text="clean as-of evidence",
            )
            return QuestionForecast(accepted=True, raw_probability=0.6, traces=(trace,))

        harness = EvalHarness(budget_ledger=InMemoryBudgetLedger(cap=1000, trials_count=lambda: 0))
        judge = LeakageJudge(FixtureLeakageJudgeLLM())
        ctx = build_eval_context(
            adapter,
            forecast_fn,
            harness=harness,
            judge=judge,
            calibration_fraction=0.5,
            extra_baselines=(consensus_baseline(adapter, price_key="community_prediction"),),
        )

        report = render_report(ctx.inputs, harness=ctx.harness)
        assert "Proper scores" in report
        assert "market_consensus" in report

        assert ctx.judge is not None
        audit = render_leakage_audit(ctx.judge, ctx.inputs.traces)
        assert "leakage" in audit.lower()


def _fixture_conductor(store: InMemoryRegistryStore) -> HeuristicConductor:
    classify = {"question_type": "binary", "entities": ["X"]}
    normalize = {
        "canonical_text": "Will X ship?",
        "domain": "tech",
        "resolution_criteria": "Resolves YES on GA.",
        "resolvable": True,
    }
    forecaster = Forecaster(
        intake=IntakeService(llm=FixtureStructuredLLM([classify, normalize]), store=store),
        searcher=FixtureAsOfSearch(
            default=(
                Evidence(
                    snippet="signal",
                    source="hosted",
                    source_id="http://a",
                    knowledge_time=datetime(2026, 6, 1, tzinfo=UTC),
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
    return HeuristicConductor(forecaster=forecaster)


class TestLiveLoopEndToEnd:
    def test_harvest_then_resolve_and_score(self) -> None:
        # Harvest one open Metaculus question at the harvest-time pin.
        open_post = {
            "id": 42,
            "title": "Will X ship?",
            "question": {"id": 4242, "type": "binary", "description": "GA"},
        }

        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [open_post], "next": None})

        open_records = MetaculusFetcher(http=_http(handler)).fetch(freeze_at=_HARVEST_TIME)
        harvest_adapter = MetaculusAdapter.from_records(open_records)

        store = InMemoryRegistryStore()
        harvest = HarvestJob(conductor=_fixture_conductor(store))
        run = harvest.run(harvest_adapter)
        assert run.count == 1

        # The benchmark id was threaded into the registry so we can resolve it.
        question = store.all_questions()[0]
        assert question.metadata["benchmark_question_id"] == "metaculus:42"

        # Build resolutions for the same benchmark id and score.
        resolved_post = {
            "id": 42,
            "title": "Will X ship?",
            "question": {
                "id": 4242,
                "type": "binary",
                "open_time": "2026-01-01T00:00:00Z",
                "actual_resolve_time": "2026-12-01T00:00:00Z",
                "resolution": "yes",
            },
        }

        def resolved_handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [resolved_post], "next": None})

        resolved_records = MetaculusFetcher(http=_http(resolved_handler)).fetch()
        resolutions = MetaculusAdapter.from_records(resolved_records).resolutions()
        source = BenchmarkResolutionSource(resolutions)
        service = ResolutionService(store=store, source=source)
        result = ScoreJob(store=store, resolution_service=service).run()

        assert len(result.resolved) == 1
        assert result.metrics.n == 1
        assert result.metrics.brier is not None
