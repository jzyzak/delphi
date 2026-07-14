"""Supervisor acceptance tests (SU1-SU8 + §8)."""

from __future__ import annotations

import inspect
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from core.forecast.ensemble import EnsembleForecast, build_ensemble
from core.forecast.llm import ForecastDraw
from core.forecast.search import Evidence, FixtureAsOfSearch
from core.forecast.supervisor import (
    Confidence,
    DisagreementKind,
    FixtureSupervisorLLM,
    FixtureSupervisorResponse,
    InMemoryReconciliationCache,
    Supervisor,
    build_resolution_query,
    build_supervisor_config,
    detect_disagreement,
)

T_AS_OF = datetime(2024, 2, 1, 12, 0, tzinfo=UTC)
T_EARLY = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
T_LATE = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


def _draw(
    probability: float,
    run_index: int = 0,
    *,
    rationale: str | None = None,
) -> ForecastDraw:
    provenance: dict[str, object] = {"run_index": run_index}
    if rationale is not None:
        provenance["rationale"] = rationale
    return ForecastDraw(
        probability=probability,
        run_index=run_index,
        model_version="m1",
        prompt_version="p1",
        provenance=provenance,
    )


def _simple_ensemble(
    probabilities: list[float], *, question: str | None = None
) -> EnsembleForecast:
    draws = [_draw(p, i) for i, p in enumerate(probabilities)]
    base = build_ensemble(draws, aggregator="median", knowledge_time=T_AS_OF)
    provenance = dict(base.provenance)
    if question is not None:
        provenance["question"] = question
    return EnsembleForecast(
        probability=base.probability,
        uncertainty=base.uncertainty,
        n=base.n,
        aggregator=base.aggregator,
        trim_fraction=base.trim_fraction,
        knowledge_time=base.knowledge_time,
        draws=base.draws,
        provenance=provenance,
    )


def _supervisor(
    *,
    probabilities: list[float] | None = None,
    llm_responses: dict[str, FixtureSupervisorResponse] | None = None,
    search_responses: dict[str, tuple[Evidence, ...]] | None = None,
    question: str | None = "Will DrugX be approved?",
) -> tuple[Supervisor, EnsembleForecast]:
    ensemble = _simple_ensemble(
        probabilities or [0.2, 0.2, 0.8, 0.8, 0.8],
        question=question,
    )
    search = FixtureAsOfSearch(responses=search_responses or {})
    llm = FixtureSupervisorLLM(responses=llm_responses)
    supervisor = Supervisor(search, llm, InMemoryReconciliationCache())
    return supervisor, ensemble


class TestSU1Floor:
    """SU1: supervisor output never underperforms the robust aggregate."""

    def test_low_confidence_falls_back_to_aggregate(self) -> None:
        supervisor, ensemble = _supervisor(
            llm_responses={
                "multimodal": FixtureSupervisorResponse(0.9, Confidence.LOW),
            },
        )
        result = supervisor.reconcile(ensemble)
        assert result.probability == pytest.approx(ensemble.probability)
        assert result.applied is False
        assert result.aggregate_probability == pytest.approx(ensemble.probability)

    def test_medium_confidence_falls_back_to_aggregate(self) -> None:
        supervisor, ensemble = _supervisor(
            llm_responses={
                "multimodal": FixtureSupervisorResponse(0.9, Confidence.MEDIUM),
            },
        )
        result = supervisor.reconcile(ensemble)
        assert result.probability == pytest.approx(ensemble.probability)
        assert result.applied is False

    def test_high_confidence_applies_update(self) -> None:
        supervisor, ensemble = _supervisor(
            llm_responses={
                "multimodal": FixtureSupervisorResponse(0.75, Confidence.HIGH),
            },
        )
        result = supervisor.reconcile(ensemble)
        assert result.probability == pytest.approx(0.75)
        assert result.applied is True

    def test_agreeing_draws_return_aggregate_unchanged(self) -> None:
        supervisor, ensemble = _supervisor(probabilities=[0.5, 0.51, 0.49, 0.5, 0.5])
        result = supervisor.reconcile(ensemble)
        assert result.probability == pytest.approx(ensemble.probability)
        assert result.applied is False


class TestSU2NoNaiveAggregation:
    """SU2: no pick-best / blend-all path exists."""

    def test_supervisor_source_has_no_naive_patterns(self) -> None:
        source = Path("core/forecast/supervisor.py").read_text(encoding="utf-8")
        forbidden = [
            "pick_best",
            "pick best",
            "blend_all",
            "blend all",
            "best_of",
            "weighted_average",
        ]
        for pattern in forbidden:
            assert pattern not in source.lower()

    def test_reconcile_only_applies_or_falls_back(self) -> None:
        supervisor, ensemble = _supervisor(
            llm_responses={
                "multimodal": FixtureSupervisorResponse(0.33, Confidence.HIGH),
            },
        )
        result = supervisor.reconcile(ensemble)
        assert result.applied is True
        assert result.probability == pytest.approx(0.33)
        assert result.probability != pytest.approx(
            sum(d.probability for d in ensemble.draws) / len(ensemble.draws)
        )


class TestSU3DisagreementSearch:
    """SU3: divergent traces trigger targeted as-of queries."""

    def test_bimodal_draws_trigger_search(self) -> None:
        supervisor, ensemble = _supervisor(probabilities=[0.15, 0.18, 0.82, 0.85, 0.88])
        supervisor.reconcile(ensemble)
        assert supervisor.search_call_count == 1
        assert supervisor._search.queries  # type: ignore[attr-defined]
        query = supervisor._search.queries[0][0]  # type: ignore[attr-defined]
        assert "bimodal" in query.lower() or "clusters" in query.lower()

    def test_agreeing_draws_skip_search(self) -> None:
        supervisor, ensemble = _supervisor(probabilities=[0.48, 0.49, 0.5, 0.51, 0.52])
        supervisor.reconcile(ensemble)
        assert supervisor.search_call_count == 0

    def test_outlier_query_uses_provenance_rationale(self) -> None:
        draws = [
            _draw(0.5, 0),
            _draw(0.52, 1),
            _draw(0.9, 2, rationale="CRL risk elevated"),
        ]
        ensemble = build_ensemble(draws, aggregator="median", knowledge_time=T_AS_OF)
        disagreement = detect_disagreement(ensemble)
        query = build_resolution_query(ensemble, disagreement)
        assert "CRL risk elevated" in query


class TestSU4ConfidenceGate:
    """SU4: high-confidence replaces aggregate; medium/low discarded."""

    def test_high_confidence_replaces_aggregate(self) -> None:
        supervisor, ensemble = _supervisor(
            llm_responses={"multimodal": FixtureSupervisorResponse(0.72, Confidence.HIGH)},
        )
        result = supervisor.reconcile(ensemble)
        assert result.applied is True
        assert result.confidence == Confidence.HIGH
        assert result.probability == pytest.approx(0.72)
        assert result.probability != pytest.approx(ensemble.probability)

    def test_low_confidence_discarded(self) -> None:
        supervisor, ensemble = _supervisor(
            llm_responses={"multimodal": FixtureSupervisorResponse(0.72, Confidence.LOW)},
        )
        result = supervisor.reconcile(ensemble)
        assert result.applied is False
        assert result.probability == pytest.approx(ensemble.probability)


class TestSU5ForecastLayerBoundary:
    """SU5: supervisor cannot import or affect gate/portfolio."""

    def test_supervisor_module_has_no_capital_path_imports(self) -> None:
        import core.forecast.supervisor as mod

        source = inspect.getsource(mod)
        assert "harness.gates" not in source
        assert "portfolio" not in source
        assert "data.ingest" not in source

    def test_forecast_package_has_no_capital_path_imports(self) -> None:
        init_source = Path("core/forecast/__init__.py").read_text(encoding="utf-8")
        assert "harness.gates" not in init_source
        assert "portfolio" not in init_source
        assert "data.ingest" not in init_source


class TestSU6LeakageFree:
    """SU6: resolution search is as-of; no post-as_of evidence."""

    def test_post_as_of_evidence_raises(self) -> None:
        future_ev = Evidence(
            snippet="future fact",
            source="fda",
            source_id="FUT",
            knowledge_time=T_LATE,
            score=1.0,
        )

        class LeakySearch:
            def as_of_search(self, query: str, *, as_of: datetime) -> tuple[Evidence, ...]:
                return (future_ev,)

        supervisor = Supervisor(
            LeakySearch(),
            FixtureSupervisorLLM(),
            InMemoryReconciliationCache(),
        )
        _, ensemble = _supervisor()
        with pytest.raises(RuntimeError, match="knowledge_time > as_of"):
            supervisor.reconcile(ensemble)

    def test_search_pinned_at_ensemble_knowledge_time(self) -> None:
        search = FixtureAsOfSearch()
        supervisor = Supervisor(
            search,
            FixtureSupervisorLLM(
                responses={"multimodal": FixtureSupervisorResponse(0.7, Confidence.LOW)}
            ),
            InMemoryReconciliationCache(),
        )
        _, ensemble = _supervisor()
        supervisor.reconcile(ensemble)
        assert search.queries[0][1] == T_AS_OF


class TestSU7Reproducibility:
    """SU7: trajectory + decision cached; identical inputs reproduce."""

    def test_identical_inputs_reproduce_from_cache(self) -> None:
        cache = InMemoryReconciliationCache()
        search = FixtureAsOfSearch()
        llm = FixtureSupervisorLLM(
            responses={"multimodal": FixtureSupervisorResponse(0.7, Confidence.HIGH)}
        )
        supervisor = Supervisor(search, llm, cache)
        _, ensemble = _supervisor()

        first = supervisor.reconcile(ensemble)
        second = supervisor.reconcile(ensemble)

        assert first.probability == second.probability
        assert first.applied == second.applied
        assert first.trajectory == second.trajectory
        assert llm.call_count == 1
        assert search.call_count == 1
        assert second.provenance.get("cached") is True


class TestSU8Determinism:
    """SU8: fixture LLM + cache -> deterministic."""

    def test_fixture_llm_is_deterministic(self) -> None:
        results: list[float] = []
        for _ in range(3):
            supervisor, ensemble = _supervisor(
                llm_responses={"multimodal": FixtureSupervisorResponse(0.65, Confidence.HIGH)},
            )
            result = supervisor.reconcile(ensemble)
            results.append(result.probability)
        assert results == [pytest.approx(0.65)] * 3


class TestDetectDisagreement:
    """§8 unit tests for disagreement detection."""

    def test_happy_path_multimodal(self) -> None:
        ensemble = _simple_ensemble([0.1, 0.15, 0.85, 0.9])
        disagreement = detect_disagreement(ensemble)
        assert disagreement.kind == DisagreementKind.MULTIMODAL
        assert disagreement.material is True

    def test_boundary_single_draw_no_disagreement(self) -> None:
        ensemble = _simple_ensemble([0.6])
        disagreement = detect_disagreement(ensemble)
        assert disagreement.kind == DisagreementKind.NONE

    def test_failure_negative_spread_threshold_raises(self) -> None:
        ensemble = _simple_ensemble([0.5, 0.6])
        with pytest.raises(ValueError, match="spread_threshold"):
            detect_disagreement(ensemble, spread_threshold=-0.1)


class TestBuildSupervisorConfig:
    def test_happy_path_serializes_params(self) -> None:
        config = build_supervisor_config(spread_threshold=0.2)
        assert "spread=0.2" in config

    def test_failure_negative_param_raises_via_detect(self) -> None:
        ensemble = _simple_ensemble([0.5])
        with pytest.raises(ValueError):
            detect_disagreement(ensemble, min_cluster_size=0)


class TestBuildResolutionQuery:
    def test_happy_path_uses_question_from_provenance(self) -> None:
        ensemble = _simple_ensemble([0.2, 0.8, 0.8], question="Will the incumbent win?")
        disagreement = detect_disagreement(ensemble)
        query = build_resolution_query(ensemble, disagreement)
        assert "Will the incumbent win?" in query

    def test_boundary_no_question_degrades_gracefully(self) -> None:
        ensemble = _simple_ensemble([0.2, 0.8, 0.8])
        disagreement = detect_disagreement(ensemble)
        query = build_resolution_query(ensemble, disagreement)
        assert "forecast reference class" in query


class TestSupervisorSourceContract:
    def test_reconcile_docstring_matches_contract(self) -> None:
        doc = Supervisor.reconcile.__doc__ or ""
        assert "high" in doc.lower()
        assert "aggregate" in doc.lower()
        assert re.search(r"forecast layer", doc, re.IGNORECASE)
