"""End-to-end forecast chain tests (C4.7/C4.8): complete record + replay."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.forecast.calibration import FrozenCalibration
from core.forecast.leakage_judge import FixtureLeakageJudgeLLM, LeakageJudge
from core.forecast.llm import FixtureForecastLLM
from core.forecast.search import Evidence, FixtureAsOfSearch
from core.forecast.supervisor import FixtureSupervisorLLM
from core.registry.store import InMemoryRegistryStore, RegistryStore
from forecaster.chain import Forecaster, build_evidence_query
from intake.llm import FixtureStructuredLLM
from intake.service import IntakeService

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)

_CLASSIFY = {"question_type": "binary", "entities": ["team"], "horizon": "2024"}
_NORMALIZE = {
    "canonical_text": "Will the team win the cup by 2025?",
    "domain": "sports",
    "resolution_criteria": "Official league result at season end.",
    "resolution_sources": ["league.example"],
    "close_time": "2025-01-01T00:00:00+00:00",
    "resolvable": True,
}
_BASE_RATE = {"reference_class": "cup finals", "base_rate": 0.4, "citations": ["http://a"]}
_DECOMPOSE = {"sub_questions": ["Will they reach the final?"], "rule": "none"}


def _evidence() -> Evidence:
    return Evidence(
        snippet="The team is in strong form.",
        source="hosted",
        source_id="http://a",
        knowledge_time=datetime(2024, 1, 1, tzinfo=UTC),
        score=0.8,
    )


def _build(
    *,
    store: RegistryStore,
    classify: dict = _CLASSIFY,
    leakage_flags: tuple[str, ...] = (),
) -> Forecaster:
    intake = IntakeService(llm=FixtureStructuredLLM([classify, _NORMALIZE]), store=store)
    return Forecaster(
        intake=intake,
        searcher=FixtureAsOfSearch(default=(_evidence(),)),
        reasoning_llm=FixtureStructuredLLM([_BASE_RATE, _DECOMPOSE]),
        forecast_llm=FixtureForecastLLM(default_response=0.6),
        supervisor_llm=FixtureSupervisorLLM(),
        leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM(flag_substrings=leakage_flags)),
        registry_store=store,
    )


def test_accepted_forecast_writes_complete_record() -> None:
    store = InMemoryRegistryStore()
    result = _build(store=store).forecast("Will the team win the cup?", as_of=AS_OF)

    assert result.accepted is True
    assert result.probability is not None and 0.0 < result.probability < 1.0
    assert result.quarantined is False
    assert result.evidence and result.evidence[0].source_id == "http://a"

    # A complete registry record: question + evidence_set + forecast.
    question = store.get_question(result.question_id)  # type: ignore[arg-type]
    assert question.domain == "sports"
    evidence_sets = store.evidence_sets_for(result.question_id)  # type: ignore[arg-type]
    assert len(evidence_sets) == 1
    assert evidence_sets[0].items[0].knowledge_time <= AS_OF
    forecast = store.get_forecast(result.forecast_id)  # type: ignore[arg-type]
    assert forecast.probability == result.probability
    assert forecast.evidence_set_id == evidence_sets[0].evidence_set_id
    assert forecast.model_provenance["forecast_llm"]["model_version"]
    assert forecast.repro_handle["as_of"] == AS_OF.isoformat()
    assert forecast.trace["ensemble"]["n_runs"] == 4


def test_market_freeze_value_injected_as_evidence() -> None:
    store = InMemoryRegistryStore()
    result = _build(store=store).forecast(
        "Will the team win the cup?",
        as_of=AS_OF,
        metadata={"market_freeze_value": 0.35},
    )
    assert result.accepted is True
    freeze_items = [e for e in result.evidence if e.source == "market_freeze"]
    assert len(freeze_items) == 1
    assert result.evidence[0] is freeze_items[0]  # injected first
    assert "0.35" in freeze_items[0].snippet
    assert freeze_items[0].knowledge_time == AS_OF  # as-of-safe by construction
    # The synthetic item is persisted in the recorded evidence set too.
    evidence_sets = store.evidence_sets_for(result.question_id)  # type: ignore[arg-type]
    assert any(item.source == "market_freeze" for item in evidence_sets[0].items)


def test_series_estimator_evidence_injected_after_freeze() -> None:
    from core.forecast.search import Evidence as _Evidence

    class _FakeSeriesEstimator:
        def evidence(self, metadata, *, as_of):
            return (
                _Evidence(
                    snippet="series base rate 0.52",
                    source="series_estimator",
                    source_id="forecastbench:fred-DFF",
                    knowledge_time=as_of,
                    score=1.0,
                ),
            )

    store = InMemoryRegistryStore()
    intake = IntakeService(llm=FixtureStructuredLLM([_CLASSIFY, _NORMALIZE]), store=store)
    forecaster = Forecaster(
        intake=intake,
        searcher=FixtureAsOfSearch(default=(_evidence(),)),
        reasoning_llm=FixtureStructuredLLM([_BASE_RATE, _DECOMPOSE]),
        forecast_llm=FixtureForecastLLM(default_response=0.6),
        supervisor_llm=FixtureSupervisorLLM(),
        leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM()),
        registry_store=store,
        series_estimator=_FakeSeriesEstimator(),  # type: ignore[arg-type]
    )
    result = forecaster.forecast(
        "Will the team win the cup?",
        as_of=AS_OF,
        metadata={"market_freeze_value": 0.35},
    )
    assert result.accepted is True
    sources = [e.source for e in result.evidence]
    # Anchor ordering: market freeze first, series estimate second, then retrieval.
    assert sources[:3] == ["market_freeze", "series_estimator", "hosted"]
    evidence_sets = store.evidence_sets_for(result.question_id)  # type: ignore[arg-type]
    assert any(item.source == "series_estimator" for item in evidence_sets[0].items)


def test_no_freeze_metadata_injects_nothing() -> None:
    result = _build(store=InMemoryRegistryStore()).forecast(
        "Will the team win the cup?", as_of=AS_OF, metadata={"benchmark_source": "x"}
    )
    assert all(e.source != "market_freeze" for e in result.evidence)


def test_non_numeric_freeze_value_ignored() -> None:
    result = _build(store=InMemoryRegistryStore()).forecast(
        "Will the team win the cup?", as_of=AS_OF, metadata={"market_freeze_value": "n/a"}
    )
    assert result.accepted is True
    assert all(e.source != "market_freeze" for e in result.evidence)


def test_refused_question_writes_no_forecast() -> None:
    store = InMemoryRegistryStore()
    result = _build(store=store, classify={"question_type": "unknown"}).forecast(
        "What is the meaning of life?", as_of=AS_OF
    )
    assert result.accepted is False
    assert result.refusal is not None
    assert result.forecast_id is None


def test_leakage_flag_quarantines_but_still_records() -> None:
    store = InMemoryRegistryStore()
    result = _build(store=store, leakage_flags=("aggregation_method",)).forecast(
        "Will the team win the cup?", as_of=AS_OF
    )
    assert result.quarantined is True
    forecast = store.get_forecast(result.forecast_id)  # type: ignore[arg-type]
    assert forecast.calibration_metadata["quarantined"] is True


def test_replay_is_byte_identical() -> None:
    r1 = _build(store=InMemoryRegistryStore()).forecast("Will the team win?", as_of=AS_OF)
    r2 = _build(store=InMemoryRegistryStore()).forecast("Will the team win?", as_of=AS_OF)
    assert r1.probability == r2.probability
    assert r1.rationale == r2.rationale


def _build_with_calibration(store: RegistryStore, calibration: FrozenCalibration) -> Forecaster:
    intake = IntakeService(llm=FixtureStructuredLLM([_CLASSIFY, _NORMALIZE]), store=store)
    return Forecaster(
        intake=intake,
        searcher=FixtureAsOfSearch(default=(_evidence(),)),
        reasoning_llm=FixtureStructuredLLM([_BASE_RATE, _DECOMPOSE]),
        forecast_llm=FixtureForecastLLM(default_response=0.6),
        supervisor_llm=FixtureSupervisorLLM(),
        leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM()),
        registry_store=store,
        calibration=calibration,
    )


def test_learned_calibration_drives_alpha_floor_and_provenance() -> None:
    # Identity-shaped Platt recalibrator, neutral alpha, and a floor of 0.45:
    # the fixture ensemble lands at 0.6, so the final probability must be the
    # floor-clamped 0.55 — provably the artifact's numbers, not DEFAULT_ALPHA.
    calibration = FrozenCalibration(
        method="platt",
        alpha=1.0,
        floor=0.45,
        a=1.0,
        b=0.0,
        n=32,
        artifact_hash="deadbeef",
    )
    store = InMemoryRegistryStore()
    result = _build_with_calibration(store, calibration).forecast(
        "Will the team win the cup?", as_of=AS_OF
    )
    assert result.accepted is True
    assert result.probability == pytest.approx(0.55)

    forecast = store.get_forecast(result.forecast_id)  # type: ignore[arg-type]
    assert forecast.repro_handle["calibration_artifact_hash"] == "deadbeef"
    assert forecast.calibration_metadata["alpha"] == 1.0
    assert forecast.calibration_metadata["floor"] == 0.45
    recal = forecast.calibration_metadata["recalibrator"]
    assert recal["recalibrator"] == "platt"
    assert recal["fitted"] is True
    assert recal["artifact_hash"] == "deadbeef"


def test_without_calibration_hash_is_recorded_as_none() -> None:
    store = InMemoryRegistryStore()
    result = _build(store=store).forecast("Will the team win the cup?", as_of=AS_OF)
    forecast = store.get_forecast(result.forecast_id)  # type: ignore[arg-type]
    assert forecast.repro_handle["calibration_artifact_hash"] is None


def test_subquestion_search_merges_and_dedupes_evidence() -> None:
    store = InMemoryRegistryStore()
    intake = IntakeService(llm=FixtureStructuredLLM([_CLASSIFY, _NORMALIZE]), store=store)
    extra = Evidence(
        snippet="semifinal odds",
        source="hosted",
        source_id="http://sub",
        knowledge_time=datetime(2024, 2, 1, tzinfo=UTC),
        score=0.7,
    )
    searcher = FixtureAsOfSearch(
        # The sub-question pass returns one new item and one duplicate of the
        # primary retrieval; the duplicate must be merged away.
        {"will they reach the final?": (extra, _evidence())},
        default=(_evidence(),),
    )
    forecaster = Forecaster(
        intake=intake,
        searcher=searcher,
        reasoning_llm=FixtureStructuredLLM([_BASE_RATE, _DECOMPOSE]),
        forecast_llm=FixtureForecastLLM(default_response=0.6),
        supervisor_llm=FixtureSupervisorLLM(),
        leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM()),
        registry_store=store,
        max_subquestion_searches=2,
    )
    result = forecaster.forecast("Will the team win the cup?", as_of=AS_OF)
    assert result.accepted is True
    assert {ev.source_id for ev in result.evidence} == {"http://a", "http://sub"}
    # One primary search + one per sub-question (the fixture decomposition has one).
    assert searcher.call_count == 2
    forecast = store.get_forecast(result.forecast_id)  # type: ignore[arg-type]
    assert forecast.repro_handle["max_subquestion_searches"] == 2
    assert forecast.repro_handle["n_evidence"] == 2
    # FixtureAsOfSearch exposes no search_config: recorded honestly as None.
    assert forecast.repro_handle["search_config"] is None


def test_agentic_searcher_config_lands_in_repro_handle() -> None:
    from core.forecast.agentic_search import AgenticAsOfSearcher, FixtureQueryPlanner

    store = InMemoryRegistryStore()
    intake = IntakeService(llm=FixtureStructuredLLM([_CLASSIFY, _NORMALIZE]), store=store)
    agentic = AgenticAsOfSearcher(
        inner=FixtureAsOfSearch(default=(_evidence(),)),
        planner=FixtureQueryPlanner(),
    )
    forecaster = Forecaster(
        intake=intake,
        searcher=agentic,
        reasoning_llm=FixtureStructuredLLM([_BASE_RATE, _DECOMPOSE]),
        forecast_llm=FixtureForecastLLM(default_response=0.6),
        supervisor_llm=FixtureSupervisorLLM(),
        leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM()),
        registry_store=store,
    )
    result = forecaster.forecast("Will the team win the cup?", as_of=AS_OF)
    forecast = store.get_forecast(result.forecast_id)  # type: ignore[arg-type]
    assert str(forecast.repro_handle["search_config"]).startswith("agentic|rounds=3")


class TestBuildEvidenceQuery:
    def test_entities_and_horizon_joined(self) -> None:
        query = build_evidence_query("Will X win by 2027?", ("X", "the cup"), "by 2027")
        assert query == "X the cup by 2027"

    def test_no_entities_falls_back_to_text(self) -> None:
        assert build_evidence_query("Will X win?", (), None) == "Will X win?"

    def test_blank_entities_ignored(self) -> None:
        assert build_evidence_query("Will X win?", ("  ", ""), "2027") == "Will X win?"

    def test_no_horizon_omitted(self) -> None:
        assert build_evidence_query("Will X win?", ("X",), None) == "X"
        assert build_evidence_query("Will X win?", ("X",), "  ") == "X"


def test_search_uses_entity_query_but_reasoning_keeps_full_text() -> None:
    store = InMemoryRegistryStore()
    classify = {"question_type": "binary", "entities": ["team", "cup"], "horizon": "by 2025"}
    intake = IntakeService(llm=FixtureStructuredLLM([classify, _NORMALIZE]), store=store)
    searcher = FixtureAsOfSearch(default=(_evidence(),))
    forecaster = Forecaster(
        intake=intake,
        searcher=searcher,
        reasoning_llm=FixtureStructuredLLM([_BASE_RATE, _DECOMPOSE]),
        forecast_llm=FixtureForecastLLM(default_response=0.6),
        supervisor_llm=FixtureSupervisorLLM(),
        leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM()),
        registry_store=store,
    )
    result = forecaster.forecast("Will the team win the cup?", as_of=AS_OF)
    assert result.accepted is True
    # Retrieval saw the compact entity+horizon query...
    assert searcher.queries[0][0] == "team cup by 2025"
    # ...while the recorded question keeps the full canonical text.
    question = store.get_question(result.question_id)  # type: ignore[arg-type]
    assert question.text == _NORMALIZE["canonical_text"]


def test_negative_subquestion_searches_raises() -> None:
    store = InMemoryRegistryStore()
    with pytest.raises(ValueError, match="max_subquestion_searches"):
        Forecaster(
            intake=IntakeService(llm=FixtureStructuredLLM([_CLASSIFY, _NORMALIZE]), store=store),
            searcher=FixtureAsOfSearch(),
            reasoning_llm=FixtureStructuredLLM([_BASE_RATE, _DECOMPOSE]),
            forecast_llm=FixtureForecastLLM(default_response=0.6),
            supervisor_llm=FixtureSupervisorLLM(),
            leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM()),
            registry_store=store,
            max_subquestion_searches=-1,
        )


def test_scaled_ensemble_with_subsets_records_knobs() -> None:
    store = InMemoryRegistryStore()
    intake = IntakeService(llm=FixtureStructuredLLM([_CLASSIFY, _NORMALIZE]), store=store)
    evidence = tuple(
        Evidence(
            snippet=f"signal {i}",
            source="hosted",
            source_id=f"http://e{i}",
            knowledge_time=datetime(2024, 1, 1, tzinfo=UTC),
            score=0.5,
        )
        for i in range(6)
    )
    forecaster = Forecaster(
        intake=intake,
        searcher=FixtureAsOfSearch(default=evidence),
        reasoning_llm=FixtureStructuredLLM([_BASE_RATE, _DECOMPOSE]),
        forecast_llm=FixtureForecastLLM(default_response=0.6),
        supervisor_llm=FixtureSupervisorLLM(),
        leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM()),
        registry_store=store,
        runs_per_agent=3,
        evidence_subset_fraction=0.5,
    )
    result = forecaster.forecast("Will the team win the cup?", as_of=AS_OF)
    assert result.accepted is True
    forecast = store.get_forecast(result.forecast_id)  # type: ignore[arg-type]
    assert forecast.trace["ensemble"]["n_runs"] == 12  # 4 agents x 3 runs
    assert forecast.repro_handle["runs_per_agent"] == 3
    assert forecast.repro_handle["evidence_subset_fraction"] == 0.5
    assert forecast.repro_handle["aggregator"] == "log_odds_trimmed_mean"


def test_bayesian_path_uses_base_rate_prior_and_likelihood_llm() -> None:
    from core.forecast.bayesian import FixtureEvidenceLikelihoodLLM

    store = InMemoryRegistryStore()
    intake = IntakeService(llm=FixtureStructuredLLM([_CLASSIFY, _NORMALIZE]), store=store)
    forecaster = Forecaster(
        intake=intake,
        searcher=FixtureAsOfSearch(default=(_evidence(),)),
        reasoning_llm=FixtureStructuredLLM([_BASE_RATE, _DECOMPOSE]),
        forecast_llm=FixtureForecastLLM(default_response=0.6),
        supervisor_llm=FixtureSupervisorLLM(),
        leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM()),
        registry_store=store,
        evidence_likelihood_llm=FixtureEvidenceLikelihoodLLM(default_response=1.0),
        bayesian_draws=6,
    )
    result = forecaster.forecast("Will the team win the cup?", as_of=AS_OF)

    assert result.accepted is True
    assert result.probability is not None
    forecast = store.get_forecast(result.forecast_id)  # type: ignore[arg-type]
    assert forecast.trace["ensemble"]["n_runs"] == 6
    # prior 0.4 + log-LR 1.0 -> posterior ~0.64; extremization pushes further up.
    # The final probability must sit clearly above both 0.5 and the 0.4 prior.
    assert result.probability > 0.55


def test_without_likelihood_llm_absolute_path_unchanged() -> None:
    store = InMemoryRegistryStore()
    result = _build(store=store).forecast("Will the team win the cup?", as_of=AS_OF)
    forecast = store.get_forecast(result.forecast_id)  # type: ignore[arg-type]
    assert forecast.trace["ensemble"]["n_runs"] == 4  # 4 method-agents x 1 run
