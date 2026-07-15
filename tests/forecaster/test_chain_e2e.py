"""End-to-end forecast chain tests (C4.7/C4.8): complete record + replay."""

from __future__ import annotations

from datetime import UTC, datetime

from core.forecast.leakage_judge import FixtureLeakageJudgeLLM, LeakageJudge
from core.forecast.llm import FixtureForecastLLM
from core.forecast.search import Evidence, FixtureAsOfSearch
from core.forecast.supervisor import FixtureSupervisorLLM
from core.registry.store import InMemoryRegistryStore, RegistryStore
from forecaster.chain import Forecaster
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
    # prior 0.4 (log-odds ~ -0.405) + log-LR 1.0 -> posterior sigmoid(0.595) ~ 0.64;
    # the aggregate must sit above the prior (supportive evidence), not at 0.5.
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
