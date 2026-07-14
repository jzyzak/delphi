"""Tests for the heuristic conductor (C8.2)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from conductor.heuristic import (
    ConductorResult,
    FixtureRedTeamLLM,
    HeuristicConductor,
    WorkflowTrace,
)
from conductor.roles import RoleId
from core.forecast.leakage_judge import FixtureLeakageJudgeLLM, LeakageJudge
from core.forecast.llm import FixtureForecastLLM
from core.forecast.search import Evidence, FixtureAsOfSearch
from core.forecast.supervisor import FixtureSupervisorLLM
from core.registry.store import InMemoryRegistryStore
from forecaster.chain import Forecaster, ForecastResult
from intake.llm import FixtureStructuredLLM
from intake.service import IntakeService

_AS_OF = datetime(2024, 6, 1, tzinfo=UTC)

_CLASSIFY = {"question_type": "binary", "entities": ["X"]}
_NORMALIZE = {
    "canonical_text": "Will X ship by 2025?",
    "domain": "tech",
    "resolution_criteria": "Resolves YES on GA.",
    "close_time": "2025-06-01T00:00:00+00:00",
    "resolvable": True,
}


def _forecaster(store: InMemoryRegistryStore, *, classify: dict = _CLASSIFY) -> Forecaster:
    return Forecaster(
        intake=IntakeService(llm=FixtureStructuredLLM([classify, _NORMALIZE]), store=store),
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


class _StubForecaster:
    """Returns a preset ForecastResult, counting calls (to drive the verifier loop)."""

    def __init__(self, result: ForecastResult) -> None:
        self._result = result
        self.calls = 0

    def forecast(
        self,
        question: str,
        *,
        as_of: datetime,
        metadata: Mapping[str, Any] | None = None,
    ) -> ForecastResult:
        self.calls += 1
        self.last_metadata = metadata
        return self._result


def _result(*, accepted: bool = True, quarantined: bool = False) -> ForecastResult:
    return ForecastResult(
        accepted=accepted,
        question_id="q1" if accepted else None,
        forecast_id="f1" if accepted else None,
        probability=0.6 if accepted else None,
        calibrated=None,
        uncertainty=None,
        evidence=(),
        leakage=None,
        quarantined=quarantined,
        rationale="rationale",
        refusal=None,
    )


class TestConductHappyPath:
    def test_records_full_workflow_and_matches_chain(self) -> None:
        # The conductor's probability equals the fixed chain's on a clean run.
        chain_prob = _forecaster(InMemoryRegistryStore()).forecast("Will X ship?", as_of=_AS_OF)
        conductor = HeuristicConductor(forecaster=_forecaster(InMemoryRegistryStore()))
        result = conductor.conduct("Will X ship?", as_of=_AS_OF)
        assert result.forecast.probability == pytest.approx(chain_prob.probability)
        assert result.verifier_accepted
        assert result.revisions == 0
        assert result.red_team_counter
        assert result.workflow.route == tuple(r.value for r in conductor.route)
        assert len(result.workflow.steps) == 8

    def test_workflow_as_dict_roundtrips_fields(self) -> None:
        conductor = HeuristicConductor(forecaster=_forecaster(InMemoryRegistryStore()))
        result = conductor.conduct("Will X ship?", as_of=_AS_OF)
        payload = result.workflow.as_dict()
        assert payload["revisions"] == 0
        assert isinstance(payload["steps"], list)
        assert payload["steps"][0]["role_id"] == RoleId.RESEARCHER.value

    def test_custom_red_team_counter(self) -> None:
        conductor = HeuristicConductor(
            forecaster=_forecaster(InMemoryRegistryStore()),
            red_team=FixtureRedTeamLLM(counter="the base rate is stale"),
        )
        result = conductor.conduct("Will X ship?", as_of=_AS_OF)
        assert result.red_team_counter == "the base rate is stale"


class TestVerifierLoop:
    def test_revises_on_quarantine_then_accepts_with_quarantine(self) -> None:
        stub = _StubForecaster(_result(quarantined=True))
        conductor = HeuristicConductor(forecaster=cast(Forecaster, stub), max_revisions=1)
        result = conductor.conduct("Q", as_of=_AS_OF)
        assert result.revisions == 1
        assert stub.calls == 2  # initial + one revision
        assert not result.verifier_accepted

    def test_no_revision_when_clean(self) -> None:
        stub = _StubForecaster(_result(quarantined=False))
        conductor = HeuristicConductor(forecaster=cast(Forecaster, stub), max_revisions=2)
        result = conductor.conduct("Q", as_of=_AS_OF)
        assert result.revisions == 0
        assert stub.calls == 1
        assert result.verifier_accepted

    def test_refusal_returns_single_step_workflow(self) -> None:
        stub = _StubForecaster(_result(accepted=False))
        conductor = HeuristicConductor(forecaster=cast(Forecaster, stub))
        result = conductor.conduct("Q", as_of=_AS_OF)
        assert not result.forecast.accepted
        assert not result.verifier_accepted
        assert result.workflow.route == (RoleId.RESEARCHER.value,)


def test_negative_max_revisions_raises() -> None:
    with pytest.raises(ValueError, match="max_revisions"):
        HeuristicConductor(forecaster=_forecaster(InMemoryRegistryStore()), max_revisions=-1)


def test_fixture_red_team_default_counter_mentions_probability() -> None:
    counter = FixtureRedTeamLLM().challenge(question="Q", probability=0.42, as_of=_AS_OF)
    assert "0.42" in counter


def test_model_provenance_records_route_and_red_team() -> None:
    conductor = HeuristicConductor(forecaster=_forecaster(InMemoryRegistryStore()))
    provenance = conductor.model_provenance()
    assert provenance["conductor"] == "heuristic"
    assert isinstance(provenance["route"], list)
    assert "red_team" in provenance


def test_conductor_result_defaults() -> None:
    result = ConductorResult(
        forecast=_result(),
        workflow=WorkflowTrace(steps=(), route=()),
    )
    assert result.verifier_accepted
    assert result.revisions == 0
