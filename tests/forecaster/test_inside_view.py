"""Unit tests for inside-view + ensemble assembly (C4.3)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.forecast.bayesian import FixtureEvidenceLikelihoodLLM
from core.forecast.llm import FixtureForecastLLM
from core.forecast.search import Evidence
from forecaster.stages.base_rate import BaseRateEstimate
from forecaster.stages.decompose import Decomposition, SubQuestion
from forecaster.stages.inside_view import (
    METHOD_AGENTS,
    assemble_bayesian_ensemble,
    assemble_ensemble,
    build_draw_requests,
    build_forecast_content,
)

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _base_rate(prior: float = 0.3) -> BaseRateEstimate:
    return BaseRateEstimate(prior=prior, reference_class="rc", rationale="because")


def _decomp() -> Decomposition:
    return Decomposition(sub_questions=(SubQuestion(text="Will A?"),), rule="product")


def _ev() -> Evidence:
    return Evidence(
        snippet="rain expected",
        source="hosted",
        source_id="http://a",
        knowledge_time=datetime(2024, 1, 1, tzinfo=UTC),
        score=0.5,
    )


class TestBuildForecastContent:
    def test_includes_key_context(self) -> None:
        content = build_forecast_content("Will it rain?", _base_rate(), _decomp(), [_ev()])
        assert "Will it rain?" in content
        assert "Base rate (prior): 0.3000" in content
        assert "http://a" in content
        assert "Will A?" in content

    def test_no_evidence_notes_absence(self) -> None:
        content = build_forecast_content("Q?", _base_rate(), Decomposition(), [])
        assert "none retrieved" in content


class TestBuildDrawRequests:
    def test_one_per_agent(self) -> None:
        reqs = build_draw_requests(content="ctx")
        assert len(reqs) == len(METHOD_AGENTS)
        assert [r.run_index for r in reqs] == list(range(len(METHOD_AGENTS)))
        assert len({r.content_hash for r in reqs}) == 1  # shared content
        assert "base_rate_heavy" in reqs[0].prompt

    def test_runs_per_agent(self) -> None:
        reqs = build_draw_requests(content="ctx", agents=("a", "b"), runs_per_agent=3)
        assert len(reqs) == 6

    def test_bad_runs_per_agent(self) -> None:
        with pytest.raises(ValueError, match="runs_per_agent"):
            build_draw_requests(content="ctx", runs_per_agent=0)

    def test_empty_agents(self) -> None:
        with pytest.raises(ValueError, match="at least one method-agent"):
            build_draw_requests(content="ctx", agents=())


class TestAssembleEnsemble:
    def test_aggregates_draws(self) -> None:
        reqs = build_draw_requests(content="ctx")
        llm = FixtureForecastLLM(default_response=0.7)
        ensemble = assemble_ensemble(llm, reqs, knowledge_time=AS_OF)
        assert ensemble.probability == pytest.approx(0.7)
        assert ensemble.n == len(METHOD_AGENTS)
        assert ensemble.knowledge_time == AS_OF

    def test_empty_requests_raises(self) -> None:
        with pytest.raises(ValueError, match="requests must be non-empty"):
            assemble_ensemble(FixtureForecastLLM(), [], knowledge_time=AS_OF)


class TestBayesianEnsemble:
    def test_combines_prior_with_log_lrs(self) -> None:
        llm = FixtureEvidenceLikelihoodLLM(default_response=0.0)  # neutral evidence
        ensemble = assemble_bayesian_ensemble(
            llm, content="ctx", base_rate=_base_rate(0.3), knowledge_time=AS_OF, n=5
        )
        # Neutral log-LR => posterior stays at the prior.
        assert ensemble.probability == pytest.approx(0.3, abs=1e-6)
        assert ensemble.n == 5
