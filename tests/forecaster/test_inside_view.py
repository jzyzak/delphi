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
    build_subset_draw_requests,
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


def _evidence_pool(n: int) -> list[Evidence]:
    return [
        Evidence(
            snippet=f"signal {i}",
            source="hosted",
            source_id=f"http://e{i}",
            knowledge_time=datetime(2024, 1, 1, tzinfo=UTC),
            score=0.5,
        )
        for i in range(n)
    ]


class TestBuildForecastContentCaps:
    def test_truncates_to_max_evidence(self) -> None:
        content = build_forecast_content("Q?", _base_rate(), Decomposition(), _evidence_pool(25))
        assert "http://e19" in content
        assert "http://e20" not in content

    def test_snippet_cap_is_applied(self) -> None:
        long_ev = Evidence(
            snippet="x" * 900,
            source="hosted",
            source_id="http://long",
            knowledge_time=datetime(2024, 1, 1, tzinfo=UTC),
            score=0.5,
        )
        content = build_forecast_content(
            "Q?", _base_rate(), Decomposition(), [long_ev], snippet_max=100
        )
        assert "x" * 100 in content
        assert "x" * 101 not in content


class TestBuildSubsetDrawRequests:
    def _kwargs(self, n_evidence: int, **overrides: object) -> dict:
        kwargs: dict = {
            "question": "Will it rain?",
            "base_rate": _base_rate(),
            "decomposition": _decomp(),
            "evidence": _evidence_pool(n_evidence),
        }
        kwargs.update(overrides)
        return kwargs

    def test_full_fraction_matches_shared_content_path(self) -> None:
        reqs = build_subset_draw_requests(**self._kwargs(5, subset_fraction=1.0))
        shared = build_draw_requests(
            content=build_forecast_content(
                "Will it rain?", _base_rate(), _decomp(), _evidence_pool(5)
            )
        )
        assert reqs == shared

    def test_single_evidence_item_is_never_subset(self) -> None:
        reqs = build_subset_draw_requests(**self._kwargs(1, subset_fraction=0.5))
        assert len({r.content_hash for r in reqs}) == 1

    def test_subsets_are_deterministic(self) -> None:
        first = build_subset_draw_requests(**self._kwargs(10, subset_fraction=0.8))
        second = build_subset_draw_requests(**self._kwargs(10, subset_fraction=0.8))
        assert first == second

    def test_subsets_decorrelate_draw_contents(self) -> None:
        reqs = build_subset_draw_requests(**self._kwargs(10, subset_fraction=0.5, runs_per_agent=3))
        assert len(reqs) == 3 * len(METHOD_AGENTS)
        # Different draws see different evidence: more than one distinct content.
        assert len({r.content_hash for r in reqs}) > 1

    def test_subset_size_and_order_are_preserved(self) -> None:
        reqs = build_subset_draw_requests(**self._kwargs(10, subset_fraction=0.8))
        for req in reqs:
            ids = [
                line.split("]")[0].split("[")[1]
                for line in req.content.splitlines()
                if line.strip().startswith("- [http://e")
            ]
            assert len(ids) == 8  # round(0.8 * 10)
            indices = [int(i.removeprefix("http://e")) for i in ids]
            assert indices == sorted(indices)

    def test_run_index_is_global_and_contiguous(self) -> None:
        reqs = build_subset_draw_requests(**self._kwargs(10, subset_fraction=0.5, runs_per_agent=2))
        assert [r.run_index for r in reqs] == list(range(2 * len(METHOD_AGENTS)))

    def test_prompts_carry_method_agents(self) -> None:
        reqs = build_subset_draw_requests(**self._kwargs(10, subset_fraction=0.5))
        for agent, req in zip(METHOD_AGENTS, reqs, strict=True):
            assert f"[method-agent: {agent}]" in req.prompt

    def test_every_prompt_carries_the_martingale_discipline(self) -> None:
        # Both request builders (shared-content and seeded-subset paths) append
        # the near-random-walk guardrail — the measured yfinance failure mode.
        subset_reqs = build_subset_draw_requests(**self._kwargs(10, subset_fraction=0.5))
        shared_reqs = build_draw_requests(content="ctx")
        for req in (*subset_reqs, *shared_reqs):
            assert "near-random-walk" in req.prompt

    def test_market_anchored_prompt_names_the_freeze_item(self) -> None:
        reqs = build_draw_requests(content="ctx", agents=("market_anchored",))
        assert "[market_freeze]" in reqs[0].prompt

    @pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
    def test_invalid_fraction_raises(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="subset_fraction"):
            build_subset_draw_requests(**self._kwargs(10, subset_fraction=fraction))

    def test_bad_runs_per_agent_raises(self) -> None:
        with pytest.raises(ValueError, match="runs_per_agent"):
            build_subset_draw_requests(**self._kwargs(10, subset_fraction=0.5, runs_per_agent=0))

    def test_empty_agents_raises(self) -> None:
        with pytest.raises(ValueError, match="method-agent"):
            build_subset_draw_requests(**self._kwargs(10, subset_fraction=0.5, agents=()))

    def test_unknown_agent_falls_back_to_inside_view_prompt(self) -> None:
        reqs = build_subset_draw_requests(
            **self._kwargs(10, subset_fraction=0.5, agents=("mystery",))
        )
        assert "[method-agent: mystery]" in reqs[0].prompt
        assert "weak prior" in reqs[0].prompt


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


def test_empty_evidence_content_instructs_against_hedging() -> None:
    from forecaster.stages.base_rate import BaseRateEstimate
    from forecaster.stages.decompose import Decomposition
    from forecaster.stages.inside_view import build_forecast_content

    content = build_forecast_content(
        "Will X happen?",
        BaseRateEstimate(prior=0.3, reference_class="events like X"),
        Decomposition(sub_questions=(), rule="none"),
        (),
    )
    assert "none retrieved" in content
    assert "commit" in content.lower()
    assert "0.5" in content  # explicit anti-hedge instruction
