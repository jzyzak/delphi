"""Unit tests for agentic iterative as-of search (§2.1 + §8)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from common.llm import BedrockStructuredClient, LLMConfig
from core.forecast.agentic_search import (
    AgenticAsOfSearcher,
    BedrockQueryPlannerLLM,
    FixtureQueryPlanner,
    QueryPlan,
    QueryPlannerLLM,
    rank_evidence,
)
from core.forecast.search import Evidence, FixtureAsOfSearch

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _ev(
    source_id: str, *, score: float = 0.5, days: int = 1, snippet: str | None = None
) -> Evidence:
    return Evidence(
        snippet=snippet if snippet is not None else f"snippet {source_id}",
        source="hosted",
        source_id=source_id,
        knowledge_time=datetime(2024, 1, days, tzinfo=UTC),
        score=score,
    )


class TestFixtureQueryPlanner:
    def test_plays_back_scripted_plans_then_stops(self) -> None:
        planner = FixtureQueryPlanner([QueryPlan(queries=("a",))])
        first = planner.plan(question="q", evidence=(), round_index=1, max_queries=3)
        second = planner.plan(question="q", evidence=(), round_index=2, max_queries=3)
        assert first.queries == ("a",)
        assert second.stop is True
        assert planner.call_count == 2
        assert planner.calls[0]["round_index"] == 1

    def test_satisfies_protocol(self) -> None:
        assert isinstance(FixtureQueryPlanner(), QueryPlannerLLM)


class TestRankEvidence:
    def test_orders_by_score_then_recency(self) -> None:
        low = _ev("low", score=0.1, days=5)
        high = _ev("high", score=0.9, days=1)
        tied_old = _ev("tied-old", score=0.5, days=1)
        tied_new = _ev("tied-new", score=0.5, days=9)
        ranked = rank_evidence([low, tied_old, high, tied_new], max_items=10)
        assert [ev.source_id for ev in ranked] == ["high", "tied-new", "tied-old", "low"]

    def test_truncates_to_max_items(self) -> None:
        ranked = rank_evidence([_ev(f"e{i}", score=i / 10) for i in range(5)], max_items=2)
        assert len(ranked) == 2
        assert ranked[0].source_id == "e4"

    def test_empty_is_empty(self) -> None:
        assert rank_evidence([], max_items=3) == ()

    def test_invalid_max_items_raises(self) -> None:
        with pytest.raises(ValueError, match="max_items"):
            rank_evidence([], max_items=0)


def _searcher(
    responses: dict[str, tuple[Evidence, ...]],
    plans: list[QueryPlan],
    **kwargs: Any,
) -> tuple[AgenticAsOfSearcher, FixtureAsOfSearch, FixtureQueryPlanner]:
    inner = FixtureAsOfSearch(responses)
    planner = FixtureQueryPlanner(plans)
    return AgenticAsOfSearcher(inner=inner, planner=planner, **kwargs), inner, planner


class TestAgenticAsOfSearcher:
    def test_single_round_runs_only_the_callers_query(self) -> None:
        agentic, inner, planner = _searcher({"q": (_ev("a"),)}, [], max_rounds=1)
        results = agentic.as_of_search("q", as_of=AS_OF)
        assert [ev.source_id for ev in results] == ["a"]
        assert inner.call_count == 1
        assert planner.call_count == 0

    def test_follow_up_rounds_accumulate_and_dedupe(self) -> None:
        agentic, inner, planner = _searcher(
            {
                "q": (_ev("a"),),
                "follow one": (_ev("b"), _ev("a")),  # 'a' is a duplicate
                "follow two": (_ev("c"),),
            },
            [QueryPlan(queries=("follow one", "follow two"))],
        )
        results = agentic.as_of_search("q", as_of=AS_OF)
        assert {ev.source_id for ev in results} == {"a", "b", "c"}
        assert inner.call_count == 3
        # The planner read the round-0 evidence before planning.
        assert planner.calls[0]["n_evidence"] == 1

    def test_seed_inner_serves_round_zero_only(self) -> None:
        seed = FixtureAsOfSearch({"q": (_ev("seed-only"),)})
        follow = FixtureAsOfSearch({"f": (_ev("follow"),)})
        agentic = AgenticAsOfSearcher(
            inner=follow,
            planner=FixtureQueryPlanner([QueryPlan(queries=("f",))]),
            seed_inner=seed,
        )
        results = agentic.as_of_search("q", as_of=AS_OF)
        assert {ev.source_id for ev in results} == {"seed-only", "follow"}
        assert [q for q, _ in seed.queries] == ["q"]
        assert [q for q, _ in follow.queries] == ["f"]

    def test_seed_inner_defaults_to_inner(self) -> None:
        inner = FixtureAsOfSearch({"q": (_ev("a"),)})
        agentic = AgenticAsOfSearcher(inner=inner, planner=FixtureQueryPlanner())
        agentic.as_of_search("q", as_of=AS_OF)
        assert inner.call_count == 1

    def test_seed_inner_leakage_recheck_still_enforced(self) -> None:
        leaky = Evidence(
            snippet="future",
            source="hosted",
            source_id="http://leak",
            knowledge_time=datetime(2025, 1, 1, tzinfo=UTC),
            score=0.9,
        )
        agentic = AgenticAsOfSearcher(
            inner=FixtureAsOfSearch(),
            planner=FixtureQueryPlanner(),
            seed_inner=FixtureAsOfSearch({"q": (leaky,)}),
        )
        with pytest.raises(RuntimeError, match="leakage"):
            agentic.as_of_search("q", as_of=AS_OF)

    def test_max_rounds_exhausts_even_with_an_eager_planner(self) -> None:
        agentic, inner, planner = _searcher(
            {},
            [QueryPlan(queries=("q1",)), QueryPlan(queries=("q2",))],
            max_rounds=2,
        )
        agentic.as_of_search("seed", as_of=AS_OF)
        # Rounds 0 and 1 run; the planner's second plan never gets a round.
        assert [q for q, _ in inner.queries] == ["seed", "q1"]
        assert planner.call_count == 1

    def test_planner_stop_ends_the_loop(self) -> None:
        agentic, inner, _ = _searcher(
            {"q": (_ev("a"),)},
            [QueryPlan(queries=("never run",), stop=True)],
            max_rounds=5,
        )
        agentic.as_of_search("q", as_of=AS_OF)
        assert inner.call_count == 1

    def test_empty_plan_ends_the_loop(self) -> None:
        agentic, inner, _ = _searcher({"q": (_ev("a"),)}, [QueryPlan()], max_rounds=5)
        agentic.as_of_search("q", as_of=AS_OF)
        assert inner.call_count == 1

    def test_total_query_budget_is_a_hard_cap(self) -> None:
        agentic, inner, _ = _searcher(
            {},
            [
                QueryPlan(queries=("q1", "q2")),
                QueryPlan(queries=("q3", "q4")),
                QueryPlan(queries=("q5", "q6")),
            ],
            max_rounds=10,
            max_queries_total=3,
        )
        agentic.as_of_search("seed", as_of=AS_OF)
        assert inner.call_count == 3  # seed + q1 + q2, then the budget is spent

    def test_per_round_cap_limits_each_plan(self) -> None:
        agentic, inner, _ = _searcher(
            {},
            [QueryPlan(queries=("q1", "q2", "q3", "q4"))],
            max_queries_per_round=2,
        )
        agentic.as_of_search("seed", as_of=AS_OF)
        assert inner.call_count == 3  # seed + first 2 of the plan
        assert [q for q, _ in inner.queries] == ["seed", "q1", "q2"]

    def test_duplicate_and_blank_queries_are_skipped(self) -> None:
        agentic, inner, _ = _searcher(
            {},
            [QueryPlan(queries=("SEED", "  ", "fresh"))],
        )
        agentic.as_of_search("seed", as_of=AS_OF)
        assert [q for q, _ in inner.queries] == ["seed", "fresh"]

    def test_every_inner_call_is_pinned_to_the_ceiling(self) -> None:
        # An adversarial planner cannot widen the knowledge window: every
        # inner call carries the caller's as-of, whatever the query says.
        agentic, inner, _ = _searcher(
            {},
            [QueryPlan(queries=("what happened after 2025?",))],
        )
        agentic.as_of_search("seed", as_of=AS_OF)
        assert inner.call_count == 2
        assert all(pinned == AS_OF for _, pinned in inner.queries)

    def test_post_as_of_evidence_from_inner_raises(self) -> None:
        leaky = Evidence(
            snippet="from the future",
            source="hosted",
            source_id="http://leak",
            knowledge_time=datetime(2025, 1, 1, tzinfo=UTC),
            score=0.9,
        )
        agentic, _, _ = _searcher({"q": (leaky,)}, [])
        with pytest.raises(RuntimeError, match="leakage"):
            agentic.as_of_search("q", as_of=AS_OF)

    def test_results_are_ranked_and_truncated(self) -> None:
        agentic, _, _ = _searcher(
            {"q": tuple(_ev(f"e{i}", score=i / 10) for i in range(6))},
            [],
            max_evidence=3,
        )
        results = agentic.as_of_search("q", as_of=AS_OF)
        assert [ev.source_id for ev in results] == ["e5", "e4", "e3"]

    def test_trace_records_rounds_queries_and_novelty(self) -> None:
        agentic, _, _ = _searcher(
            {"q": (_ev("a"),), "follow": (_ev("a"), _ev("b"))},
            [QueryPlan(queries=("follow",))],
        )
        agentic.as_of_search("q", as_of=AS_OF)
        assert agentic.last_run_trace == (
            {"round": 0, "query": "q", "n_results": 1, "n_new": 1},
            {"round": 1, "query": "follow", "n_results": 2, "n_new": 1},
        )

    def test_deterministic_replay(self) -> None:
        def build() -> AgenticAsOfSearcher:
            agentic, _, _ = _searcher(
                {"q": (_ev("a"),), "f": (_ev("b"),)}, [QueryPlan(queries=("f",))]
            )
            return agentic

        first = build().as_of_search("q", as_of=AS_OF)
        second = build().as_of_search("q", as_of=AS_OF)
        assert first == second

    def test_search_config_serializes_budgets_and_planner(self) -> None:
        agentic, _, _ = _searcher({}, [], max_rounds=2, max_queries_total=5)
        assert agentic.search_config == (
            "agentic|rounds=2|queries=5|per_round=3|max_evidence=40"
            "|planner=fixture-planner-v1:query-planner-v1"
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_rounds": 0},
            {"max_queries_total": 0},
            {"max_queries_per_round": 0},
            {"max_evidence": 0},
        ],
    )
    def test_invalid_budgets_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            AgenticAsOfSearcher(inner=FixtureAsOfSearch(), planner=FixtureQueryPlanner(), **kwargs)


class _FixedTransport:
    """Fake boto converse transport returning a fixed JSON body (per repo idiom)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.users: list[str] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.users.append(kwargs["messages"][0]["content"][0]["text"])
        return {"output": {"message": {"content": [{"text": self.text}]}}}


def _planner_client(payload: str) -> tuple[BedrockQueryPlannerLLM, _FixedTransport]:
    transport = _FixedTransport(payload)
    client = BedrockStructuredClient(
        model_id="fake-model", client=transport, config=LLMConfig(max_retries=1)
    )
    return BedrockQueryPlannerLLM(client), transport


class TestBedrockQueryPlannerLLM:
    def test_parses_queries_and_stop(self) -> None:
        planner, transport = _planner_client('{"queries": ["alpha", "beta"], "stop": false}')
        plan = planner.plan(question="q", evidence=(_ev("a"),), round_index=1, max_queries=3)
        assert plan == QueryPlan(queries=("alpha", "beta"), stop=False)
        assert planner.model_version == "fake-model"
        assert planner.prompt_version == "query-planner-v1"
        assert "Question: q" in transport.users[0]
        assert "snippet a" in transport.users[0]

    def test_no_evidence_is_stated(self) -> None:
        planner, transport = _planner_client('{"queries": [], "stop": true}')
        plan = planner.plan(question="q", evidence=(), round_index=1, max_queries=3)
        assert plan.stop is True
        assert "none" in transport.users[0]

    @pytest.mark.parametrize(
        "payload",
        ['{"queries": "not a list"}', "{}", '{"queries": [1, "", null]}'],
    )
    def test_malformed_payload_degrades_to_empty_plan(self, payload: str) -> None:
        planner, _ = _planner_client(payload)
        plan = planner.plan(question="q", evidence=(), round_index=1, max_queries=3)
        assert plan.queries == ()

    def test_truncates_to_max_queries(self) -> None:
        planner, _ = _planner_client('{"queries": ["a", "b", "c", "d"]}')
        plan = planner.plan(question="q", evidence=(), round_index=1, max_queries=2)
        assert plan.queries == ("a", "b")

    def test_satisfies_protocol(self) -> None:
        planner, _ = _planner_client("{}")
        assert isinstance(planner, QueryPlannerLLM)
