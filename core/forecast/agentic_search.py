"""Agentic multi-query iterative as-of search (CLAUDE.md §5 ``core/forecast``).

Wraps any :class:`~core.forecast.search.AsOfSearcher` in an LLM-directed
retrieval loop: a query planner emits seed queries, reads the accumulated
evidence, and issues follow-up queries until it stops or a hard budget runs
out. Every underlying read goes through the inner searcher — which enforces
the as-of ceiling (§2.1) — and the wrapper re-checks every returned item, so
no planner-crafted query can widen the knowledge window. Search quality
dominates forecast accuracy (§1); this is where breadth comes from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from common.llm import StructuredLLMClient
from core.forecast.search import AsOfSearcher, Evidence
from core.pit.models import ensure_utc

__all__ = [
    "AgenticAsOfSearcher",
    "BedrockQueryPlannerLLM",
    "FixtureQueryPlanner",
    "QueryPlan",
    "QueryPlannerLLM",
    "rank_evidence",
]

QUERY_PLANNER_PROMPT_VERSION = "query-planner-v1"

DEFAULT_MAX_ROUNDS = 3
DEFAULT_MAX_QUERIES_TOTAL = 8
DEFAULT_MAX_QUERIES_PER_ROUND = 3
DEFAULT_MAX_EVIDENCE = 40


@dataclass(frozen=True)
class QueryPlan:
    """One planner turn: follow-up queries to run, or a stop signal."""

    queries: tuple[str, ...] = ()
    stop: bool = False


@runtime_checkable
class QueryPlannerLLM(Protocol):
    """Mockable LLM seam that plans search queries from evidence-so-far."""

    @property
    def model_version(self) -> str: ...

    @property
    def prompt_version(self) -> str: ...

    def plan(
        self,
        *,
        question: str,
        evidence: Sequence[Evidence],
        round_index: int,
        max_queries: int,
    ) -> QueryPlan:
        """Emit up to ``max_queries`` follow-up queries, or stop."""
        ...


class FixtureQueryPlanner:
    """Deterministic query planner for tests (no network).

    Plays back one scripted :class:`QueryPlan` per round; when the script runs
    out it stops, so a fixture can never drive an unbounded loop.
    """

    def __init__(
        self,
        plans: Sequence[QueryPlan] = (),
        *,
        model_version: str = "fixture-planner-v1",
        prompt_version: str = QUERY_PLANNER_PROMPT_VERSION,
    ) -> None:
        self._plans = tuple(plans)
        self._model_version = model_version
        self._prompt_version = prompt_version
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def plan(
        self,
        *,
        question: str,
        evidence: Sequence[Evidence],
        round_index: int,
        max_queries: int,
    ) -> QueryPlan:
        self.calls.append(
            {
                "question": question,
                "n_evidence": len(evidence),
                "round_index": round_index,
                "max_queries": max_queries,
            }
        )
        index = self.call_count
        self.call_count += 1
        if index >= len(self._plans):
            return QueryPlan(stop=True)
        return self._plans[index]


_PLANNER_SYSTEM = (
    "You plan retrieval for a forecasting system. Given a question and the "
    "evidence gathered so far, emit the next search queries that would most "
    "reduce uncertainty: fill gaps, disambiguate entities, check base rates, "
    "and seek disconfirming evidence. Respond with ONLY a JSON object "
    '{"queries": ["..."], "stop": false}. Set "stop": true when further '
    "search is unlikely to change the forecast. Never ask about events after "
    "the knowledge cutoff given in the context."
)


def _compose_planner_user(
    question: str, evidence: Sequence[Evidence], round_index: int, max_queries: int
) -> str:
    lines = [
        f"Question: {question}",
        f"Search round: {round_index}",
        f"Emit at most {max_queries} queries.",
    ]
    if evidence:
        lines.append("Evidence so far:")
        lines.extend(f"- [{ev.source_id}] {ev.snippet[:300]}" for ev in evidence[:20])
    else:
        lines.append("Evidence so far: none.")
    return "\n".join(lines)


class BedrockQueryPlannerLLM:
    """Structured-LLM-backed :class:`QueryPlannerLLM` over any provider transport."""

    def __init__(
        self,
        client: StructuredLLMClient,
        *,
        prompt_version: str = QUERY_PLANNER_PROMPT_VERSION,
        system: str = _PLANNER_SYSTEM,
    ) -> None:
        self._client = client
        self._prompt_version = prompt_version
        self._system = system

    @property
    def model_version(self) -> str:
        return self._client.model_id

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def plan(
        self,
        *,
        question: str,
        evidence: Sequence[Evidence],
        round_index: int,
        max_queries: int,
    ) -> QueryPlan:
        payload = self._client.invoke_structured(
            system=self._system,
            user=_compose_planner_user(question, evidence, round_index, max_queries),
        )
        raw_queries = payload.get("queries", [])
        queries: list[str] = []
        if isinstance(raw_queries, list):
            for item in raw_queries:
                if isinstance(item, str) and item.strip():
                    queries.append(item.strip())
        return QueryPlan(queries=tuple(queries[:max_queries]), stop=bool(payload.get("stop")))


def rank_evidence(evidence: Sequence[Evidence], *, max_items: int) -> tuple[Evidence, ...]:
    """Rank by provider score, then recency — the truncation order for context caps."""
    if max_items < 1:
        msg = f"max_items must be >= 1, got {max_items!r}"
        raise ValueError(msg)
    ranked = sorted(evidence, key=lambda ev: (-ev.score, -ev.knowledge_time.timestamp()))
    return tuple(ranked[:max_items])


@dataclass(frozen=True)
class _RoundTrace:
    """One executed query and how much it contributed (recorded for the audit trail)."""

    round_index: int
    query: str
    n_results: int
    n_new: int


@dataclass
class AgenticAsOfSearcher:
    """LLM-directed iterative search implementing the ``AsOfSearcher`` protocol.

    Round 0 executes the caller's query; each later round lets the planner read
    the accumulated evidence and emit follow-up queries (or stop).
    Hard budgets (``max_rounds``, ``max_queries_total``) bound the loop, results
    are deduplicated by ``(source, source_id)`` + snippet, and every item is
    re-checked against the as-of ceiling — a defense-in-depth on top of the
    inner searcher's own guarantee (§2.1). ``last_run_trace`` exposes the
    executed queries for the workflow trace/leakage audit.
    """

    inner: AsOfSearcher
    planner: QueryPlannerLLM
    max_rounds: int = DEFAULT_MAX_ROUNDS
    max_queries_total: int = DEFAULT_MAX_QUERIES_TOTAL
    max_queries_per_round: int = DEFAULT_MAX_QUERIES_PER_ROUND
    max_evidence: int = DEFAULT_MAX_EVIDENCE
    # Optional round-0 searcher: slow/rate-limited providers (e.g. GDELT's 6s
    # politeness interval) contribute their one high-value bounded query on the
    # seed round only, while planner follow-ups use the faster ``inner``.
    seed_inner: AsOfSearcher | None = None
    last_run_trace: tuple[Mapping[str, Any], ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            msg = f"max_rounds must be >= 1, got {self.max_rounds!r}"
            raise ValueError(msg)
        if self.max_queries_total < 1:
            msg = f"max_queries_total must be >= 1, got {self.max_queries_total!r}"
            raise ValueError(msg)
        if self.max_queries_per_round < 1:
            msg = f"max_queries_per_round must be >= 1, got {self.max_queries_per_round!r}"
            raise ValueError(msg)
        if self.max_evidence < 1:
            msg = f"max_evidence must be >= 1, got {self.max_evidence!r}"
            raise ValueError(msg)

    @property
    def search_config(self) -> str:
        """Serialized loop parameters for content-addressed cache keys."""
        return (
            f"agentic|rounds={self.max_rounds}|queries={self.max_queries_total}"
            f"|per_round={self.max_queries_per_round}|max_evidence={self.max_evidence}"
            f"|planner={self.planner.model_version}:{self.planner.prompt_version}"
        )

    def as_of_search(self, query: str, *, as_of: datetime) -> Sequence[Evidence]:
        """Run the iterative retrieval loop pinned at ``as_of``."""
        ceiling = ensure_utc(as_of)
        collected: dict[tuple[str, str, str], Evidence] = {}
        executed: set[str] = set()
        trace: list[_RoundTrace] = []
        queries_used = 0

        pending: list[str] = [query]
        for round_index in range(self.max_rounds):
            budget = self.max_queries_total - queries_used
            if budget <= 0:
                break
            if round_index > 0:
                plan = self.planner.plan(
                    question=query,
                    evidence=tuple(collected.values()),
                    round_index=round_index,
                    max_queries=min(self.max_queries_per_round, budget),
                )
                if plan.stop or not plan.queries:
                    break
                pending = list(plan.queries)
            for q in pending[: min(self.max_queries_per_round, budget)]:
                normalized = q.strip().lower()
                if not normalized or normalized in executed:
                    continue
                executed.add(normalized)
                searcher = (
                    self.seed_inner
                    if round_index == 0 and self.seed_inner is not None
                    else self.inner
                )
                results = searcher.as_of_search(q, as_of=ceiling)
                queries_used += 1
                n_new = 0
                for item in results:
                    if item.knowledge_time > ceiling:
                        msg = (
                            "inner as_of_search returned evidence dated after the "
                            "as-of ceiling (leakage)."
                        )
                        raise RuntimeError(msg)
                    key = (item.source, item.source_id, item.snippet)
                    if key not in collected:
                        collected[key] = item
                        n_new += 1
                trace.append(
                    _RoundTrace(
                        round_index=round_index, query=q, n_results=len(results), n_new=n_new
                    )
                )
        self.last_run_trace = tuple(
            {
                "round": t.round_index,
                "query": t.query,
                "n_results": t.n_results,
                "n_new": t.n_new,
            }
            for t in trace
        )
        return rank_evidence(tuple(collected.values()), max_items=self.max_evidence)
