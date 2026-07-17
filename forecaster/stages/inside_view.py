"""Inside-view modeling + ensemble assembly (C4.3).

Produces a diverse set of independent draws by prompting several *method-agents*
— base-rate-heavy, inside-view-heavy, market-anchored, extrapolation — over a
shared as-of context, then aggregates them with the robust core ensemble
(CLAUDE.md §3.4: decorrelated estimators are what make aggregation work).

A Bayesian assembly path (C4.3.d) is also exposed: it combines per-run evidence
log-likelihood-ratios with the base-rate prior via the core Bayesian ensemble.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime

import numpy as np

from core.forecast.bayesian import EvidenceLikelihoodLLM, elicit_and_build_bayesian_ensemble
from core.forecast.ensemble import Aggregator, EnsembleForecast, build_ensemble
from core.forecast.llm import ForecastLLM, ForecastRequest
from core.forecast.search import Evidence
from core.pit.models import ensure_utc
from core.registry.fingerprint import content_hash
from forecaster.stages.base_rate import BaseRateEstimate
from forecaster.stages.decompose import Decomposition

__all__ = [
    "METHOD_AGENTS",
    "assemble_bayesian_ensemble",
    "assemble_ensemble",
    "build_draw_requests",
    "build_forecast_content",
    "build_subset_draw_requests",
]

METHOD_AGENTS: tuple[str, ...] = (
    "base_rate_heavy",
    "inside_view_heavy",
    "market_anchored",
    "extrapolation",
)

_AGENT_PROMPTS: dict[str, str] = {
    "base_rate_heavy": (
        "Anchor hard on the reference-class base rate. Move off it only for "
        "decisive, as-of evidence. Report your probability of the event."
    ),
    "inside_view_heavy": (
        "Reason from the specifics of this case and the as-of evidence, treating "
        "the base rate as a weak prior. Report your probability of the event."
    ),
    "market_anchored": (
        "Weigh any market, crowd, or expert-consensus signal in the evidence most "
        "heavily — especially a [market_freeze] item, which is the market's own "
        "estimate of this exact question at the forecast time; treat it as the "
        "anchor and move off it only for concrete as-of evidence the market "
        "plausibly missed. Report your probability of the event."
    ),
    "extrapolation": (
        "Extrapolate the most recent as-of trend forward to the resolution date. "
        "Report your probability of the event."
    ),
}

# Appended to every method-agent prompt. Targets a measured failure mode:
# confident directional calls on questions that are near-martingales (e.g.
# "will this stock close above its current price on date X"), where the honest
# probability sits near the base rate and the model's inside view adds ~nothing.
_SHARED_DISCIPLINE = (
    " Discipline: if the question asks whether a traded asset's price (or any "
    "near-random-walk series) will finish above its current level on a future "
    "date, the honest probability is close to the base rate / market anchor — "
    "a confident directional call there requires decisive case-specific as-of "
    "evidence, not narrative. A [series_estimator] evidence item is a "
    "deterministic base rate computed arithmetically from the series' own "
    "history: for series-direction questions, anchor on it."
)

DEFAULT_SNIPPET_MAX = 500
DEFAULT_MAX_EVIDENCE = 20


def build_forecast_content(
    question: str,
    base_rate: BaseRateEstimate,
    decomposition: Decomposition,
    evidence: Sequence[Evidence],
    *,
    max_evidence: int = DEFAULT_MAX_EVIDENCE,
    snippet_max: int = DEFAULT_SNIPPET_MAX,
) -> str:
    """Assemble the shared as-of context every method-agent draw reads."""
    lines = [
        f"Question: {question}",
        f"Reference class: {base_rate.reference_class}",
        f"Base rate (prior): {base_rate.prior:.4f}",
    ]
    if base_rate.rationale:
        lines.append(f"Base-rate rationale: {base_rate.rationale}")
    if decomposition.sub_questions:
        lines.append("Sub-questions:")
        lines.extend(f"  - {s.text}" for s in decomposition.sub_questions)
        lines.append(f"Recomposition rule: {decomposition.rule}")
    if evidence:
        lines.append("Evidence (as of the ceiling):")
        for ev in evidence[:max_evidence]:
            lines.append(f"  - [{ev.source_id}] {ev.snippet[:snippet_max]}")
    else:
        lines.append(
            "Evidence: none retrieved as of the ceiling. Do NOT treat missing "
            "retrieval as maximal uncertainty: anchor on the reference-class "
            "base rate above and on what you already know from before the "
            "as-of date, and commit to your best calibrated estimate rather "
            "than defaulting toward 0.5."
        )
    return "\n".join(lines)


def build_draw_requests(
    *,
    content: str,
    agents: Sequence[str] = METHOD_AGENTS,
    runs_per_agent: int = 1,
) -> tuple[ForecastRequest, ...]:
    """Build one shared-content request per (agent, run); run_index is global."""
    if runs_per_agent < 1:
        msg = "runs_per_agent must be >= 1."
        raise ValueError(msg)
    if not agents:
        msg = "at least one method-agent is required."
        raise ValueError(msg)
    digest = content_hash(content)
    requests: list[ForecastRequest] = []
    run_index = 0
    for agent in agents:
        prompt = _AGENT_PROMPTS.get(agent, _AGENT_PROMPTS["inside_view_heavy"])
        for _ in range(runs_per_agent):
            requests.append(
                ForecastRequest(
                    content=content,
                    content_hash=digest,
                    run_index=run_index,
                    prompt=f"[method-agent: {agent}] {prompt}{_SHARED_DISCIPLINE}",
                )
            )
            run_index += 1
    return tuple(requests)


def _subset_seed(digest: str, run_index: int) -> int:
    """Stable per-draw seed derived from the full-content hash and run index."""
    raw = hashlib.sha256(f"{digest}:{run_index}".encode()).digest()
    return int.from_bytes(raw[:8], "big")


def build_subset_draw_requests(
    *,
    question: str,
    base_rate: BaseRateEstimate,
    decomposition: Decomposition,
    evidence: Sequence[Evidence],
    agents: Sequence[str] = METHOD_AGENTS,
    runs_per_agent: int = 1,
    subset_fraction: float = 1.0,
    max_evidence: int = DEFAULT_MAX_EVIDENCE,
    snippet_max: int = DEFAULT_SNIPPET_MAX,
) -> tuple[ForecastRequest, ...]:
    """Per-draw seeded evidence subsets: decorrelated draws beyond prompt steering.

    Each (agent, run) draw reads a deterministic ~``subset_fraction`` sample of
    the evidence (original order preserved), so draws differ in *evidence*, not
    just method prompt — the decorrelation §3.4 asks the ensemble to have. With
    ``subset_fraction >= 1`` or fewer than two evidence items this degrades to
    the shared-content :func:`build_draw_requests` path exactly.
    """
    if not 0.0 < subset_fraction <= 1.0:
        msg = f"subset_fraction must be in (0, 1], got {subset_fraction!r}"
        raise ValueError(msg)
    full_content = build_forecast_content(
        question,
        base_rate,
        decomposition,
        evidence,
        max_evidence=max_evidence,
        snippet_max=snippet_max,
    )
    if subset_fraction >= 1.0 or len(evidence) < 2:
        return build_draw_requests(
            content=full_content, agents=agents, runs_per_agent=runs_per_agent
        )
    if runs_per_agent < 1:
        msg = "runs_per_agent must be >= 1."
        raise ValueError(msg)
    if not agents:
        msg = "at least one method-agent is required."
        raise ValueError(msg)
    digest = content_hash(full_content)
    n_keep = max(1, round(subset_fraction * len(evidence)))
    requests: list[ForecastRequest] = []
    run_index = 0
    for agent in agents:
        prompt = _AGENT_PROMPTS.get(agent, _AGENT_PROMPTS["inside_view_heavy"])
        for _ in range(runs_per_agent):
            rng = np.random.default_rng(_subset_seed(digest, run_index))
            keep = sorted(rng.choice(len(evidence), size=n_keep, replace=False).tolist())
            subset = [evidence[i] for i in keep]
            content = build_forecast_content(
                question,
                base_rate,
                decomposition,
                subset,
                max_evidence=max_evidence,
                snippet_max=snippet_max,
            )
            requests.append(
                ForecastRequest(
                    content=content,
                    content_hash=content_hash(content),
                    run_index=run_index,
                    prompt=f"[method-agent: {agent}] {prompt}{_SHARED_DISCIPLINE}",
                )
            )
            run_index += 1
    return tuple(requests)


def assemble_ensemble(
    llm: ForecastLLM,
    requests: Sequence[ForecastRequest],
    *,
    aggregator: Aggregator = "median",
    knowledge_time: datetime,
) -> EnsembleForecast:
    """Elicit one draw per request and aggregate them into an ensemble."""
    if not requests:
        msg = "requests must be non-empty."
        raise ValueError(msg)
    draws = llm.forecast_batch(list(requests))
    return build_ensemble(draws, aggregator=aggregator, knowledge_time=ensure_utc(knowledge_time))


def assemble_bayesian_ensemble(
    llm: EvidenceLikelihoodLLM,
    *,
    content: str,
    base_rate: BaseRateEstimate,
    knowledge_time: datetime,
    n: int = 10,
    aggregator: Aggregator = "median",
) -> EnsembleForecast:
    """Bayesian path: combine per-run evidence log-LRs with the base-rate prior."""
    result = elicit_and_build_bayesian_ensemble(
        llm,
        content=content,
        content_hash=content_hash(content),
        base_rate=base_rate.prior,
        prompt="Estimate the evidence log-likelihood-ratio for the event.",
        knowledge_time=ensure_utc(knowledge_time),
        n=n,
        aggregator=aggregator,
    )
    return result.ensemble
