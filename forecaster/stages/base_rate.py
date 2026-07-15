"""Reference-class / base-rate stage (C4.1).

The anchor of every forecast (CLAUDE.md §3): find a reference class and its
frequency, as of the ceiling, and bind the cited evidence to the resulting prior
for provenance. Reasoning is done through an injected :class:`StructuredLLM`; the
prior is clamped to the open unit interval so downstream log-odds math is stable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.forecast.search import Evidence
from core.pit.models import ensure_utc
from intake.llm import StructuredLLM

__all__ = [
    "BaseRateEstimate",
    "estimate_base_rate",
]

_PRIOR_EPS = 1e-4
_MAX_EVIDENCE = 12
_SNIPPET_MAX = 300

_SYSTEM = (
    "You are a superforecaster establishing a reference class and its base rate. "
    "Given a question, an as-of date, and evidence known as of that date, name the "
    "most apt reference class and estimate the historical frequency of the event "
    "within it. ALWAYS commit to a concrete reference class and a numeric "
    "frequency, even with no retrieved evidence: draw on your knowledge of "
    "history before the as-of date. 0.5 is almost never a true base rate; "
    "prefer the honest frequency of the reference class. Respond with ONLY a "
    'JSON object of the form {"reference_class": "...", "base_rate": p, '
    '"rationale": "...", "citations": ["source_id", ...]} where p is in [0, 1]. '
    "Cite only the given source ids. Do not include prose outside the JSON object."
)


class BaseRateEstimate(BaseModel):
    """A reference class and its as-of frequency, with bound citations."""

    model_config = ConfigDict(frozen=True)

    prior: float = Field(gt=0.0, lt=1.0)
    reference_class: str
    rationale: str = ""
    citations: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)


def _clamp_prior(value: float) -> float:
    return min(max(value, _PRIOR_EPS), 1.0 - _PRIOR_EPS)


def _render_evidence(evidence: Sequence[Evidence]) -> str:
    if not evidence:
        return "No evidence retrieved as of the ceiling."
    lines = [
        f"- [{ev.source_id}] ({ev.knowledge_time.date().isoformat()}) {ev.snippet[:_SNIPPET_MAX]}"
        for ev in evidence[:_MAX_EVIDENCE]
    ]
    return "\n".join(lines)


def estimate_base_rate(
    question: str,
    evidence: Sequence[Evidence],
    *,
    llm: StructuredLLM,
    as_of: datetime,
) -> BaseRateEstimate:
    """Elicit a reference class + prior for ``question`` from as-of ``evidence``."""
    ceiling = ensure_utc(as_of)
    user = (
        f"Question: {question}\n"
        f"As-of date (knowledge ceiling): {ceiling.isoformat()}\n\n"
        f"Evidence known as of the ceiling:\n{_render_evidence(evidence)}"
    )
    payload = llm.invoke_structured(system=_SYSTEM, user=user)

    raw_rate = payload.get("base_rate")
    prior = 0.5
    defaulted = True
    if raw_rate is not None:
        try:
            value = float(raw_rate)
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            prior = _clamp_prior(value)
            defaulted = False

    evidence_ids = {ev.source_id for ev in evidence}
    raw_citations = payload.get("citations", [])
    citations: tuple[str, ...]
    if isinstance(raw_citations, list):
        cited = tuple(str(c) for c in raw_citations if str(c) in evidence_ids)
        citations = cited or tuple(sorted(evidence_ids))
    else:
        citations = tuple(sorted(evidence_ids))

    reference_class = str(payload.get("reference_class") or "unspecified reference class")
    rationale = str(payload.get("rationale") or "")
    return BaseRateEstimate(
        prior=prior,
        reference_class=reference_class,
        rationale=rationale,
        citations=citations,
        provenance={
            "as_of": ceiling.isoformat(),
            "defaulted_prior": defaulted,
            "raw_base_rate": raw_rate,
            "n_evidence": len(evidence),
        },
    )
