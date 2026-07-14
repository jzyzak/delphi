"""Decomposition stage (C4.2).

Break a question into estimable sub-questions and define how they recompose
(CLAUDE.md §3). Two recomposition rules are supported: a Fermi ``product`` of
independent factor probabilities, and a ``scenario_tree`` of mutually exclusive
path probabilities that sum. ``none`` passes a single estimate through unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from intake.llm import StructuredLLM

__all__ = [
    "Decomposition",
    "RecomposeRule",
    "SubQuestion",
    "decompose_question",
    "recompose",
]

RecomposeRule = Literal["product", "scenario_tree", "none"]

_SYSTEM = (
    "You are a superforecaster decomposing a question into estimable parts. "
    "Produce sub-questions and a recomposition rule: 'product' when the event "
    "requires several independent things to all happen, 'scenario_tree' when it "
    "is one of several mutually exclusive paths, or 'none' when it is atomic. "
    "Respond with ONLY a JSON object of the form "
    '{"sub_questions": ["...", ...], "rule": "product"|"scenario_tree"|"none"}. '
    "Do not include prose outside the JSON object."
)


class SubQuestion(BaseModel):
    """One estimable component of a decomposed question."""

    model_config = ConfigDict(frozen=True)

    text: str


class Decomposition(BaseModel):
    """A set of sub-questions plus the rule that recomposes their estimates."""

    model_config = ConfigDict(frozen=True)

    sub_questions: tuple[SubQuestion, ...] = ()
    rule: RecomposeRule = "none"
    provenance: dict[str, Any] = Field(default_factory=dict)


def decompose_question(question: str, *, llm: StructuredLLM) -> Decomposition:
    """Elicit sub-questions and a recomposition rule for ``question``."""
    payload = llm.invoke_structured(system=_SYSTEM, user=f"Question: {question}")
    raw_subs = payload.get("sub_questions", [])
    subs: tuple[SubQuestion, ...] = ()
    if isinstance(raw_subs, list):
        subs = tuple(SubQuestion(text=str(s).strip()) for s in raw_subs if str(s).strip())
    raw_rule = payload.get("rule")
    rule: RecomposeRule = raw_rule if raw_rule in ("product", "scenario_tree", "none") else "none"
    return Decomposition(
        sub_questions=subs,
        rule=rule,
        provenance={"raw_rule": raw_rule, "n_sub_questions": len(subs)},
    )


def recompose(rule: RecomposeRule, values: Sequence[float]) -> float:
    """Recompose sub-question estimates into a single probability in [0, 1].

    ``product``: independent factors multiply. ``scenario_tree``: mutually
    exclusive paths sum (clamped to 1). ``none``: the first value passes through.
    """
    if not values:
        msg = "recompose requires at least one value."
        raise ValueError(msg)
    for v in values:
        if not 0.0 <= v <= 1.0:
            msg = f"recompose values must be probabilities in [0, 1], got {v!r}."
            raise ValueError(msg)
    if rule == "product":
        product = 1.0
        for v in values:
            product *= v
        return product
    if rule == "scenario_tree":
        return min(1.0, sum(values))
    return values[0]
