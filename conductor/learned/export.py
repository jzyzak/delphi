"""Corpus export (A.1) — the training set from the registry corpus.

Turns scored corpus tuples (C8.3) into flat training examples: per-question
features, the workflow route the heuristic emitted, and the realized
proper-score target. Only *resolved* tuples are exported (an unresolved forecast
has no target yet). Model provenance is anonymized to "Model 0, Model 1…" so the
learned policy cannot pick up brand priors (CLAUDE.md §4).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from conductor.corpus import CorpusTuple

__all__ = ["TrainingExample", "anonymize_provenance", "export_corpus"]


@dataclass(frozen=True)
class TrainingExample:
    """One flat training row: features + route + proper-score target."""

    question_id: str
    domain: str
    features: dict[str, float]
    route: tuple[str, ...]
    proper_score: float
    provenance: dict[str, Any] = field(default_factory=dict)


def anonymize_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Replace model-version brands with stable anonymous ids ("model_0"…).

    Every distinct ``model_version`` value found (at any nesting level) is mapped
    to ``model_{i}`` by sorted first-appearance order, so the mapping is
    deterministic and brand-free.
    """
    versions: set[str] = set()

    def _collect(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key == "model_version" and isinstance(value, str):
                    versions.add(value)
                else:
                    _collect(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _collect(item)

    _collect(provenance)
    mapping = {v: f"model_{i}" for i, v in enumerate(sorted(versions))}

    def _rewrite(node: Any) -> Any:
        if isinstance(node, Mapping):
            rewritten: dict[str, Any] = {}
            for key, value in node.items():
                if key == "model_version" and value in mapping:
                    rewritten[key] = mapping[value]
                else:
                    rewritten[key] = _rewrite(value)
            return rewritten
        if isinstance(node, list):
            return [_rewrite(item) for item in node]
        if isinstance(node, tuple):
            return tuple(_rewrite(item) for item in node)
        return node

    return _rewrite(dict(provenance))


def _features(tuple_: CorpusTuple) -> dict[str, float]:
    n_evidence = sum(len(es.items) for es in tuple_.evidence)
    workflow = tuple_.workflow
    revisions = float(workflow.get("revisions", 0))
    route_len = float(len(workflow.get("route", [])))
    return {
        "n_evidence": float(n_evidence),
        "revisions": revisions,
        "route_len": route_len,
    }


def export_corpus(corpus: Iterable[CorpusTuple]) -> tuple[TrainingExample, ...]:
    """Export resolved, scored corpus tuples into training examples."""
    examples: list[TrainingExample] = []
    for tuple_ in corpus:
        if tuple_.proper_score is None:
            continue  # unresolved: no target yet
        route = tuple(str(r) for r in tuple_.workflow.get("route", []))
        examples.append(
            TrainingExample(
                question_id=tuple_.question.question_id,
                domain=tuple_.question.domain,
                features=_features(tuple_),
                route=route,
                proper_score=tuple_.proper_score,
                provenance=anonymize_provenance(tuple_.forecast.model_provenance),
            )
        )
    return tuple(examples)
