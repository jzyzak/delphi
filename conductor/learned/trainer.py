"""Learned-conductor trainer (A.2) — deterministic RL stand-in.

The papers train a Conductor-style model with RL to emit a natural-language
workflow over the role set. Shipping that model is future work; what we build
now is the *trainer interface* and a deterministic, dependency-light baseline
trainer so the rest of the machinery (export → reward → rollout) is exercisable
end to end. ``GreedyPolicyTrainer`` selects, among the routes actually observed
in the corpus, the one whose mean **proper score** is lowest (best), scoring
candidates by proper-score reward against the corpus mean (never correctness,
CLAUDE.md §4). Provenance is preserved on the learned policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Protocol, runtime_checkable

from conductor.learned.export import TrainingExample
from conductor.learned.reward import proper_score_reward

__all__ = ["GreedyPolicyTrainer", "LearnedConductorPolicy", "RLTrainer"]


@dataclass(frozen=True)
class LearnedConductorPolicy:
    """A trained policy: the route it emits + its expected proper score.

    This scaffold emits a single best route regardless of context; a real
    learned conductor would emit a per-question workflow. ``provenance`` records
    how the policy was fit so the registry can preserve it (CLAUDE.md §4).
    """

    route: tuple[str, ...]
    expected_score: float
    reward: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def emit_route(self, example: TrainingExample | None = None) -> tuple[str, ...]:
        """Emit the workflow route for a question (context-independent here)."""
        _ = example
        return self.route


@runtime_checkable
class RLTrainer(Protocol):
    """Trains a :class:`LearnedConductorPolicy` from exported corpus examples."""

    def train(self, examples: Sequence[TrainingExample]) -> LearnedConductorPolicy:
        """Fit a policy on ``examples`` (must be non-empty)."""
        ...


class GreedyPolicyTrainer:
    """Deterministic proper-score-greedy trainer (an RL trainer stand-in)."""

    def train(self, examples: Sequence[TrainingExample]) -> LearnedConductorPolicy:
        if not examples:
            msg = "cannot train on an empty corpus export."
            raise ValueError(msg)

        baseline = fmean(ex.proper_score for ex in examples)

        by_route: dict[tuple[str, ...], list[float]] = {}
        for ex in examples:
            by_route.setdefault(ex.route, []).append(ex.proper_score)

        # Deterministic tie-break: (mean score, route) so the same corpus always
        # yields the same policy.
        best_route, best_scores = min(by_route.items(), key=lambda item: (fmean(item[1]), item[0]))
        expected = fmean(best_scores)
        reward = proper_score_reward(baseline=baseline, candidate=expected)

        return LearnedConductorPolicy(
            route=best_route,
            expected_score=expected,
            reward=reward,
            provenance={
                "trainer": "greedy_proper_score",
                "baseline_mean": baseline,
                "n_examples": len(examples),
                "n_routes": len(by_route),
            },
        )
