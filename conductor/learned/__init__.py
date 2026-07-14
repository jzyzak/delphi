"""Stage-2 learned conductor scaffold (Appendix, pure upside — CLAUDE.md §4).

This is the *interface + governance* for the learned stage, not a shipped model:
a corpus export, a proper-score reward, an as-of-safe training loop, a
deterministic policy trainer (a stand-in for the RL trainer), and a
holdout-gated rollout. The three required adaptations are enforced here:
(1) reward is on **proper-score improvement**, never binary correctness;
(2) the loop is **as-of safe** — no example may reference the future;
(3) provenance is **preserved and anonymized** ("Model 0, Model 1…") so the
policy learns forecasting strength from reward, not brand priors. Never-beating
the heuristic is a fine outcome — the heuristic conductor is already the product.
"""

from __future__ import annotations

from conductor.learned.export import (
    TrainingExample,
    anonymize_provenance,
    export_corpus,
)
from conductor.learned.reward import as_of_safe, assert_as_of_safe, proper_score_reward
from conductor.learned.rollout import RolloutDecision, holdout_gated_rollout
from conductor.learned.trainer import (
    GreedyPolicyTrainer,
    LearnedConductorPolicy,
    RLTrainer,
)

__all__ = [
    "GreedyPolicyTrainer",
    "LearnedConductorPolicy",
    "RLTrainer",
    "RolloutDecision",
    "TrainingExample",
    "anonymize_provenance",
    "as_of_safe",
    "assert_as_of_safe",
    "export_corpus",
    "holdout_gated_rollout",
    "proper_score_reward",
]
