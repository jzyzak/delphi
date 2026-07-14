"""Meta-layer governance primitives that are domain-agnostic.

Only the boundary check, config, and holdout governor are shared. The productivity
metrics, oversight, director, and their monitoring-coupled types are rebuilt in the
app layer around forecasting (proper-score) semantics.
"""

from core.orchestration.meta.boundary import (
    MetaBoundaryViolation,
    find_meta_boundary_violations,
    run_meta_boundary_check,
)
from core.orchestration.meta.config import MetaConfig, default_meta_config
from core.orchestration.meta.holdout import (
    HoldoutAccessRecord,
    HoldoutBudgetExhausted,
    HoldoutChainVerification,
    HoldoutGovernor,
    HoldoutSource,
    HoldoutView,
    InMemoryHoldoutGovernor,
    StaticHoldoutSource,
)

__all__ = [
    "HoldoutAccessRecord",
    "HoldoutBudgetExhausted",
    "HoldoutChainVerification",
    "HoldoutGovernor",
    "HoldoutSource",
    "HoldoutView",
    "InMemoryHoldoutGovernor",
    "MetaBoundaryViolation",
    "MetaConfig",
    "StaticHoldoutSource",
    "default_meta_config",
    "find_meta_boundary_violations",
    "run_meta_boundary_check",
]
