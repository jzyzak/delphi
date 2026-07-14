"""Configuration for the Research Director meta-layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MetaConfig(BaseModel):
    """Thresholds for deterministic meta-layer policies.

    Contract: all binding retirement/focus/budget decisions derive from these
    values — not from LLM output.
    """

    model_config = ConfigDict(frozen=True)

    decay_retire_consecutive_windows: int = Field(
        default=5,
        ge=1,
        description="Minimum consecutive decay windows before strategy retirement.",
    )
    min_trials_adjusted_promotion_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum promoted/total experiment ratio for an agent.",
    )
    min_promoted_persistence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of an agent's promoted strategies still active.",
    )
    min_experiments_for_agent_eval: int = Field(
        default=5,
        ge=1,
        description="Minimum experiments before agent productivity is evaluated.",
    )
    holdout_budget: int = Field(
        default=3,
        ge=0,
        description="Hard cap on holdout accesses; each access debits one unit.",
    )
    co_decay_cluster_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of promoted strategies decaying to flag regime co-decay.",
    )
    agent_gaming_promotion_rate_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Suspiciously high promotion rate for agent-gaming detection.",
    )
    agent_gaming_max_persistence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Max persistence paired with high promotion rate to flag gaming.",
    )
    agent_gaming_min_experiments: int = Field(
        default=10,
        ge=1,
        description="Minimum experiments before agent-gaming heuristics apply.",
    )
    agent_catalog: dict[str, str] = Field(
        default_factory=lambda: {
            "agent.geopolitics": "us_elections",
            "agent.macro": "macro_indicators",
        },
        description="agent_id -> niche mapping for research-focus dispatch.",
    )


def default_meta_config() -> MetaConfig:
    """Return the default Research Director configuration."""
    return MetaConfig()
