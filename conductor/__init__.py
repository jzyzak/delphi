"""Heuristic conductor (ships) + learned conductor scaffolding (§4).

App layer: a hand-designed, deterministic orchestration over the role set that
produces scored ``(question, workflow, evidence, forecast, resolution, score)``
tuples for the registry corpus — the training data for the later learned stage.
"""

from __future__ import annotations

from conductor.corpus import CorpusStore, CorpusTuple, CorpusWriter, InMemoryCorpusStore
from conductor.heuristic import (
    ConductorResult,
    FixtureRedTeamLLM,
    HeuristicConductor,
    RedTeamLLM,
    WorkflowStep,
    WorkflowTrace,
)
from conductor.roles import BLACKBOARD_FIELDS, ROLE_CONTRACTS, RoleId

__all__ = [
    "BLACKBOARD_FIELDS",
    "ROLE_CONTRACTS",
    "ConductorResult",
    "CorpusStore",
    "CorpusTuple",
    "CorpusWriter",
    "FixtureRedTeamLLM",
    "HeuristicConductor",
    "InMemoryCorpusStore",
    "RedTeamLLM",
    "RoleId",
    "WorkflowStep",
    "WorkflowTrace",
]
