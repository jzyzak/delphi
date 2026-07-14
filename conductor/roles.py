"""DELPHI forecasting role set + contracts (C8.1).

The eight roles the conductor deploys (CLAUDE.md §4), each with an explicit
access/visibility list over a shared blackboard. The blackboard vocabulary is the
set of fields a forecast run produces; a role may only read/write the fields its
contract names, which is what an orchestrator records to prove no role saw the
future. These are declarative *contracts*; the heuristic orchestrator (C8.2)
binds concrete stage callables to them.
"""

from __future__ import annotations

from enum import StrEnum

from core.agents import AccessList, RoleContract

__all__ = ["BLACKBOARD_FIELDS", "ROLE_CONTRACTS", "RoleId"]


class RoleId(StrEnum):
    """The eight conductor roles (CLAUDE.md §4)."""

    RESEARCHER = "researcher"
    REFERENCE_CLASS = "reference_class"
    DECOMPOSER = "decomposer"
    ESTIMATOR = "estimator"
    RED_TEAM = "red_team"
    VERIFIER = "verifier"
    AGGREGATOR = "aggregator"
    CALIBRATOR = "calibrator"


# The shared blackboard vocabulary a forecast run reads/writes.
BLACKBOARD_FIELDS: frozenset[str] = frozenset(
    {
        "question",
        "as_of",
        "evidence",
        "base_rate",
        "decomposition",
        "ensemble",
        "reconciled",
        "red_team",
        "verdict",
        "calibrated",
    }
)


def _contract(
    role_id: RoleId, name: str, description: str, *, reads: set[str], writes: set[str]
) -> RoleContract:
    unknown = (reads | writes) - BLACKBOARD_FIELDS
    if unknown:  # pragma: no cover - guards contract definitions below
        msg = f"role {role_id} references unknown blackboard fields: {sorted(unknown)}"
        raise ValueError(msg)
    return RoleContract(
        role_id=role_id.value,
        name=name,
        description=description,
        access=AccessList(reads=frozenset(reads), writes=frozenset(writes)),
    )


ROLE_CONTRACTS: dict[RoleId, RoleContract] = {
    RoleId.RESEARCHER: _contract(
        RoleId.RESEARCHER,
        "Researcher",
        "Retrieve as-of evidence for the question (never past the ceiling).",
        reads={"question", "as_of"},
        writes={"evidence"},
    ),
    RoleId.REFERENCE_CLASS: _contract(
        RoleId.REFERENCE_CLASS,
        "Reference-class",
        "Establish the reference class and its base rate from as-of evidence.",
        reads={"question", "as_of", "evidence"},
        writes={"base_rate"},
    ),
    RoleId.DECOMPOSER: _contract(
        RoleId.DECOMPOSER,
        "Decomposer",
        "Break the question into estimable sub-questions and a recomposition rule.",
        reads={"question", "base_rate"},
        writes={"decomposition"},
    ),
    RoleId.ESTIMATOR: _contract(
        RoleId.ESTIMATOR,
        "Estimator",
        "Produce the diverse method-agent draws and assemble the ensemble.",
        reads={"question", "evidence", "base_rate", "decomposition"},
        writes={"ensemble"},
    ),
    RoleId.RED_TEAM: _contract(
        RoleId.RED_TEAM,
        "Red-team",
        "Argue the strongest counter-case against the current estimate.",
        reads={"question", "as_of", "ensemble"},
        writes={"red_team"},
    ),
    RoleId.VERIFIER: _contract(
        RoleId.VERIFIER,
        "Verifier",
        "Check coherence + leakage; accept or request one revision.",
        reads={"ensemble", "reconciled", "red_team"},
        writes={"verdict"},
    ),
    RoleId.AGGREGATOR: _contract(
        RoleId.AGGREGATOR,
        "Aggregator",
        "Aggregate/extremize and reconcile disagreement via targeted as-of search.",
        reads={"ensemble"},
        writes={"reconciled"},
    ),
    RoleId.CALIBRATOR: _contract(
        RoleId.CALIBRATOR,
        "Calibrator",
        "Map through the recalibrator + extremization and quantify uncertainty.",
        reads={"reconciled"},
        writes={"calibrated"},
    ),
}
