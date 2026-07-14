"""Agent template + role-contract primitives (CLAUDE.md §5).

A :class:`RoleContract` binds a role's identity to an explicit access/visibility
list — the fields it may read and the fields it may write on a shared blackboard.
Making visibility explicit is what lets an orchestrator (heuristic now, learned
later) reason about — and record — exactly what each role could see, so a
conductor can never silently grant a role access to the future (§4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["AccessList", "Role", "RoleContract"]


@dataclass(frozen=True)
class AccessList:
    """The blackboard fields a role may read and write (visibility control)."""

    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()

    def may_read(self, field: str) -> bool:
        return field in self.reads

    def may_write(self, field: str) -> bool:
        return field in self.writes


@dataclass(frozen=True)
class RoleContract:
    """A role's identity + description + access list."""

    role_id: str
    name: str
    description: str
    access: AccessList


@runtime_checkable
class Role(Protocol):
    """An identified actor in an orchestration."""

    @property
    def role_id(self) -> str: ...

    @property
    def contract(self) -> RoleContract: ...
