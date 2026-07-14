"""Shared agent template + role contracts (CLAUDE.md §5).

Domain-agnostic: a :class:`Role` is an identified actor in an orchestration, and
its :class:`RoleContract` declares what blackboard fields it may read and write
(the access/visibility list). Applications build concrete role sets on top of
these primitives; nothing domain-specific lives here.
"""

from __future__ import annotations

from core.agents.base import AccessList, Role, RoleContract

__all__ = ["AccessList", "Role", "RoleContract"]
