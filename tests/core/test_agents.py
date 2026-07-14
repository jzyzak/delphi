"""Tests for the shared agent template primitives (core/agents)."""

from __future__ import annotations

from core.agents import AccessList, Role, RoleContract


def test_access_list_read_write() -> None:
    access = AccessList(reads=frozenset({"a", "b"}), writes=frozenset({"c"}))
    assert access.may_read("a")
    assert not access.may_read("c")
    assert access.may_write("c")
    assert not access.may_write("a")


def test_access_list_defaults_empty() -> None:
    access = AccessList()
    assert not access.may_read("anything")
    assert not access.may_write("anything")


def test_role_protocol_conformance() -> None:
    contract = RoleContract(role_id="r", name="R", description="d", access=AccessList())

    class _StubRole:
        role_id = "r"

        @property
        def contract(self) -> RoleContract:
            return contract

    assert isinstance(_StubRole(), Role)
