"""Tests for the DELPHI role set + contracts (C8.1)."""

from __future__ import annotations

from conductor.roles import BLACKBOARD_FIELDS, ROLE_CONTRACTS, RoleId


def test_all_eight_roles_have_contracts() -> None:
    assert set(ROLE_CONTRACTS) == set(RoleId)
    assert len(ROLE_CONTRACTS) == 8


def test_contracts_reference_only_known_fields() -> None:
    for role_id, contract in ROLE_CONTRACTS.items():
        assert contract.role_id == role_id.value
        assert contract.access.reads <= BLACKBOARD_FIELDS
        assert contract.access.writes <= BLACKBOARD_FIELDS


def test_researcher_reads_asof_and_writes_evidence() -> None:
    researcher = ROLE_CONTRACTS[RoleId.RESEARCHER]
    assert researcher.access.may_read("as_of")
    assert researcher.access.may_write("evidence")
    assert not researcher.access.may_write("calibrated")


def test_each_role_writes_at_least_one_field() -> None:
    for contract in ROLE_CONTRACTS.values():
        assert contract.access.writes
