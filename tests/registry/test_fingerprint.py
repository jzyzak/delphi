"""Unit tests for canonical hashing and the trial fingerprint (F5 / R4)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.registry.fingerprint import (
    canonical_json,
    compute_record_hash,
    content_hash,
    trial_fingerprint,
)
from tests.registry.conftest import AS_OF, make_repro


class TestCanonicalJson:
    def test_key_order_is_irrelevant(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_nested_key_order_is_irrelevant(self) -> None:
        left = canonical_json({"x": {"q": 1, "p": 2}})
        right = canonical_json({"x": {"p": 2, "q": 1}})
        assert left == right

    def test_list_order_is_preserved(self) -> None:
        assert canonical_json([1, 2]) != canonical_json([2, 1])

    def test_datetime_serialized_as_isoformat(self) -> None:
        dt = datetime(2024, 6, 1, tzinfo=UTC)
        assert canonical_json({"t": dt}) == '{"t":"2024-06-01T00:00:00+00:00"}'

    def test_set_is_sorted(self) -> None:
        assert canonical_json({3, 1, 2}) == "[1,2,3]"


class TestContentHash:
    def test_deterministic(self) -> None:
        obj = {"a": 1, "b": [1, 2, 3]}
        assert content_hash(obj) == content_hash(obj)

    def test_sensitive_to_change(self) -> None:
        assert content_hash({"a": 1}) != content_hash({"a": 2})

    def test_is_hex_sha256(self) -> None:
        digest = content_hash({"a": 1})
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


class TestComputeRecordHash:
    def _args(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "stream_id": "exp_1",
            "seq": 0,
            "record_kind": "experiment",
            "record_id": "exp_1",
            "payload": {"hypothesis": "h"},
            "prev_hash": None,
            "knowledge_time": datetime(2025, 1, 1, tzinfo=UTC),
        }
        base.update(overrides)
        return base

    def test_deterministic(self) -> None:
        assert compute_record_hash(**self._args()) == compute_record_hash(  # type: ignore[arg-type]
            **self._args()  # type: ignore[arg-type]
        )

    def test_prev_hash_changes_digest(self) -> None:
        a = compute_record_hash(**self._args(prev_hash=None))  # type: ignore[arg-type]
        b = compute_record_hash(**self._args(prev_hash="abc"))  # type: ignore[arg-type]
        assert a != b

    def test_payload_edit_changes_digest(self) -> None:
        a = compute_record_hash(**self._args(payload={"hypothesis": "h"}))  # type: ignore[arg-type]
        b = compute_record_hash(**self._args(payload={"hypothesis": "EDIT"}))  # type: ignore[arg-type]
        assert a != b


class TestTrialFingerprint:
    def test_identical_metadata_same_fingerprint(self) -> None:
        assert trial_fingerprint(make_repro()) == trial_fingerprint(make_repro())

    def test_param_key_order_is_irrelevant(self) -> None:
        a = trial_fingerprint(make_repro(params={"lookback": 20, "z_entry": 1.5}))
        b = trial_fingerprint(make_repro(params={"z_entry": 1.5, "lookback": 20}))
        assert a == b

    def test_changing_params_changes_fingerprint(self) -> None:
        a = trial_fingerprint(make_repro(params={"lookback": 20}))
        b = trial_fingerprint(make_repro(params={"lookback": 21}))
        assert a != b

    def test_changing_spec_hash_changes_fingerprint(self) -> None:
        a = trial_fingerprint(make_repro(spec_hash="s1"))
        b = trial_fingerprint(make_repro(spec_hash="s2"))
        assert a != b

    def test_changing_as_of_changes_fingerprint(self) -> None:
        from core.registry.models import DataSnapshot

        other = DataSnapshot(
            as_of=datetime(2024, 7, 1, tzinfo=UTC),
            universe_spec={"classification": "elections", "min_price": 1.0},
        )
        a = trial_fingerprint(make_repro())
        b = trial_fingerprint(make_repro(data_snapshot=other))
        assert a != b

    def test_changing_universe_changes_fingerprint(self) -> None:
        from core.registry.models import DataSnapshot

        other = DataSnapshot(
            as_of=AS_OF, universe_spec={"classification": "elections", "min_price": 5.0}
        )
        a = trial_fingerprint(make_repro())
        b = trial_fingerprint(make_repro(data_snapshot=other))
        assert a != b

    def test_code_sha_and_seeds_do_not_change_fingerprint(self) -> None:
        # A re-run on a different build / seed is the SAME trial (anti-cherry-pick).
        a = trial_fingerprint(make_repro(code_sha="aaaa", seeds={"numpy": 1}))
        b = trial_fingerprint(make_repro(code_sha="bbbb", seeds={"numpy": 999}))
        assert a == b

    @pytest.mark.parametrize("bad", [None, 123])
    def test_rejects_non_metadata(self, bad: object) -> None:
        with pytest.raises(AttributeError):
            trial_fingerprint(bad)  # type: ignore[arg-type]
