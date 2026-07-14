"""Round-trip + determinism tests for the evidence snapshot store (C3.3)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from core.forecast.search import Evidence
from sources.snapshot import (
    FileSnapshotStore,
    InMemorySnapshotStore,
    Snapshot,
    snapshot_key,
)

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _snapshot(key: str = "k") -> Snapshot:
    return Snapshot(
        key=key,
        query="will it rain",
        as_of=AS_OF,
        provider="hosted",
        version="v1",
        raw={"pages": [{"results": []}]},
        evidence=(
            Evidence(
                snippet="s",
                source="hosted",
                source_id="http://a",
                knowledge_time=datetime(2024, 1, 1, tzinfo=UTC),
                score=0.4,
            ),
        ),
    )


class TestSnapshotKey:
    def test_deterministic(self) -> None:
        a = snapshot_key(query="q", as_of=AS_OF, provider="hosted", version="v1")
        b = snapshot_key(query="q", as_of=AS_OF, provider="hosted", version="v1")
        assert a == b

    def test_as_of_included(self) -> None:
        a = snapshot_key(query="q", as_of=AS_OF, provider="hosted", version="v1")
        b = snapshot_key(
            query="q", as_of=datetime(2025, 1, 1, tzinfo=UTC), provider="hosted", version="v1"
        )
        assert a != b

    def test_query_included(self) -> None:
        a = snapshot_key(query="q1", as_of=AS_OF, provider="hosted", version="v1")
        b = snapshot_key(query="q2", as_of=AS_OF, provider="hosted", version="v1")
        assert a != b


class TestInMemorySnapshotStore:
    def test_write_read_round_trip(self) -> None:
        store = InMemorySnapshotStore()
        store.write(_snapshot())
        got = store.read("k")
        assert got is not None
        assert got.evidence[0].source_id == "http://a"
        assert len(store) == 1
        assert list(store)[0].key == "k"

    def test_missing_returns_none(self) -> None:
        assert InMemorySnapshotStore().read("absent") is None


class TestFileSnapshotStore:
    def test_offline_reproduce_identical_evidence(self, tmp_path: Path) -> None:
        store = FileSnapshotStore(tmp_path)
        original = _snapshot("abc")
        store.write(original)
        # A fresh store over the same dir reproduces byte-identical evidence offline.
        reread = FileSnapshotStore(tmp_path).read("abc")
        assert reread is not None
        assert reread.evidence == original.evidence
        assert reread.raw == original.raw
        assert reread.as_of == AS_OF

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        assert FileSnapshotStore(tmp_path).read("nope") is None
