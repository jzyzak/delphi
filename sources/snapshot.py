"""Evidence snapshot store (C3.3) — reproducible, leakage-auditable retrieval.

A snapshot is keyed by ``hash(provider, version, query, as_of)`` and holds both
the raw provider response and the normalized as-of :class:`Evidence`. Reading a
snapshot back reproduces the exact evidence set offline (no network), which is
what makes retrospective scoring reproducible and leakage-auditable (CLAUDE.md
§7). ``as_of`` is part of the key, so the same query at two ceilings never
collides.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator

from core.forecast.search import Evidence
from core.pit.models import ensure_utc

__all__ = [
    "FileSnapshotStore",
    "InMemorySnapshotStore",
    "Snapshot",
    "SnapshotStore",
    "snapshot_key",
]

_NULL = "\x00"


def snapshot_key(*, query: str, as_of: datetime, provider: str, version: str) -> str:
    """Deterministic content-addressed key for a retrieval snapshot."""
    payload = _NULL.join([provider, version, query, ensure_utc(as_of).isoformat()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Snapshot(BaseModel):
    """A persisted retrieval: raw provider output + normalized as-of evidence."""

    model_config = ConfigDict(frozen=True)

    key: str
    query: str
    as_of: datetime
    provider: str
    version: str
    raw: dict[str, Any]
    evidence: tuple[Evidence, ...]

    @field_validator("as_of")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


@runtime_checkable
class SnapshotStore(Protocol):
    """Read/write seam for retrieval snapshots."""

    def read(self, key: str) -> Snapshot | None:
        """Return the snapshot for ``key`` or ``None`` if absent."""
        ...

    def write(self, snapshot: Snapshot) -> None:
        """Persist ``snapshot`` (idempotent overwrite by key)."""
        ...


class InMemorySnapshotStore:
    """In-memory snapshot store for tests and single-process runs."""

    def __init__(self) -> None:
        self._by_key: dict[str, Snapshot] = {}

    def read(self, key: str) -> Snapshot | None:
        return self._by_key.get(key)

    def write(self, snapshot: Snapshot) -> None:
        self._by_key[snapshot.key] = snapshot

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self) -> Iterator[Snapshot]:
        return iter(self._by_key.values())


class FileSnapshotStore:
    """Filesystem snapshot store: one JSON file per key under ``root``.

    Durable and offline-reproducible; a natural stand-in for the S3/Parquet lake
    in local runs and tests.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._root / f"{key}.json"

    def read(self, key: str) -> Snapshot | None:
        path = self._path(key)
        if not path.exists():
            return None
        return Snapshot.model_validate_json(path.read_text(encoding="utf-8"))

    def write(self, snapshot: Snapshot) -> None:
        self._path(snapshot.key).write_text(snapshot.model_dump_json(), encoding="utf-8")
