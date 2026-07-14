"""Unit tests for resolution source adapters (C5.1)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from core.registry.models import QuestionInput
from core.registry.store import InMemoryRegistryStore
from resolution.sources import (
    MappingResolutionSource,
    ResolvedOutcome,
    load_mapping_source,
    provenance_source,
)

RESOLVED_AT = datetime(2025, 1, 2, tzinfo=UTC)


def _question(store: InMemoryRegistryStore, *, sources: list[str] | None = None):
    qid = store.record_question(
        QuestionInput(
            text="Will X win?",
            question_type="binary",
            domain="politics",
            resolution_criteria="Official result.",
            metadata={"resolution_sources": sources or []},
        )
    )
    return store.get_question(qid)


class TestProvenanceSource:
    def test_explicit_fallback_wins(self) -> None:
        store = InMemoryRegistryStore()
        q = _question(store, sources=["ignored"])
        assert provenance_source(q, "official.gov") == "official.gov"

    def test_uses_metadata_sources(self) -> None:
        store = InMemoryRegistryStore()
        q = _question(store, sources=["a.example", "b.example"])
        assert provenance_source(q, "") == "a.example; b.example"

    def test_unspecified_when_none(self) -> None:
        store = InMemoryRegistryStore()
        q = _question(store, sources=[])
        assert provenance_source(q, "") == "unspecified"

    def test_unspecified_when_sources_blank(self) -> None:
        store = InMemoryRegistryStore()
        q = _question(store, sources=["   "])
        assert provenance_source(q, "") == "unspecified"


class TestMappingResolutionSource:
    def test_resolves_known_question_with_derived_provenance(self) -> None:
        store = InMemoryRegistryStore()
        q = _question(store, sources=["league.example"])
        src = MappingResolutionSource(
            {q.question_id: ResolvedOutcome(resolved_value=1.0, resolved_at=RESOLVED_AT)}
        )
        outcome = src.resolve(q)
        assert outcome is not None
        assert outcome.resolved_value == 1.0
        assert outcome.source == "league.example"

    def test_unknown_question_returns_none(self) -> None:
        store = InMemoryRegistryStore()
        q = _question(store)
        assert MappingResolutionSource({}).resolve(q) is None


class TestLoadMappingSource:
    def _write(self, tmp_path: Path, payload: object) -> Path:
        path = tmp_path / "answers.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_full_record(self, tmp_path: Path) -> None:
        store = InMemoryRegistryStore()
        q = _question(store, sources=["gov.example"])
        path = self._write(
            tmp_path,
            {
                q.question_id: {
                    "value": 1.0,
                    "resolved_at": "2025-01-02T00:00:00Z",
                    "source": "official result",
                    "label": "YES",
                    "notes": "certified",
                }
            },
        )
        outcome = load_mapping_source(path).resolve(q)
        assert outcome is not None
        assert outcome.resolved_value == 1.0
        assert outcome.resolved_at == RESOLVED_AT
        assert outcome.source == "official result"
        assert outcome.resolved_label == "YES"
        assert outcome.notes == "certified"

    def test_defaults_provenance_from_metadata_when_source_blank(self, tmp_path: Path) -> None:
        store = InMemoryRegistryStore()
        q = _question(store, sources=["league.example"])
        path = self._write(tmp_path, {q.question_id: {"value": 0.0, "resolved_at": "2025-01-02"}})
        outcome = load_mapping_source(path).resolve(q)
        assert outcome is not None
        assert outcome.source == "league.example"

    def test_rejects_non_object_root(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [1, 2, 3])
        with pytest.raises(ValueError, match="JSON object"):
            load_mapping_source(path)

    def test_rejects_record_without_value(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"q-1": {"resolved_at": "2025-01-02"}})
        with pytest.raises(ValueError, match="must be an object with a 'value'"):
            load_mapping_source(path)

    def test_rejects_missing_resolved_at(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"q-1": {"value": 1.0}})
        with pytest.raises(ValueError, match="resolved_at"):
            load_mapping_source(path)
