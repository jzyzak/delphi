"""Unit tests for the resolution writer service (C5.2)."""

from __future__ import annotations

from datetime import UTC, datetime

from core.registry.models import ForecastInput, QuestionInput
from core.registry.store import InMemoryRegistryStore
from resolution.service import ResolutionService
from resolution.sources import MappingResolutionSource, ResolvedOutcome

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)
RESOLVED_AT = datetime(2025, 1, 2, tzinfo=UTC)


def _record_question(store: InMemoryRegistryStore, *, kt_domain: str = "politics") -> str:
    return store.record_question(
        QuestionInput(
            text="Will X win?",
            question_type="binary",
            domain=kt_domain,
            resolution_criteria="Official result.",
            metadata={"resolution_sources": ["league.example"]},
        )
    )


def _record_forecast(store: InMemoryRegistryStore, question_id: str) -> str:
    return store.record_forecast(
        ForecastInput(
            question_id=question_id,
            as_of=AS_OF,
            probability=0.6,
            rationale="because",
            model_provenance={"forecast_llm": "fixture"},
            repro_handle={"as_of": AS_OF.isoformat()},
        )
    )


def _outcome() -> ResolvedOutcome:
    return ResolvedOutcome(resolved_value=1.0, resolved_at=RESOLVED_AT, resolved_label="YES")


def test_resolves_and_links_to_latest_forecast() -> None:
    store = InMemoryRegistryStore()
    qid = _record_question(store)
    _record_forecast(store, qid)  # earlier forecast
    fid = _record_forecast(store, qid)  # latest forecast -> resolution links here
    service = ResolutionService(store=store, source=MappingResolutionSource({qid: _outcome()}))

    run = service.resolve_open()
    assert len(run.resolved) == 1
    resolution = store.resolutions_for(qid)[0]
    assert resolution.resolved_value == 1.0
    assert resolution.forecast_id == fid
    assert resolution.source == "league.example"


def test_idempotent_rerun_skips_resolved() -> None:
    store = InMemoryRegistryStore()
    qid = _record_question(store)
    service = ResolutionService(store=store, source=MappingResolutionSource({qid: _outcome()}))
    first = service.resolve_open()
    second = service.resolve_open()
    assert len(first.resolved) == 1
    assert second.resolved == ()
    assert qid in second.skipped


def test_since_filter_skips_older_questions() -> None:
    store = InMemoryRegistryStore()
    qid = _record_question(store)
    service = ResolutionService(store=store, source=MappingResolutionSource({qid: _outcome()}))
    run = service.resolve_open(since=datetime(2999, 1, 1, tzinfo=UTC))
    assert run.resolved == ()
    assert qid in run.skipped


def test_unresolvable_question_skipped() -> None:
    store = InMemoryRegistryStore()
    qid = _record_question(store)
    service = ResolutionService(store=store, source=MappingResolutionSource({}))
    run = service.resolve_open()
    assert run.resolved == ()
    assert qid in run.skipped


def test_resolution_without_forecast_has_no_forecast_link() -> None:
    store = InMemoryRegistryStore()
    qid = _record_question(store)
    service = ResolutionService(store=store, source=MappingResolutionSource({qid: _outcome()}))
    service.resolve_open()
    assert store.resolutions_for(qid)[0].forecast_id is None
