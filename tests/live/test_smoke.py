"""Live smoke suite: the operator's end-to-end proof against real infrastructure.

Ordered by dependency: LLM transport (the direct Anthropic API) -> Tavily
retrieval -> full forecast against real Postgres -> conductor -> published API.
Every test asserts the CLAUDE.md invariants that matter live: no look-ahead
(evidence knowledge-times <= as-of), and a complete registry record (provenance +
reproducibility handle). Skipped entirely unless ``DELPHI_LIVE_SMOKE=1`` (see
conftest).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from common.settings import Settings

pytestmark = pytest.mark.live


def _now() -> datetime:
    # As-of is supplied by the caller here (a test), never inside forecast code.
    return datetime.now(UTC)


class TestLLM:
    def test_structured_roundtrip_per_tier(self, require_llm: Settings) -> None:
        from common.llm.tiering import structured_client_for_tier

        for tier in ("opus", "fable"):
            client = structured_client_for_tier(require_llm, tier)
            result = client.invoke_structured(
                system="You reply only with compact JSON.",
                user='Return the JSON object {"ok": true}.',
            )
            assert isinstance(result, dict)


class TestTavily:
    def test_search_filters_and_snapshots(self, require_tavily: None, tmp_path: Path) -> None:
        from common.http.client import HttpClient
        from common.secrets import EnvSecretProvider
        from sources.providers.tavily import TavilySearchClient, tavily_config
        from sources.searcher import build_as_of_searcher
        from sources.snapshot import FileSnapshotStore

        as_of = _now()
        http = HttpClient()
        client = TavilySearchClient(http=http, config=tavily_config(), secrets=EnvSecretProvider())
        store = FileSnapshotStore(tmp_path)
        searcher = build_as_of_searcher(http_client=http, client=client, snapshot_store=store)

        evidence = searcher.as_of_search("major world events this week", as_of=as_of)
        # Prime Directive §2.1: nothing dated after the as-of ceiling survives.
        assert all(e.knowledge_time <= as_of for e in evidence)
        # Snapshot-first replay returns the identical evidence set offline.
        assert searcher.as_of_search("major world events this week", as_of=as_of) == evidence
        assert list(tmp_path.glob("*.json")), "a durable snapshot file must be written"


class TestForecastEndToEnd:
    def test_forecast_writes_complete_record(
        self, require_llm: Settings, require_postgres: str, require_tavily: None
    ) -> None:
        from common.cli import _default_forecaster
        from core.registry.store import PostgresRegistryStore

        as_of = _now()
        forecaster = _default_forecaster()
        result = forecaster.forecast(
            "Will a magnitude 6+ earthquake be reported anywhere in the next 30 days?",
            as_of=as_of,
        )
        assert result.accepted
        assert result.probability is not None
        assert result.question_id is not None
        assert 0.0 <= result.probability <= 1.0
        assert all(e.knowledge_time <= as_of for e in result.evidence)

        # Complete, immutable registry record (CLAUDE.md §3): reload and verify.
        store = PostgresRegistryStore.connect(require_postgres)
        forecasts = store.forecasts_for(result.question_id)
        assert forecasts, "forecast must be persisted to the registry"
        latest = forecasts[-1]
        assert latest.model_provenance, "provenance is required (no anonymous forecasts)"
        assert latest.repro_handle.get("as_of"), "reproducibility handle must pin the as-of"


class TestConductorEndToEnd:
    def test_conductor_produces_trace(
        self, require_llm: Settings, require_postgres: str, require_tavily: None
    ) -> None:
        from common.cli import _default_conductor

        result = _default_conductor().conduct(
            "Will global average temperature set a record this year?", as_of=_now()
        )
        assert result.forecast.accepted
        assert result.workflow.route  # a non-empty routed workflow was recorded


class TestPublishedApi:
    def test_forecast_route_returns_full_envelope(
        self, require_llm: Settings, require_postgres: str, require_tavily: None
    ) -> None:
        from common.cli import _default_api_app

        app = _default_api_app()
        status, health = app.handle("GET", "/healthz")
        assert status == 200 and health == {"status": "ok"}

        status, payload = app.handle(
            "POST",
            "/v1/forecast",
            {"question": "Will it rain somewhere on Earth tomorrow?", "as_of": _now().isoformat()},
        )
        assert status == 200
        envelope = payload["delphi"]
        # Section 9 surface: probability + provenance + reproducibility handle.
        assert envelope["refused"] is False
        assert envelope["probability"] is not None
        assert envelope["evidence"] is not None
        assert envelope["reproducibility_handle"].get("as_of")
