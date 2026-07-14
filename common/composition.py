"""Composition root (dependency injection).

The single place that wires concrete implementations of the shared infra
(registry store, PIT store, as-of evidence view, tiered LLM transport) into a
``Composition`` bundle. Two profiles:

* ``test`` — in-memory stores; hermetic and deterministic (CLAUDE.md §2.8).
* ``postgres`` — the Postgres-backed spine for real runs.

Higher layers (intake, sources, forecaster, ...) accept the specific deps they
need; this module is the only sanctioned place that decides which concrete
implementation to hand them. Construction never touches the network (LLM clients
build their provider client + resolve the API key lazily on first call).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from common.llm.config import LLMConfig
from common.llm.structured import StructuredLLMClient
from common.llm.tiering import structured_client_for_tier
from common.settings import Settings, load_settings
from core.pit.store import InMemoryPitStore, PitStore, PostgresPitStore
from core.pit.view import PitEvidenceView
from core.registry.store import (
    InMemoryRegistryStore,
    PostgresRegistryStore,
    RegistryStore,
)

__all__ = [
    "Composition",
    "Profile",
    "build_composition",
    "build_postgres_composition",
    "build_test_composition",
]

Profile = Literal["test", "postgres"]


@dataclass(frozen=True)
class Composition:
    """The wired shared dependencies for one process/profile."""

    settings: Settings
    registry_store: RegistryStore
    pit_store: PitStore
    evidence_view: PitEvidenceView

    def structured_client(
        self,
        tier: str,
        *,
        config: LLMConfig | None = None,
        client: Any | None = None,
    ) -> StructuredLLMClient:
        """Build a structured LLM transport for ``tier`` from these settings.

        Provider is chosen by ``settings.llm_provider`` (default: the direct
        Anthropic API). ``client`` may be injected (tests / no network);
        otherwise the transport lazily builds a real provider client + resolves
        the API key on first call.
        """
        return structured_client_for_tier(self.settings, tier, config=config, client=client)

    def hosted_searcher(self, *, http_client: Any, **kwargs: Any) -> Any:
        """Build the hosted ``AsOfSearcher`` (sources layer, lazily imported).

        Kept lazy so ``common`` carries no static dependency on the app-layer
        ``sources`` package. The ``test`` profile uses ``FixtureAsOfSearch``
        directly instead of calling this.
        """
        from sources.searcher import build_as_of_searcher

        return build_as_of_searcher(http_client=http_client, **kwargs)


def build_test_composition(
    *,
    settings: Settings | None = None,
    registry_store: RegistryStore | None = None,
    pit_store: PitStore | None = None,
) -> Composition:
    """Wire an all-in-memory composition for tests and local development.

    Any dependency may be overridden; unset ones default to in-memory backends.
    """
    resolved_settings = settings or Settings()
    resolved_pit = pit_store or InMemoryPitStore()
    return Composition(
        settings=resolved_settings,
        registry_store=registry_store or InMemoryRegistryStore(),
        pit_store=resolved_pit,
        evidence_view=PitEvidenceView(resolved_pit),
    )


def build_postgres_composition(
    settings: Settings | None = None,
) -> Composition:  # pragma: no cover - requires a reachable PostgreSQL instance
    """Wire the Postgres-backed spine. Requires ``pg_dsn`` in settings."""
    resolved_settings = settings or load_settings()
    dsn = resolved_settings.require_pg_dsn()
    pit = PostgresPitStore.connect(dsn)
    return Composition(
        settings=resolved_settings,
        registry_store=PostgresRegistryStore.connect(dsn),
        pit_store=pit,
        evidence_view=PitEvidenceView(pit),
    )


def build_composition(
    profile: Profile = "test", *, settings: Settings | None = None
) -> Composition:
    """Build a composition for the given ``profile``."""
    if profile == "test":
        return build_test_composition(settings=settings)
    if profile == "postgres":
        return build_postgres_composition(settings)  # pragma: no cover - requires PG
    msg = f"unknown composition profile: {profile!r}"
    raise ValueError(msg)
