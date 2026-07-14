"""Unit tests for the composition root (DI). Hermetic; no network, no DB."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from common.composition import (
    build_composition,
    build_test_composition,
)
from common.llm.anthropic_api import AnthropicStructuredClient
from common.settings import Settings
from core.pit.store import InMemoryPitStore, PitStore
from core.registry.store import InMemoryRegistryStore, RegistryStore


class _StubMessages:
    def create(self, **_kwargs: Any) -> Mapping[str, Any]:  # pragma: no cover - not invoked
        return {}


class TestBuildTestComposition:
    def test_defaults_are_in_memory(self) -> None:
        comp = build_test_composition()
        assert isinstance(comp.registry_store, InMemoryRegistryStore)
        assert isinstance(comp.pit_store, InMemoryPitStore)
        # The evidence view reads from the same PIT store it was wired with.
        assert comp.evidence_view._store is comp.pit_store  # noqa: SLF001

    def test_overrides_are_respected(self) -> None:
        registry: RegistryStore = InMemoryRegistryStore()
        pit: PitStore = InMemoryPitStore()
        settings = Settings(aws_region="us-west-2")
        comp = build_test_composition(settings=settings, registry_store=registry, pit_store=pit)
        assert comp.registry_store is registry
        assert comp.pit_store is pit
        assert comp.settings.aws_region == "us-west-2"

    def test_structured_client_builds_for_tier_without_network(self) -> None:
        comp = build_test_composition()
        client = comp.structured_client("opus", client=_StubMessages())
        assert isinstance(client, AnthropicStructuredClient)
        assert client.model_id == comp.settings.model_for_tier("opus")


class TestBuildComposition:
    def test_test_profile(self) -> None:
        assert isinstance(build_composition("test").registry_store, InMemoryRegistryStore)

    def test_unknown_profile_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown composition profile"):
            build_composition("nope")  # type: ignore[arg-type]
