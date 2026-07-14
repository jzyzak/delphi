"""Unit tests for tier-to-transport resolution. Transport-mocked, no network."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from common.llm.anthropic_api import AnthropicStructuredClient
from common.llm.bedrock import BedrockStructuredClient
from common.llm.tiering import structured_client_for_tier
from common.settings import Settings


class _StubMessages:
    def create(self, **_kwargs: Any) -> Any:  # pragma: no cover - not invoked
        return {}


class _StubConverse:
    def converse(self, **_kwargs: Any) -> Mapping[str, Any]:  # pragma: no cover - not invoked
        return {}


@pytest.mark.parametrize("tier", ["opus", "fable"])
def test_default_provider_resolves_to_anthropic(tier: str) -> None:
    settings = Settings()  # default llm_provider == "anthropic"
    client = structured_client_for_tier(settings, tier, client=_StubMessages())
    assert isinstance(client, AnthropicStructuredClient)
    assert client.model_id == settings.model_for_tier(tier)
    assert client.provider == "anthropic"


@pytest.mark.parametrize("tier", ["opus", "fable"])
def test_bedrock_provider_is_opt_in(tier: str) -> None:
    settings = Settings(llm_provider="bedrock")
    client = structured_client_for_tier(settings, tier, client=_StubConverse())
    assert isinstance(client, BedrockStructuredClient)
    assert client.model_id == settings.model_for_tier(tier)
    assert client.provider == "bedrock"


def test_unknown_tier_raises_key_error() -> None:
    with pytest.raises(KeyError):
        structured_client_for_tier(Settings(), "titan", client=_StubMessages())
