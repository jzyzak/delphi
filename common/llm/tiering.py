"""Tier-to-transport helper for the LLM layer.

Resolves a capability tier ('opus'/'fable') to a
:class:`StructuredLLMClient` bound to the model id pinned in ``Settings``.
Construction never touches the network or a provider SDK (the real client is
built lazily on first call), so this is safe to call eagerly and inject in tests.

Transport selection is driven by ``settings.llm_provider``:

* ``"anthropic"`` (the **default**) — the direct Claude API (Claude Console key).
* ``"bedrock"`` — AWS Bedrock (opt-in; set ``DELPHI_LLM_PROVIDER=bedrock``).
"""

from __future__ import annotations

from typing import Any

from common.llm.anthropic_api import AnthropicStructuredClient
from common.llm.config import LLMConfig
from common.llm.structured import StructuredLLMClient
from common.settings import Settings

__all__ = ["structured_client_for_tier"]


def structured_client_for_tier(
    settings: Settings,
    tier: str,
    *,
    config: LLMConfig | None = None,
    client: Any | None = None,
) -> StructuredLLMClient:
    """Build a structured transport for ``tier`` from ``settings``.

    Args:
        settings: Source of the pinned per-tier model id and the provider.
        tier: Capability tier ('opus' | 'fable'); ``KeyError`` if
            unknown (delegated to :meth:`Settings.model_for_tier`).
        config: Optional inference/transport config; defaults to ``LLMConfig()``.
        client: Optional injected provider client (tests / no network). For the
            Anthropic transport this is a ``MessagesClient``; for Bedrock a
            ``ConverseClient``.
    """
    model_id = settings.model_for_tier(tier)

    # --- AWS Bedrock (opt-in; disabled by default) --------------------------
    # Re-enable AWS by setting DELPHI_LLM_PROVIDER=bedrock (and DELPHI_AWS_REGION
    # + Bedrock-style model ids). boto3 is imported lazily inside the client.
    if settings.llm_provider == "bedrock":
        from common.llm.bedrock import BedrockStructuredClient

        return BedrockStructuredClient(
            model_id=model_id,
            region_name=settings.aws_region,
            config=config,
            client=client,
        )

    # --- Direct Anthropic (Claude) API — the default transport --------------
    return AnthropicStructuredClient(model_id=model_id, config=config, client=client)
