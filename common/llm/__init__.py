"""Generic LLM transport layer (domain-agnostic core, CLAUDE.md section 11).

Kept out of ``common/__init__`` so that importing ``common`` never pulls the
transport (or its lazy provider SDKs). Import directly, e.g.: ``from common.llm
import AnthropicStructuredClient``.

DELPHI's default transport is the **direct Anthropic (Claude) API**
(:class:`AnthropicStructuredClient`); AWS Bedrock (:class:`BedrockStructured
Client`) is opt-in via ``DELPHI_LLM_PROVIDER=bedrock``.
"""

from __future__ import annotations

from common.llm.anthropic_api import (
    ANTHROPIC_API_KEY_SECRET,
    AnthropicStructuredClient,
    MessagesClient,
)
from common.llm.bedrock import (
    BedrockStructuredClient,
    ConverseClient,
)
from common.llm.config import LLMConfig
from common.llm.embedding import (
    AMAZON_TITAN_EMBED_V2_ID,
    TITAN_V2_SUPPORTED_DIMENSIONS,
    BedrockEmbeddingClient,
    InvokeModelClient,
)
from common.llm.errors import LLMError, LLMThrottledError, MalformedLLMOutput
from common.llm.structured import (
    StructuredClientBase,
    StructuredLLMClient,
    StructuredPrompt,
    parse_json_object,
)
from common.llm.tiering import structured_client_for_tier

__all__ = [
    "AMAZON_TITAN_EMBED_V2_ID",
    "ANTHROPIC_API_KEY_SECRET",
    "TITAN_V2_SUPPORTED_DIMENSIONS",
    "AnthropicStructuredClient",
    "BedrockEmbeddingClient",
    "BedrockStructuredClient",
    "ConverseClient",
    "InvokeModelClient",
    "LLMConfig",
    "LLMError",
    "LLMThrottledError",
    "MalformedLLMOutput",
    "MessagesClient",
    "StructuredClientBase",
    "StructuredLLMClient",
    "StructuredPrompt",
    "parse_json_object",
    "structured_client_for_tier",
]
