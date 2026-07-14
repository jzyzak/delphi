"""Amazon Bedrock structured-output transport (opt-in; see CLAUDE.md §7).

Domain-agnostic (CLAUDE.md §11): converts a ``(system, user)`` prompt pair into
a parsed JSON object via the Bedrock Runtime ``converse`` API. The shared retry +
bounded-concurrency engine lives in :mod:`common.llm.structured`; this module
only implements the Bedrock-specific call.

NOTE (transport selection): DELPHI defaults to the **direct Anthropic API**
transport (:mod:`common.llm.anthropic_api`). This Bedrock transport is retained
for AWS-native deployment and is selected only when ``DELPHI_LLM_PROVIDER=bedrock``
(see :func:`common.llm.tiering.structured_client_for_tier`).

``boto3`` is imported lazily so importing this module never requires it, and an
explicit ``converse``-compatible client may be injected for tests (no network).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, runtime_checkable

from common.llm.config import LLMConfig
from common.llm.errors import LLMError, LLMThrottledError, MalformedLLMOutput
from common.llm.structured import StructuredClientBase, StructuredPrompt

__all__ = [
    "BedrockStructuredClient",
    "ConverseClient",
    "StructuredPrompt",
]

_THROTTLING_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
        "InternalServerException",
    }
)


@runtime_checkable
class ConverseClient(Protocol):
    """Minimal seam matching the boto3 ``bedrock-runtime`` ``converse`` method."""

    def converse(self, **kwargs: Any) -> Mapping[str, Any]:
        """Invoke the model and return the raw converse response mapping."""
        ...


def _default_bedrock_client(region_name: str | None) -> ConverseClient:
    """Construct a real boto3 bedrock-runtime client (lazy boto3 import)."""
    try:
        boto3 = importlib.import_module("boto3")
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        msg = (
            "boto3 is required for BedrockStructuredClient; install it or inject "
            "a client. Hint: `uv add boto3`."
        )
        raise RuntimeError(msg) from exc
    return boto3.client("bedrock-runtime", region_name=region_name)


def _is_throttling(exc: Exception) -> bool:
    """Heuristically classify a provider exception as retryable throttling."""
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        code = response.get("Error", {}).get("Code")
        if code in _THROTTLING_CODES:
            return True
    return any(token in type(exc).__name__ for token in ("Throttling", "TooManyRequests"))


def _extract_text(response: Mapping[str, Any]) -> str:
    """Pull the first assistant text block out of a converse response."""
    try:
        content = response["output"]["message"]["content"]
    except (KeyError, TypeError) as exc:
        msg = f"converse response missing output.message.content: {response!r}"
        raise MalformedLLMOutput(msg) from exc
    for block in content:
        if isinstance(block, Mapping) and "text" in block:
            return str(block["text"])
    msg = f"converse response has no text content block: {content!r}"
    raise MalformedLLMOutput(msg)


class BedrockStructuredClient(StructuredClientBase):
    """Bedrock ``converse`` transport bound to a single model id.

    One client serves one capability tier (one ``model_id``). Construction never
    touches the network or boto3; the real client is created lazily on first call.
    """

    provider: ClassVar[str] = "bedrock"

    def __init__(
        self,
        *,
        model_id: str,
        region_name: str | None = None,
        config: LLMConfig | None = None,
        client: ConverseClient | None = None,
    ) -> None:
        super().__init__(model_id=model_id, config=config)
        self._region = region_name
        self._client = client

    def _ensure_client(self) -> ConverseClient:
        if self._client is None:
            self._client = _default_bedrock_client(self._region)
        return self._client

    def _generate_text(self, *, system: str, user: str) -> str:
        """Single converse call; normalizes provider errors to typed errors."""
        client = self._ensure_client()
        system_blocks = [{"text": system}] if system else []
        try:
            response = client.converse(
                modelId=self._model_id,
                system=system_blocks,
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={
                    "maxTokens": self._config.max_tokens,
                    "temperature": self._config.temperature,
                    "topP": self._config.top_p,
                },
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            if _is_throttling(exc):
                raise LLMThrottledError(str(exc)) from exc
            raise LLMError(str(exc)) from exc
        return _extract_text(response)
