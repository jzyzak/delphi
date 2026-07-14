"""Direct Anthropic (Claude) API structured-output transport — the default.

Domain-agnostic (CLAUDE.md §11): converts a ``(system, user)`` prompt pair into
a parsed JSON object via the Anthropic Messages API (``api.anthropic.com``). The
shared retry + bounded-concurrency engine lives in :mod:`common.llm.structured`;
this module only implements the Anthropic-specific call.

This is DELPHI's default transport (vs. AWS Bedrock). The API key is resolved at
call time from the secret ``anthropic-api-key`` (env var
``DELPHI_SECRET_ANTHROPIC_API_KEY``), never hardcoded (CLAUDE.md §7). The
``anthropic`` SDK is imported lazily so importing this module never requires it,
and an explicit messages client may be injected for tests (no network, no key).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, runtime_checkable

from common.llm.config import LLMConfig
from common.llm.errors import LLMError, LLMThrottledError, MalformedLLMOutput
from common.llm.structured import StructuredClientBase
from common.secrets import EnvSecretProvider, SecretProvider

__all__ = [
    "ANTHROPIC_API_KEY_SECRET",
    "AnthropicStructuredClient",
    "MessagesClient",
]

# Logical secret name (resolved via common.secrets to
# ``DELPHI_SECRET_ANTHROPIC_API_KEY`` by the env provider).
ANTHROPIC_API_KEY_SECRET = "anthropic-api-key"

# HTTP statuses the Anthropic API uses for transient/overloaded conditions
# (529 = "overloaded"). Retryable via the shared engine.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_RETRYABLE_NAME_TOKENS = (
    "RateLimit",
    "Overloaded",
    "InternalServer",
    "ServiceUnavailable",
    "APIConnection",
    "APITimeout",
    "Timeout",
)


@runtime_checkable
class MessagesClient(Protocol):
    """Minimal seam matching ``anthropic.Anthropic().messages`` (its ``create``)."""

    def create(self, **kwargs: Any) -> Any:
        """Invoke the Messages API and return the raw response object."""
        ...


def _default_messages_client(api_key: str) -> MessagesClient:
    """Construct the real Anthropic Messages client (lazy anthropic import)."""
    try:
        anthropic = importlib.import_module("anthropic")
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        msg = (
            "anthropic is required for AnthropicStructuredClient; install it or "
            "inject a client. Hint: `uv add anthropic`."
        )
        raise RuntimeError(msg) from exc
    return anthropic.Anthropic(api_key=api_key).messages


def _is_throttling(exc: Exception) -> bool:
    """Heuristically classify an Anthropic SDK exception as retryable."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS:
        return True
    name = type(exc).__name__
    return any(token in name for token in _RETRYABLE_NAME_TOKENS)


def _extract_text(response: Any) -> str:
    """Pull the first assistant text block from a Messages API response.

    Tolerant of both SDK objects (``response.content[i].text``) and plain
    mappings, and skips non-text blocks (e.g. thinking blocks emitted by
    adaptive-thinking models) to find the JSON-bearing text.
    """
    content = getattr(response, "content", None)
    if content is None and isinstance(response, Mapping):
        content = response.get("content")
    if not content:
        msg = f"anthropic response missing content: {response!r}"
        raise MalformedLLMOutput(msg)
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, Mapping):
            text = block.get("text")
        if isinstance(text, str):
            return text
    msg = f"anthropic response has no text content block: {content!r}"
    raise MalformedLLMOutput(msg)


class AnthropicStructuredClient(StructuredClientBase):
    """Anthropic Messages transport bound to a single model id.

    One client serves one capability tier (one ``model_id``). Construction never
    touches the network, the ``anthropic`` SDK, or the API key; all three are
    resolved lazily on first call.
    """

    provider: ClassVar[str] = "anthropic"

    def __init__(
        self,
        *,
        model_id: str,
        config: LLMConfig | None = None,
        client: MessagesClient | None = None,
        api_key: str | None = None,
        secrets: SecretProvider | None = None,
        secret_name: str = ANTHROPIC_API_KEY_SECRET,
    ) -> None:
        super().__init__(model_id=model_id, config=config)
        self._client = client
        self._api_key = api_key
        self._secrets = secrets
        self._secret_name = secret_name

    def _ensure_client(self) -> MessagesClient:
        if self._client is None:
            api_key = self._api_key
            if api_key is None:
                provider = self._secrets or EnvSecretProvider()
                api_key = provider.get_secret(self._secret_name)
            self._client = _default_messages_client(api_key)
        return self._client

    def _generate_text(self, *, system: str, user: str) -> str:
        """Single Messages call; normalizes provider errors to typed errors."""
        client = self._ensure_client()
        # We send ``temperature`` (drives ensemble diversity) but NOT ``top_p``:
        # Anthropic recommends tuning one or the other, and adaptive-thinking
        # models (e.g. Opus 4.8, Fable 5) reject ``top_p`` with a 400. Bedrock's
        # converse API is unaffected and keeps sending topP.
        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        try:
            response = client.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            if _is_throttling(exc):
                raise LLMThrottledError(str(exc)) from exc
            raise LLMError(str(exc)) from exc
        return _extract_text(response)
