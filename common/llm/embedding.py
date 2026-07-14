"""Generic Amazon Bedrock text-embedding transport.

Domain-agnostic (CLAUDE.md section 11): this module knows nothing about agent
memory or any forecast domain. It turns text into fixed-dimension unit vectors
via the Bedrock Runtime ``invoke_model`` API (Amazon Titan Text Embeddings V2),
with retries and bounded concurrency. The agent-memory layer builds an
``Embedder`` around it (``core/memory/embedder.py``).

``boto3`` is imported lazily so importing this module never requires it, and an
explicit ``invoke_model``-compatible client may be injected for tests (no
network). Model id and supported dimensions are pinned from the Bedrock docs
(CLAUDE.md section 4), not from memory.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol, runtime_checkable

import structlog
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.llm.config import LLMConfig
from common.llm.errors import LLMError, LLMThrottledError, MalformedLLMOutput

__all__ = [
    "AMAZON_TITAN_EMBED_V2_ID",
    "TITAN_V2_SUPPORTED_DIMENSIONS",
    "BedrockEmbeddingClient",
    "InvokeModelClient",
]

_LOG = structlog.get_logger(__name__)

# Pinned from the AWS Bedrock model card for Amazon Titan Text Embeddings V2
# (CLAUDE.md section 4 — confirmed from docs, not hardcoded from memory).
AMAZON_TITAN_EMBED_V2_ID = "amazon.titan-embed-text-v2:0"
TITAN_V2_SUPPORTED_DIMENSIONS = frozenset({256, 512, 1024})

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
class InvokeModelClient(Protocol):
    """Minimal seam matching the boto3 ``bedrock-runtime`` ``invoke_model``."""

    def invoke_model(self, **kwargs: Any) -> Mapping[str, Any]:
        """Invoke the model and return the raw ``invoke_model`` response mapping."""
        ...


def _default_bedrock_client(region_name: str | None) -> InvokeModelClient:
    """Construct a real boto3 bedrock-runtime client (lazy boto3 import)."""
    try:
        boto3 = importlib.import_module("boto3")
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        msg = (
            "boto3 is required for BedrockEmbeddingClient; install it or inject "
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


def _read_body(response: Mapping[str, Any]) -> dict[str, Any]:
    """Decode an ``invoke_model`` response body into a JSON object.

    Handles the real boto3 shape (``response['body']`` is a streaming object
    with ``.read()`` returning bytes/str) and injected test doubles that hand
    back a str, bytes, or mapping directly.
    """
    try:
        body = response["body"]
    except (KeyError, TypeError) as exc:
        msg = f"invoke_model response missing 'body': {response!r}"
        raise MalformedLLMOutput(msg) from exc
    raw: Any = body.read() if hasattr(body, "read") else body
    if isinstance(raw, bytes | bytearray):
        raw = bytes(raw).decode("utf-8")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"invoke_model body is not valid JSON: {raw[:200]!r}"
            raise MalformedLLMOutput(msg) from exc
    elif isinstance(raw, Mapping):
        parsed = dict(raw)
    else:
        msg = f"invoke_model body has unexpected type: {type(raw)!r}"
        raise MalformedLLMOutput(msg)
    if not isinstance(parsed, dict):
        msg = f"invoke_model body JSON is not an object: {parsed!r}"
        raise MalformedLLMOutput(msg)
    return parsed


def _extract_embedding(payload: Mapping[str, Any], *, dimensions: int) -> list[float]:
    """Pull the ``embedding`` vector out of a Titan response payload."""
    embedding = payload.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        msg = f"invoke_model response has no 'embedding' array: {payload!r}"
        raise MalformedLLMOutput(msg)
    try:
        vector = [float(x) for x in embedding]
    except (TypeError, ValueError) as exc:
        msg = f"embedding contains non-numeric values: {embedding[:8]!r}"
        raise MalformedLLMOutput(msg) from exc
    if len(vector) != dimensions:
        msg = f"embedding dimension {len(vector)} != requested {dimensions}"
        raise MalformedLLMOutput(msg)
    return vector


class BedrockEmbeddingClient:
    """Bedrock ``invoke_model`` embedding transport bound to one model + dim.

    Contract: ``embed`` returns one vector per input text, in input order, each
    of length ``dimensions``. Construction never touches the network or boto3;
    the real client is created lazily on first call. Titan V2 returns unit
    vectors when ``normalize=True`` (the default and the RAG-optimal setting).
    """

    def __init__(
        self,
        *,
        model_id: str = AMAZON_TITAN_EMBED_V2_ID,
        dimensions: int = 1024,
        normalize: bool = True,
        region_name: str | None = None,
        config: LLMConfig | None = None,
        client: InvokeModelClient | None = None,
    ) -> None:
        if dimensions not in TITAN_V2_SUPPORTED_DIMENSIONS:
            supported = sorted(TITAN_V2_SUPPORTED_DIMENSIONS)
            msg = f"dimensions must be one of {supported}, got {dimensions!r}"
            raise ValueError(msg)
        self._model_id = model_id
        self._dimensions = dimensions
        self._normalize = normalize
        self._region = region_name
        self._config = config or LLMConfig()
        self._client = client

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _ensure_client(self) -> InvokeModelClient:
        if self._client is None:
            self._client = _default_bedrock_client(self._region)
        return self._client

    def _invoke_once(self, text: str) -> list[float]:
        """Single invoke_model call; normalizes provider errors to typed errors."""
        client = self._ensure_client()
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": self._dimensions,
                "normalize": self._normalize,
            }
        )
        try:
            response = client.invoke_model(
                modelId=self._model_id,
                accept="application/json",
                contentType="application/json",
                body=body,
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            if _is_throttling(exc):
                raise LLMThrottledError(str(exc)) from exc
            raise LLMError(str(exc)) from exc
        payload = _read_body(response)
        return _extract_embedding(payload, dimensions=self._dimensions)

    def _embed_one(self, text: str) -> list[float]:
        """Embed one text with retry on throttling/malformed output."""
        cfg = self._config
        for attempt in Retrying(
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential(multiplier=cfg.retry_backoff_base, max=cfg.retry_backoff_max),
            retry=retry_if_exception_type((LLMThrottledError, MalformedLLMOutput)),
            reraise=True,
        ):
            with attempt:
                try:
                    return self._invoke_once(text)
                except (LLMThrottledError, MalformedLLMOutput) as exc:
                    _LOG.info(
                        "llm.bedrock.embedding.retryable_error",
                        model_id=self._model_id,
                        error_type=type(exc).__name__,
                    )
                    raise
        raise LLMError("unreachable")  # pragma: no cover - reraise=True propagates

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` concurrently; results preserve input order."""
        if not texts:
            return []
        max_workers = min(self._config.max_concurrency, len(texts))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(self._embed_one, text) for text in texts]
            return [future.result() for future in futures]
