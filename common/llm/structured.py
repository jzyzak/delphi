"""Transport-neutral structured-output layer (domain-agnostic, CLAUDE.md §11).

This module owns everything about turning a ``(system, user)`` prompt pair into a
parsed JSON object that does **not** depend on a specific provider: the batch
request type, tolerant JSON extraction, and the retry + bounded-concurrency
engine. Concrete transports (:mod:`common.llm.anthropic_api` for the direct
Claude API, :mod:`common.llm.bedrock` for AWS Bedrock) subclass
:class:`StructuredClientBase` and implement a single ``_generate_text`` method.

Keeping this seam provider-neutral is what lets the rest of the system (intake,
the forecast ensemble, research agents) stay ignorant of *which* Claude endpoint
answered — the value is the pipeline, not the transport (CLAUDE.md §1).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

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
    "StructuredClientBase",
    "StructuredLLMClient",
    "StructuredPrompt",
    "parse_json_object",
]

_LOG = structlog.get_logger(__name__)

# First "{" through last "}" — captures a single (possibly nested) JSON object
# even when the model wraps it in prose or a code fence.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class StructuredPrompt:
    """One structured-output request within a batch.

    ``run_index`` identifies the draw for provenance and order preservation.
    """

    system: str
    user: str
    run_index: int


@runtime_checkable
class StructuredLLMClient(Protocol):
    """The structured-output transport contract shared by all providers.

    Satisfied by both :class:`common.llm.anthropic_api.AnthropicStructuredClient`
    and :class:`common.llm.bedrock.BedrockStructuredClient`. Higher layers depend
    on this Protocol, never on a concrete transport, so the provider can change
    without touching the forecast pipeline.
    """

    provider: ClassVar[str]

    @property
    def model_id(self) -> str: ...

    @property
    def config(self) -> LLMConfig: ...

    def invoke_structured(self, *, system: str, user: str) -> dict[str, Any]: ...

    def invoke_structured_batch(
        self, prompts: Sequence[StructuredPrompt]
    ) -> list[dict[str, Any]]: ...


def parse_json_object(text: str) -> dict[str, Any]:
    """Tolerantly extract and parse the first JSON object in ``text``."""
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        msg = f"no JSON object found in model output: {text[:200]!r}"
        raise MalformedLLMOutput(msg)
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        msg = f"model output is not valid JSON: {text[:200]!r}"
        raise MalformedLLMOutput(msg) from exc
    if not isinstance(parsed, dict):
        msg = f"model output JSON is not an object: {parsed!r}"
        raise MalformedLLMOutput(msg)
    return parsed


class StructuredClientBase:
    """Provider-agnostic structured client: retries + bounded-concurrency batch.

    One client serves one capability tier (one ``model_id``). Construction never
    touches the network; the real provider client is built lazily on first call
    by the subclass. Subclasses set the ``provider`` class attribute and implement
    :meth:`_generate_text`.
    """

    provider: ClassVar[str] = "structured"

    def __init__(self, *, model_id: str, config: LLMConfig | None = None) -> None:
        self._model_id = model_id
        self._config = config or LLMConfig()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def config(self) -> LLMConfig:
        return self._config

    def _generate_text(self, *, system: str, user: str) -> str:
        """Call the provider once and return the assistant's text.

        Implementations must normalize provider errors to :class:`LLMThrottled
        Error` (retryable) / :class:`LLMError`, and raise :class:`MalformedLLM
        Output` when no text can be extracted.
        """
        raise NotImplementedError  # pragma: no cover - abstract hook

    def invoke_structured(self, *, system: str, user: str) -> dict[str, Any]:
        """Invoke the model once and return the parsed JSON object.

        Retries throttling and malformed output up to ``config.max_retries``
        with exponential backoff; on exhaustion the last error propagates.
        """
        cfg = self._config
        last_exc: Exception | None = None
        for attempt in Retrying(
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential(multiplier=cfg.retry_backoff_base, max=cfg.retry_backoff_max),
            retry=retry_if_exception_type((LLMThrottledError, MalformedLLMOutput)),
            reraise=True,
        ):
            with attempt:
                try:
                    text = self._generate_text(system=system, user=user)
                    return parse_json_object(text)
                except (LLMThrottledError, MalformedLLMOutput) as exc:
                    last_exc = exc
                    _LOG.info(
                        "llm.retryable_error",
                        provider=self.provider,
                        model_id=self._model_id,
                        error_type=type(exc).__name__,
                    )
                    raise
        # Unreachable: reraise=True propagates the final attempt's exception.
        raise last_exc if last_exc is not None else LLMError("unreachable")  # pragma: no cover

    def invoke_structured_batch(self, prompts: Sequence[StructuredPrompt]) -> list[dict[str, Any]]:
        """Invoke ``prompts`` concurrently; results preserve input order.

        Concurrency is bounded by ``config.max_concurrency``. This is the batch
        seam for the ~10-run ensemble; provider-native batch inference is
        reserved for bulk workloads.
        """
        if not prompts:
            return []
        max_workers = min(self._config.max_concurrency, len(prompts))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(self.invoke_structured, system=p.system, user=p.user) for p in prompts
            ]
            # ``.result()`` in submission order preserves run order and
            # re-raises the first failure.
            return [future.result() for future in futures]
