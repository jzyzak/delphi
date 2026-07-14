"""Mockable LLM seam for independent forecast draws."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.llm import StructuredLLMClient, StructuredPrompt
from common.llm.errors import MalformedLLMOutput


@dataclass(frozen=True)
class ForecastRequest:
    """One independent forecast draw request within a batched call."""

    content: str
    content_hash: str
    run_index: int
    prompt: str


class ForecastDraw(BaseModel):
    """Structured output from a single forecast draw.

    Emits a probability plus provenance so downstream prompts (20, 24) can add
    fields without changing the ensemble aggregation contract.
    """

    model_config = ConfigDict(frozen=True)

    probability: float = Field(ge=0.0, le=1.0)
    run_index: int = Field(ge=0)
    model_version: str
    prompt_version: str
    provenance: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("provenance")
    @classmethod
    def _freeze_provenance(cls, v: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(v)


@runtime_checkable
class ForecastLLM(Protocol):
    """Batched forecast LLM seam — the only call path for N-run ensembles."""

    @property
    def model_version(self) -> str: ...

    @property
    def prompt_version(self) -> str: ...

    def forecast_batch(self, requests: Sequence[ForecastRequest]) -> Sequence[ForecastDraw]:
        """Return one structured draw per request in a single batched call."""
        ...


class FixtureForecastLLM:
    """Deterministic forecast LLM for tests (no network).

    Supports explicit per-run probability sequences keyed by ``content_hash``,
    or a seeded base+noise mode for variance-reduction fixtures.
    """

    def __init__(
        self,
        responses: Mapping[str, Sequence[float]] | None = None,
        *,
        model_version: str = "fixture-forecast-v1",
        prompt_version: str = "delphi_forecast_v1",
        default_response: Sequence[float] | float | None = None,
        base_probability: float | None = None,
        noise_std: float = 0.05,
        seed: int = 42,
    ) -> None:
        self._responses = {k: tuple(v) for k, v in (responses or {}).items()}
        if default_response is None:
            self._default: tuple[float, ...] = (0.5,)
        elif isinstance(default_response, (int, float)):
            self._default = (float(default_response),)
        else:
            self._default = tuple(float(v) for v in default_response)
        self._base_probability = base_probability
        self._noise_std = noise_std
        self._seed = seed
        self._model_version = model_version
        self._prompt_version = prompt_version
        self.batch_call_count = 0
        self.request_count = 0

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def forecast_batch(self, requests: Sequence[ForecastRequest]) -> Sequence[ForecastDraw]:
        if not requests:
            return ()
        self.batch_call_count += 1
        self.request_count += len(requests)
        content_hash = requests[0].content_hash
        draws: list[ForecastDraw] = []
        for req in requests:
            probability = self._probability_for_run(content_hash, req.run_index, req.content)
            draws.append(
                ForecastDraw(
                    probability=probability,
                    run_index=req.run_index,
                    model_version=self._model_version,
                    prompt_version=self._prompt_version,
                    provenance={
                        "fixture": True,
                        "content_hash": content_hash,
                        "run_index": req.run_index,
                    },
                )
            )
        return tuple(draws)

    def _probability_for_run(self, content_hash: str, run_index: int, content: str) -> float:
        if content_hash in self._responses:
            seq = self._responses[content_hash]
            idx = min(run_index, len(seq) - 1)
            return float(np.clip(seq[idx], 0.0, 1.0))
        if self._base_probability is not None:
            rng = np.random.default_rng(self._seed + run_index + hash(content_hash) % 10_000)
            draw = float(rng.normal(self._base_probability, self._noise_std))
            return float(np.clip(draw, 0.0, 1.0))
        idx = min(run_index, len(self._default) - 1)
        return float(np.clip(self._default[idx], 0.0, 1.0))


# System instruction kept domain-agnostic: the caller's per-request ``prompt``
# (and document content) supplies all domain context.
_FORECAST_SYSTEM = (
    "You are a careful probabilistic forecaster. Read the question and the "
    "document, then respond with ONLY a JSON object of the form "
    '{"probability": p} where p is your probability of the event in [0, 1]. '
    "Do not include any prose outside the JSON object."
)


def _compose_user(prompt: str, content: str) -> str:
    """Combine the per-request prompt with the document body."""
    return f"{prompt}\n\nDocument:\n{content}"


def _coerce_probability(payload: Mapping[str, Any]) -> float:
    """Validate and extract ``probability`` in [0, 1] from a parsed payload."""
    if "probability" not in payload:
        msg = f"forecast payload missing 'probability' key: {payload!r}"
        raise MalformedLLMOutput(msg)
    try:
        value = float(payload["probability"])
    except (TypeError, ValueError) as exc:
        msg = f"forecast 'probability' is not a number: {payload['probability']!r}"
        raise MalformedLLMOutput(msg) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        msg = f"forecast 'probability' out of [0, 1]: {value!r}"
        raise MalformedLLMOutput(msg)
    return value


class BedrockForecastLLM:
    """Structured-LLM-backed ``ForecastLLM``: elicits an absolute probability per run.

    Implements the ``ForecastLLM`` protocol over a shared ``StructuredLLMClient``
    (the direct Anthropic API by default, or Bedrock). The per-run ``prompt``
    arrives on each ``ForecastRequest`` so this adapter carries no domain
    assumptions (CLAUDE.md section 11).
    """

    def __init__(
        self,
        client: StructuredLLMClient,
        *,
        prompt_version: str = "delphi_forecast_v1",
        system: str = _FORECAST_SYSTEM,
    ) -> None:
        self._client = client
        self._prompt_version = prompt_version
        self._system = system

    @property
    def model_version(self) -> str:
        return self._client.model_id

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def forecast_batch(self, requests: Sequence[ForecastRequest]) -> Sequence[ForecastDraw]:
        if not requests:
            return ()
        prompts = [
            StructuredPrompt(
                system=self._system,
                user=_compose_user(req.prompt, req.content),
                run_index=req.run_index,
            )
            for req in requests
        ]
        payloads = self._client.invoke_structured_batch(prompts)
        draws: list[ForecastDraw] = []
        for req, payload in zip(requests, payloads, strict=True):
            probability = _coerce_probability(payload)
            draws.append(
                ForecastDraw(
                    probability=probability,
                    run_index=req.run_index,
                    model_version=self.model_version,
                    prompt_version=self._prompt_version,
                    provenance={
                        "provider": self._client.provider,
                        "model_id": self._client.model_id,
                        "content_hash": req.content_hash,
                        "run_index": req.run_index,
                    },
                )
            )
        return tuple(draws)
