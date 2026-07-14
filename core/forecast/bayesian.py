"""Explicit Bayesian forecast formation: PIT prior x evidence log-LR -> posterior."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.llm import StructuredLLMClient, StructuredPrompt
from common.llm.errors import MalformedLLMOutput
from core.forecast.ensemble import (
    DEFAULT_TRIM_FRACTION,
    Aggregator,
    EnsembleForecast,
    build_ensemble,
)
from core.forecast.llm import ForecastDraw

_PROB_EPS = 1e-12


def _clamp_probability(p: float) -> float:
    """Clamp ``p`` to the open unit interval for stable log-odds."""
    return min(max(p, _PROB_EPS), 1.0 - _PROB_EPS)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _logit(p: float) -> float:
    """Log-odds of a probability in the open unit interval."""
    clamped = _clamp_probability(p)
    return math.log(clamped / (1.0 - clamped))


def _validate_base_rate(base_rate: float) -> None:
    """Reject non-finite or boundary probabilities unsuitable as a Bayesian prior."""
    if not math.isfinite(base_rate):
        msg = f"base_rate must be finite, got {base_rate!r}"
        raise ValueError(msg)
    if base_rate <= 0.0 or base_rate >= 1.0:
        msg = f"base_rate must be in the open interval (0, 1) for log-odds prior, got {base_rate!r}"
        raise ValueError(msg)


def _validate_evidence_log_lr(evidence_log_lr: float) -> None:
    if not math.isfinite(evidence_log_lr):
        msg = f"evidence_log_lr must be finite, got {evidence_log_lr!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class Posterior:
    """Bayesian update in log-odds space with full audit trail.

    ``prior`` is the PIT base rate (12). ``evidence_log_lr`` is the model's
    evidence strength relative to that prior. ``posterior`` is sigmoid(
    prior_logodds + evidence_log_lr). Precedes calibration (19).
    """

    prior: float
    prior_logodds: float
    evidence_log_lr: float
    evidence_lr: float
    posterior_logodds: float
    posterior: float
    provenance: Mapping[str, Any]


def posterior(*, base_rate: float, evidence_log_lr: float) -> Posterior:
    """Bayesian update in log-odds space: posterior_logodds = logit(base_rate) + evidence_log_lr.

    ``base_rate`` is the PIT prior (12); ``evidence_log_lr`` is the model's evidence strength
    relative to that prior (elicited as a likelihood ratio, NOT an absolute probability).
    Returns ``Posterior{ prior, evidence_lr, posterior }`` — recorded for audit. Precedes
    calibration (19).
    """
    _validate_base_rate(base_rate)
    _validate_evidence_log_lr(evidence_log_lr)
    prior_logodds = _logit(base_rate)
    post_logodds = prior_logodds + evidence_log_lr
    post_prob = _sigmoid(post_logodds)
    evidence_lr = math.exp(evidence_log_lr)
    audit: dict[str, Any] = {
        "method": "bayesian_logodds_update",
        "prior": base_rate,
        "prior_logodds": prior_logodds,
        "evidence_log_lr": evidence_log_lr,
        "evidence_lr": evidence_lr,
        "posterior_logodds": post_logodds,
        "posterior": post_prob,
    }
    return Posterior(
        prior=base_rate,
        prior_logodds=prior_logodds,
        evidence_log_lr=evidence_log_lr,
        evidence_lr=evidence_lr,
        posterior_logodds=post_logodds,
        posterior=post_prob,
        provenance=audit,
    )


@dataclass(frozen=True)
class LikelihoodRequest:
    """One independent evidence log-LR elicitation request within a batched call."""

    content: str
    content_hash: str
    run_index: int
    prompt: str
    prior: float


class LikelihoodDraw(BaseModel):
    """Structured output from a single evidence log-LR elicitation draw.

    Emits an evidence log-likelihood-ratio relative to the prior — never an
    absolute probability. Downstream Bayesian combination converts to posterior.
    """

    model_config = ConfigDict(frozen=True)

    evidence_log_lr: float
    run_index: int = Field(ge=0)
    model_version: str
    prompt_version: str
    provenance: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("provenance")
    @classmethod
    def _freeze_provenance(cls, v: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(v)


@runtime_checkable
class EvidenceLikelihoodLLM(Protocol):
    """Batched evidence log-LR elicitation seam — no absolute-probability path."""

    @property
    def model_version(self) -> str: ...

    @property
    def prompt_version(self) -> str: ...

    def elicit_log_lr_batch(
        self, requests: Sequence[LikelihoodRequest]
    ) -> Sequence[LikelihoodDraw]:
        """Return one structured log-LR draw per request in a single batched call."""
        ...


class FixtureEvidenceLikelihoodLLM:
    """Deterministic evidence log-LR LLM for tests (no network).

    Supports explicit per-run log-LR sequences keyed by ``content_hash``, or a
    seeded base+noise mode for variance-reduction fixtures.
    """

    def __init__(
        self,
        responses: Mapping[str, Sequence[float]] | None = None,
        *,
        model_version: str = "fixture-likelihood-v1",
        prompt_version: str = "evidence_log_lr_v1",
        default_response: Sequence[float] | float | None = None,
        base_log_lr: float | None = None,
        noise_std: float = 0.15,
        seed: int = 42,
    ) -> None:
        self._responses = {k: tuple(v) for k, v in (responses or {}).items()}
        if default_response is None:
            self._default: tuple[float, ...] = (0.0,)
        elif isinstance(default_response, (int, float)):
            self._default = (float(default_response),)
        else:
            self._default = tuple(float(v) for v in default_response)
        self._base_log_lr = base_log_lr
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

    def elicit_log_lr_batch(
        self, requests: Sequence[LikelihoodRequest]
    ) -> Sequence[LikelihoodDraw]:
        if not requests:
            return ()
        self.batch_call_count += 1
        self.request_count += len(requests)
        content_hash = requests[0].content_hash
        draws: list[LikelihoodDraw] = []
        for req in requests:
            log_lr = self._log_lr_for_run(content_hash, req.run_index, req.content)
            draws.append(
                LikelihoodDraw(
                    evidence_log_lr=log_lr,
                    run_index=req.run_index,
                    model_version=self._model_version,
                    prompt_version=self._prompt_version,
                    provenance={
                        "fixture": True,
                        "content_hash": content_hash,
                        "run_index": req.run_index,
                        "prior": req.prior,
                        "elicitation": "evidence_log_lr",
                    },
                )
            )
        return tuple(draws)

    def _log_lr_for_run(self, content_hash: str, run_index: int, content: str) -> float:
        if content_hash in self._responses:
            seq = self._responses[content_hash]
            idx = min(run_index, len(seq) - 1)
            return float(seq[idx])
        if self._base_log_lr is not None:
            rng = np.random.default_rng(self._seed + run_index + hash(content_hash) % 10_000)
            return float(rng.normal(self._base_log_lr, self._noise_std))
        idx = min(run_index, len(self._default) - 1)
        return float(self._default[idx])


# System instruction enforces the section 10 contract: elicit an evidence
# likelihood-ratio relative to the prior, never an absolute probability.
_LIKELIHOOD_SYSTEM = (
    "You are a careful probabilistic forecaster. You are given a prior "
    "probability and a document of evidence. Respond with ONLY a JSON object of "
    'the form {"evidence_log_lr": x} where x is the natural-log likelihood ratio '
    "of the evidence relative to the prior (positive supports the event, "
    "negative weighs against it, 0 is uninformative). Do NOT output an absolute "
    "probability and include no prose outside the JSON object."
)


def _compose_likelihood_user(prompt: str, *, prior: float, content: str) -> str:
    """Combine the per-request prompt, prior, and document body."""
    return f"{prompt}\n\nPrior probability: {prior}\n\nDocument:\n{content}"


def _coerce_log_lr(payload: Mapping[str, Any]) -> float:
    """Validate and extract a finite ``evidence_log_lr`` from a parsed payload."""
    if "evidence_log_lr" not in payload:
        msg = f"likelihood payload missing 'evidence_log_lr' key: {payload!r}"
        raise MalformedLLMOutput(msg)
    try:
        value = float(payload["evidence_log_lr"])
    except (TypeError, ValueError) as exc:
        msg = f"'evidence_log_lr' is not a number: {payload['evidence_log_lr']!r}"
        raise MalformedLLMOutput(msg) from exc
    if not math.isfinite(value):
        msg = f"'evidence_log_lr' must be finite, got {value!r}"
        raise MalformedLLMOutput(msg)
    return value


class BedrockEvidenceLikelihoodLLM:
    """Structured-LLM-backed ``EvidenceLikelihoodLLM``: elicits evidence log-LRs per run.

    Implements the ``EvidenceLikelihoodLLM`` protocol over a shared
    ``StructuredLLMClient`` (the direct Anthropic API by default, or Bedrock).
    Never elicits an absolute probability (CLAUDE.md sections 10/14); the per-run
    ``prompt`` and ``prior`` arrive on each ``LikelihoodRequest``.
    """

    def __init__(
        self,
        client: StructuredLLMClient,
        *,
        prompt_version: str = "evidence_log_lr_v1",
        system: str = _LIKELIHOOD_SYSTEM,
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

    def elicit_log_lr_batch(
        self, requests: Sequence[LikelihoodRequest]
    ) -> Sequence[LikelihoodDraw]:
        if not requests:
            return ()
        prompts = [
            StructuredPrompt(
                system=self._system,
                user=_compose_likelihood_user(req.prompt, prior=req.prior, content=req.content),
                run_index=req.run_index,
            )
            for req in requests
        ]
        payloads = self._client.invoke_structured_batch(prompts)
        draws: list[LikelihoodDraw] = []
        for req, payload in zip(requests, payloads, strict=True):
            log_lr = _coerce_log_lr(payload)
            draws.append(
                LikelihoodDraw(
                    evidence_log_lr=log_lr,
                    run_index=req.run_index,
                    model_version=self.model_version,
                    prompt_version=self._prompt_version,
                    provenance={
                        "provider": self._client.provider,
                        "model_id": self._client.model_id,
                        "content_hash": req.content_hash,
                        "run_index": req.run_index,
                        "prior": req.prior,
                        "elicitation": "evidence_log_lr",
                    },
                )
            )
        return tuple(draws)


@dataclass(frozen=True)
class BayesianEnsembleResult:
    """Per-run Bayesian posteriors aggregated into an ensemble forecast."""

    ensemble: EnsembleForecast
    posteriors: tuple[Posterior, ...]
    prior: float
    prior_logodds: float


def _posterior_to_forecast_draw(
    post: Posterior,
    *,
    likelihood_draw: LikelihoodDraw,
) -> ForecastDraw:
    provenance: dict[str, Any] = {
        "formation": "bayesian",
        "prior": post.prior,
        "prior_logodds": post.prior_logodds,
        "evidence_log_lr": post.evidence_log_lr,
        "evidence_lr": post.evidence_lr,
        "posterior_logodds": post.posterior_logodds,
        "posterior": post.posterior,
        "likelihood_provenance": dict(likelihood_draw.provenance),
    }
    return ForecastDraw(
        probability=post.posterior,
        run_index=likelihood_draw.run_index,
        model_version=likelihood_draw.model_version,
        prompt_version=likelihood_draw.prompt_version,
        provenance=provenance,
    )


def build_bayesian_ensemble(
    *,
    base_rate: float,
    likelihood_draws: Sequence[LikelihoodDraw],
    knowledge_time: datetime,
    aggregator: Aggregator,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
) -> BayesianEnsembleResult:
    """Combine per-run evidence log-LRs with a PIT prior, then aggregate posteriors.

    For each draw: ``posterior_logodds = logit(base_rate) + evidence_log_lr``.
    Per-run posterior probabilities feed the existing robust ensemble aggregator
    (18). The result precedes calibration (19) and supervisor reconciliation (21).
    """
    _validate_base_rate(base_rate)
    if not likelihood_draws:
        msg = "likelihood_draws must be non-empty"
        raise ValueError(msg)

    prior_logodds = _logit(base_rate)
    posteriors: list[Posterior] = []
    forecast_draws: list[ForecastDraw] = []
    for draw in likelihood_draws:
        post = posterior(base_rate=base_rate, evidence_log_lr=draw.evidence_log_lr)
        posteriors.append(post)
        forecast_draws.append(_posterior_to_forecast_draw(post, likelihood_draw=draw))

    ensemble = build_ensemble(
        forecast_draws,
        aggregator=aggregator,
        knowledge_time=knowledge_time,
        trim_fraction=trim_fraction,
    )
    enriched_provenance: dict[str, Any] = {
        **dict(ensemble.provenance),
        "formation": "bayesian_per_run",
        "prior": base_rate,
        "prior_logodds": prior_logodds,
        "per_run_posteriors": [p.provenance for p in posteriors],
    }
    enriched = EnsembleForecast(
        probability=ensemble.probability,
        uncertainty=ensemble.uncertainty,
        n=ensemble.n,
        aggregator=ensemble.aggregator,
        trim_fraction=ensemble.trim_fraction,
        knowledge_time=ensemble.knowledge_time,
        draws=ensemble.draws,
        provenance=enriched_provenance,
    )
    return BayesianEnsembleResult(
        ensemble=enriched,
        posteriors=tuple(posteriors),
        prior=base_rate,
        prior_logodds=prior_logodds,
    )


def build_likelihood_requests(
    *,
    content: str,
    content_hash: str,
    prior: float,
    prompt: str,
    n: int,
) -> tuple[LikelihoodRequest, ...]:
    """Build N independent log-LR elicitation requests sharing one document."""
    if n <= 0:
        msg = "n must be positive"
        raise ValueError(msg)
    _validate_base_rate(prior)
    return tuple(
        LikelihoodRequest(
            content=content,
            content_hash=content_hash,
            run_index=i,
            prompt=prompt,
            prior=prior,
        )
        for i in range(n)
    )


def elicit_and_build_bayesian_ensemble(
    llm: EvidenceLikelihoodLLM,
    *,
    content: str,
    content_hash: str,
    base_rate: float,
    prompt: str,
    knowledge_time: datetime,
    n: int = 10,
    aggregator: Aggregator = "median",
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
) -> BayesianEnsembleResult:
    """End-to-end: batch-elicit log-LRs, combine with prior, aggregate posteriors."""
    requests = build_likelihood_requests(
        content=content,
        content_hash=content_hash,
        prior=base_rate,
        prompt=prompt,
        n=n,
    )
    draws = llm.elicit_log_lr_batch(requests)
    return build_bayesian_ensemble(
        base_rate=base_rate,
        likelihood_draws=draws,
        knowledge_time=knowledge_time,
        aggregator=aggregator,
        trim_fraction=trim_fraction,
    )
