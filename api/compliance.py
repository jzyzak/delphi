"""Compliance features (C10.4).

Enterprise/compliance surface (CLAUDE.md §9): provider opt-out, per-request
usage/cost reporting, and data-retention opt-out. Provider opt-out is honored
*end-to-end* — if the caller opts out of every available provider the request is
refused rather than silently served by a forbidden one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ComplianceOptions",
    "ProviderOptOutError",
    "UsageReport",
    "filter_providers",
    "should_retain",
    "usage_for",
]


class ProviderOptOutError(RuntimeError):
    """Raised when every available provider has been opted out (§9)."""


@dataclass(frozen=True)
class ComplianceOptions:
    """Per-request compliance choices."""

    provider_opt_out: frozenset[str] = field(default_factory=frozenset)
    retention_opt_out: bool = False


def filter_providers(available: Iterable[str], options: ComplianceOptions) -> tuple[str, ...]:
    """Return the permitted providers; raise if the opt-out leaves none (fail-closed)."""
    permitted = tuple(p for p in available if p not in options.provider_opt_out)
    if not permitted:
        msg = "all available providers were opted out; cannot serve the request."
        raise ProviderOptOutError(msg)
    return permitted


def should_retain(options: ComplianceOptions) -> bool:
    """Whether the caller's request/response may be retained (logged)."""
    return not options.retention_opt_out


class UsageReport(BaseModel):
    """Per-request usage + (optional) cost. Pricing is never hardcoded (§7)."""

    model_config = ConfigDict(frozen=True)

    tier: str
    model_calls: int
    cost_usd: float | None = None


def usage_for(tier: str, *, model_calls: int, price_per_call: float | None = None) -> UsageReport:
    """Build a usage report; cost is ``None`` unless a price is supplied."""
    cost = None if price_per_call is None else round(model_calls * price_per_call, 6)
    return UsageReport(tier=tier, model_calls=model_calls, cost_usd=cost)
