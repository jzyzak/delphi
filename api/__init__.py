"""Published OpenAI-compatible endpoint (DELPHI / DELPHI Deep) (§9).

App layer: the product surface. Hides the *how* (routing/orchestration) and
exposes the evidence and track record buyers need to trust the probability.
"""

from __future__ import annotations

from api.compliance import (
    ComplianceOptions,
    ProviderOptOutError,
    UsageReport,
    filter_providers,
    should_retain,
    usage_for,
)
from api.envelope import ConfidenceBand, EvidenceProvenance, ForecastEnvelope, build_envelope
from api.routes import ForecastService
from api.schema import (
    Choice,
    ForecastAPIRequest,
    ForecastAPIResponse,
    Message,
    build_response,
)
from api.server import DelphiApp, serve, wsgi_application

__all__ = [
    "Choice",
    "ComplianceOptions",
    "ConfidenceBand",
    "DelphiApp",
    "EvidenceProvenance",
    "ForecastAPIRequest",
    "ForecastAPIResponse",
    "ForecastEnvelope",
    "ForecastService",
    "Message",
    "ProviderOptOutError",
    "UsageReport",
    "build_envelope",
    "build_response",
    "filter_providers",
    "serve",
    "should_retain",
    "usage_for",
    "wsgi_application",
]
