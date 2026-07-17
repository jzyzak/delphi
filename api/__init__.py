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
    ClassificationResult,
    ClassifyAPIResponse,
    ForecastAPIRequest,
    ForecastAPIResponse,
    FormalizeAPIResponse,
    FormalizedQuestion,
    IntakeAPIRequest,
    Message,
    build_classify_response,
    build_formalize_response,
    build_response,
)
from api.server import DelphiApp, serve, wsgi_application

__all__ = [
    "Choice",
    "ClassificationResult",
    "ClassifyAPIResponse",
    "ComplianceOptions",
    "ConfidenceBand",
    "DelphiApp",
    "EvidenceProvenance",
    "ForecastAPIRequest",
    "ForecastAPIResponse",
    "ForecastEnvelope",
    "ForecastService",
    "FormalizeAPIResponse",
    "FormalizedQuestion",
    "IntakeAPIRequest",
    "Message",
    "ProviderOptOutError",
    "UsageReport",
    "build_classify_response",
    "build_envelope",
    "build_formalize_response",
    "build_response",
    "filter_providers",
    "serve",
    "should_retain",
    "usage_for",
    "wsgi_application",
]
