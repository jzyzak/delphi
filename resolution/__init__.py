"""Ground-truth resolution once a question closes.

App layer: reads the resolution criteria captured at intake, extracts the
outcome with provenance, and writes the immutable ``resolution`` record.
"""

from __future__ import annotations

from resolution.service import ResolutionRun, ResolutionService
from resolution.sources import (
    MappingResolutionSource,
    ResolutionSource,
    ResolvedOutcome,
)

__all__ = [
    "MappingResolutionSource",
    "ResolutionRun",
    "ResolutionService",
    "ResolutionSource",
    "ResolvedOutcome",
]
