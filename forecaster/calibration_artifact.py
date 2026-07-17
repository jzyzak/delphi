"""Calibration-artifact loading for the live forecast chain (C4.5 wiring).

Loads the JSON artifact written by ``delphi calibration fit`` (fit ONLY on the
disjoint calibration split, CLAUDE.md §2.5) into the apply-only
:class:`~core.forecast.calibration.FrozenCalibration` the chain consumes. The
artifact hash travels into forecast provenance so every registry record names
exactly which fitted map produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.forecast.calibration import FrozenCalibration

__all__ = [
    "CalibrationArtifactError",
    "artifact_filename",
    "load_calibration_artifact",
]


class CalibrationArtifactError(RuntimeError):
    """A calibration artifact is missing, corrupt, or schema-incompatible."""


def _hash_payload(data: dict[str, object]) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def artifact_filename(data: dict[str, object], *, date: str) -> str:
    """Versioned artifact filename: ``calibration-<date>-<hash8>.json``."""
    return f"calibration-{date}-{_hash_payload(data)[:8]}.json"


def load_calibration_artifact(path: str | Path) -> FrozenCalibration:
    """Load and validate a fitted calibration artifact from ``path``.

    Raises :class:`CalibrationArtifactError` with an actionable message when the
    file is missing, is not valid JSON, or does not match the artifact schema —
    a misconfigured artifact must fail loudly, never silently fall back.
    """
    file = Path(path).expanduser()
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        msg = f"calibration artifact not found at {file}"
        raise CalibrationArtifactError(msg) from exc
    except OSError as exc:
        msg = f"calibration artifact at {file} is unreadable: {exc}"
        raise CalibrationArtifactError(msg) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"calibration artifact at {file} is not valid JSON: {exc}"
        raise CalibrationArtifactError(msg) from exc
    if not isinstance(data, dict):
        msg = f"calibration artifact at {file} must be a JSON object."
        raise CalibrationArtifactError(msg)
    try:
        return FrozenCalibration.from_dict(data, artifact_hash=_hash_payload(data))
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"calibration artifact at {file} does not match the schema: {exc}"
        raise CalibrationArtifactError(msg) from exc
