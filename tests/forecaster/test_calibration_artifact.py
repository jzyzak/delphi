"""Unit tests for calibration-artifact loading (C4.5 wiring)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forecaster.calibration_artifact import (
    CalibrationArtifactError,
    artifact_filename,
    load_calibration_artifact,
)

_ARTIFACT = {
    "schema_version": 1,
    "method": "platt",
    "recalibrator": {"a": 1.5, "b": 0.1, "n": 40},
    "alpha": 1.25,
    "floor": 0.01,
    "n": 40,
    "fitted": {"seed": 0, "fitted_at": "2026-07-01T00:00:00+00:00"},
}


def _write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestLoad:
    def test_loads_valid_artifact(self, tmp_path: Path) -> None:
        learned = load_calibration_artifact(_write(tmp_path, _ARTIFACT))
        assert learned.method == "platt"
        assert learned.alpha == 1.25
        assert learned.floor == 0.01
        assert learned.n == 40
        assert learned.fitted_meta["seed"] == 0

    def test_hash_is_stable_and_content_addressed(self, tmp_path: Path) -> None:
        first = load_calibration_artifact(_write(tmp_path, _ARTIFACT))
        second = load_calibration_artifact(_write(tmp_path, _ARTIFACT))
        changed = load_calibration_artifact(_write(tmp_path, {**_ARTIFACT, "alpha": 2.0}))
        assert first.artifact_hash == second.artifact_hash
        assert first.artifact_hash != changed.artifact_hash
        assert len(first.artifact_hash) == 64

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationArtifactError, match="not found"):
            load_calibration_artifact(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "artifact.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(CalibrationArtifactError, match="not valid JSON"):
            load_calibration_artifact(path)

    def test_non_object_json_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationArtifactError, match="JSON object"):
            load_calibration_artifact(_write(tmp_path, [1, 2, 3]))

    def test_schema_mismatch_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationArtifactError, match="schema"):
            load_calibration_artifact(_write(tmp_path, {"alpha": 1.0}))

    def test_unknown_method_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationArtifactError, match="schema"):
            load_calibration_artifact(_write(tmp_path, {**_ARTIFACT, "method": "spline"}))

    def test_unreadable_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationArtifactError, match="unreadable"):
            load_calibration_artifact(tmp_path)  # a directory is unreadable as a file


class TestFilename:
    def test_versioned_name_embeds_date_and_hash(self) -> None:
        name = artifact_filename(_ARTIFACT, date="20260715")
        assert name.startswith("calibration-20260715-")
        assert name.endswith(".json")
        assert name == artifact_filename(_ARTIFACT, date="20260715")
        assert name != artifact_filename({**_ARTIFACT, "alpha": 2.0}, date="20260715")
