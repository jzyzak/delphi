"""Hand-computed ECE/MCE fixtures for the reliability diagram (C6.2)."""

from __future__ import annotations

import pytest

from evaluation.reliability import reliability


def test_perfect_calibration_zero_ece() -> None:
    probs = [0.0, 0.0, 1.0, 1.0]
    outcomes = [0.0, 0.0, 1.0, 1.0]
    diag = reliability(probs, outcomes)
    assert diag.ece == pytest.approx(0.0)
    assert diag.mce == pytest.approx(0.0)


def test_hand_computed_ece() -> None:
    # Two full bins, each with gap 0.2, equal weight -> ECE 0.2, MCE 0.2.
    probs = [0.2, 0.2, 0.8, 0.8]
    outcomes = [0.0, 0.0, 1.0, 1.0]
    diag = reliability(probs, outcomes)
    assert diag.ece == pytest.approx(0.2)
    assert diag.mce == pytest.approx(0.2)
    assert diag.n == 4


def test_unequal_weight_ece() -> None:
    # Bin [0.2): 3 samples gap 0.2; bin [0.8): 1 sample gap 0.2 -> ECE 0.2 still.
    probs = [0.2, 0.2, 0.2, 0.8]
    outcomes = [0.0, 0.0, 0.0, 1.0]
    diag = reliability(probs, outcomes)
    assert diag.ece == pytest.approx(0.2)


def test_probability_one_lands_in_last_bin() -> None:
    diag = reliability([1.0], [1.0], n_bins=10)
    assert diag.bins[0].lo == pytest.approx(0.9)
    assert diag.bins[0].hi == pytest.approx(1.0)


def test_render_contains_summary() -> None:
    rendered = reliability([0.2, 0.8], [0.0, 1.0]).render()
    assert "ECE=" in rendered
    assert "mean_pred" in rendered


def test_validation() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        reliability([0.5], [1.0], n_bins=0)
    with pytest.raises(ValueError, match="equal length"):
        reliability([0.5], [1.0, 0.0])
    with pytest.raises(ValueError, match="empty"):
        reliability([], [])
    with pytest.raises(ValueError, match="out of"):
        reliability([1.5], [1.0])
