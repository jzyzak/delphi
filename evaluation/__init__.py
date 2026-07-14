"""Proper scoring, reliability, baselines, bootstrap CIs — plus the trials ledger
and guarded holdout. THE HOUSE (CLAUDE.md §2.2, §2.4).

This package may only ever be made *stricter*. Forecast/agent code must never
modify, weaken, or bypass it to make a number look better.
"""

from __future__ import annotations

from evaluation.aggregate import (
    ScoreSummary,
    baseline_delta,
    bootstrap_ci,
    per_domain_summary,
    summarize_scores,
)
from evaluation.baselines import Baseline
from evaluation.calibration_split import (
    CalibrationArtifact,
    IsotonicRecalibrator,
    assign_calibration_split,
    fit_calibration_artifact,
)
from evaluation.harness import EvalHarness, HoldoutUnavailable, TrialsLedgerExhausted
from evaluation.leakage_audit import LeakageAudit, audit_suite
from evaluation.reliability import ReliabilityDiagram, reliability
from evaluation.report import EvalContext, EvalInputs, render_leakage_audit, render_report
from evaluation.scoring import (
    BrierScorer,
    CRPSScorer,
    LogScorer,
    ScoredRecord,
    Scorer,
    brier_score,
    crps_from_quantiles,
    log_score,
)

__all__ = [
    "Baseline",
    "BrierScorer",
    "CRPSScorer",
    "CalibrationArtifact",
    "EvalContext",
    "EvalHarness",
    "EvalInputs",
    "HoldoutUnavailable",
    "IsotonicRecalibrator",
    "LeakageAudit",
    "LogScorer",
    "ReliabilityDiagram",
    "ScoreSummary",
    "ScoredRecord",
    "Scorer",
    "TrialsLedgerExhausted",
    "assign_calibration_split",
    "audit_suite",
    "baseline_delta",
    "bootstrap_ci",
    "brier_score",
    "crps_from_quantiles",
    "fit_calibration_artifact",
    "log_score",
    "per_domain_summary",
    "reliability",
    "render_leakage_audit",
    "render_report",
    "summarize_scores",
]
