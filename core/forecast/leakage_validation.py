"""Self-validation for the leakage judge on a labeled fixture set."""

from __future__ import annotations

from collections.abc import Sequence

import structlog
from pydantic import BaseModel, ConfigDict, Field

from core.forecast.leakage_judge import LeakageJudge, Trace

_LOG = structlog.get_logger(__name__)
DEFAULT_RECALL_FLOOR = 0.95


class LabeledTrace(BaseModel):
    """One labeled trace for judge validation."""

    model_config = ConfigDict(frozen=True)

    trace: Trace
    is_leak: bool


class ValidationReport(BaseModel):
    """Recall/precision metrics recorded for a labeled fixture set."""

    model_config = ConfigDict(frozen=True)

    recall: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    n: int = Field(ge=0)
    recall_floor: float = Field(ge=0.0, le=1.0)
    meets_recall_floor: bool


def validate_judge(
    judge: LeakageJudge,
    labeled: Sequence[LabeledTrace],
    *,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
) -> ValidationReport:
    """Measure and record recall/precision on a labeled set.

    Contract: tuned for high recall — noisy precision is acceptable. Raises
    ``ValueError`` when recall falls below ``recall_floor``.
    """
    if recall_floor < 0.0 or recall_floor > 1.0:
        msg = "recall_floor must be in [0, 1]"
        raise ValueError(msg)

    tp = fp = tn = fn = 0
    for item in labeled:
        verdict = judge.audit(item.trace)
        predicted = verdict.flagged
        actual = item.is_leak
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    n = tp + fp + tn + fn
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    meets = recall >= recall_floor

    report = ValidationReport(
        recall=recall,
        precision=precision,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        n=n,
        recall_floor=recall_floor,
        meets_recall_floor=meets,
    )
    _LOG.info(
        "leakage_judge_validated",
        recall=report.recall,
        precision=report.precision,
        n=report.n,
        meets_recall_floor=report.meets_recall_floor,
        judge_model=judge.model_version,
        judge_prompt=judge.prompt_version,
    )
    if not meets:
        msg = (
            f"Leakage judge recall {recall:.3f} below floor {recall_floor:.3f}; "
            "tune for high recall before relying on unflagged traces."
        )
        raise ValueError(msg)
    return report


__all__ = [
    "DEFAULT_RECALL_FLOOR",
    "LabeledTrace",
    "ValidationReport",
    "validate_judge",
]
