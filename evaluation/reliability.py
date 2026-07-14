"""Reliability diagrams + calibration error (C6.2).

A forecaster that says 70% must be right 70% of the time (CLAUDE.md §1). This
module bins forecasts by predicted probability and compares each bin's mean
prediction to its empirical frequency, summarizing the gap as Expected (ECE) and
Maximum (MCE) Calibration Error, and rendering a text reliability diagram.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ReliabilityBin", "ReliabilityDiagram", "reliability"]


@dataclass(frozen=True)
class ReliabilityBin:
    """One probability bin: predicted vs realized frequency."""

    lo: float
    hi: float
    count: int
    mean_prediction: float
    mean_outcome: float

    @property
    def gap(self) -> float:
        return abs(self.mean_prediction - self.mean_outcome)


@dataclass(frozen=True)
class ReliabilityDiagram:
    """Binned reliability with ECE/MCE and a text render."""

    bins: tuple[ReliabilityBin, ...]
    ece: float
    mce: float
    n: int

    def render(self) -> str:
        """Render a markdown reliability table (diagram artifact, C6.2.c)."""
        lines = [
            "| bin | n | mean_pred | mean_outcome | gap |",
            "| --- | --- | --- | --- | --- |",
        ]
        for b in self.bins:
            lines.append(
                f"| [{b.lo:.2f}, {b.hi:.2f}) | {b.count} | {b.mean_prediction:.3f} "
                f"| {b.mean_outcome:.3f} | {b.gap:.3f} |"
            )
        lines.append(f"\nECE={self.ece:.4f}  MCE={self.mce:.4f}  n={self.n}")
        return "\n".join(lines)


def reliability(
    probabilities: Sequence[float], outcomes: Sequence[float], *, n_bins: int = 10
) -> ReliabilityDiagram:
    """Bin forecasts into ``n_bins`` equal-width bins and compute ECE/MCE."""
    if n_bins < 1:
        msg = "n_bins must be >= 1."
        raise ValueError(msg)
    if len(probabilities) != len(outcomes):
        msg = "probabilities and outcomes must have equal length."
        raise ValueError(msg)
    n = len(probabilities)
    if n == 0:
        msg = "cannot compute reliability over an empty set."
        raise ValueError(msg)

    edges = [i / n_bins for i in range(n_bins + 1)]
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for p, o in zip(probabilities, outcomes, strict=True):
        if not 0.0 <= p <= 1.0:
            msg = f"probability out of [0, 1]: {p!r}"
            raise ValueError(msg)
        # p == 1.0 belongs in the final bin (upper edge inclusive there only).
        idx = min(int(p * n_bins), n_bins - 1)
        buckets[idx].append((p, o))

    bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        count = len(bucket)
        mean_pred = sum(p for p, _ in bucket) / count
        mean_out = sum(o for _, o in bucket) / count
        b = ReliabilityBin(edges[i], edges[i + 1], count, mean_pred, mean_out)
        bins.append(b)
        ece += (count / n) * b.gap
        mce = max(mce, b.gap)
    return ReliabilityDiagram(bins=tuple(bins), ece=ece, mce=mce, n=n)
