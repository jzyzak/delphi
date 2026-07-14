"""Registry-wide leakage-rate estimation and regression surfacing."""

from __future__ import annotations

from collections import defaultdict

import structlog

from core.forecast.leakage_judge import (
    ComponentLeakageRate,
    LeakageJudge,
    LeakageRegression,
    LeakageReport,
    RegistrySlice,
    TraceComponent,
)

_LOG = structlog.get_logger(__name__)
DEFAULT_SPIKE_THRESHOLD = 0.05


def estimate_leakage_rate(
    judge: LeakageJudge,
    slice: RegistrySlice,
    *,
    baseline: LeakageReport | None = None,
    spike_threshold: float = DEFAULT_SPIKE_THRESHOLD,
) -> LeakageReport:
    """Batch audit a slice of recorded forecasts.

    Contract: report per-component and aggregate leakage-rate estimates and
    surface regressions (spikes vs ``baseline`` or absolute ``spike_threshold``).
    """
    if spike_threshold < 0.0:
        msg = "spike_threshold must be non-negative"
        raise ValueError(msg)

    totals: dict[TraceComponent, int] = defaultdict(int)
    flagged_counts: dict[TraceComponent, int] = defaultdict(int)
    total_flagged = 0

    for trace in slice.traces:
        verdict = judge.audit(trace)
        totals[trace.component] += 1
        if verdict.flagged:
            flagged_counts[trace.component] += 1
            total_flagged += 1

    total = len(slice.traces)
    aggregate_rate = total_flagged / total if total > 0 else 0.0

    by_component: list[ComponentLeakageRate] = []
    for component in TraceComponent:
        comp_total = totals.get(component, 0)
        if comp_total == 0:
            continue
        comp_flagged = flagged_counts.get(component, 0)
        by_component.append(
            ComponentLeakageRate(
                component=component,
                total=comp_total,
                flagged=comp_flagged,
                rate=comp_flagged / comp_total,
            )
        )

    baseline_rates: dict[TraceComponent, float] = {}
    if baseline is not None:
        baseline_rates = {item.component: item.rate for item in baseline.by_component}

    regressions: list[LeakageRegression] = []
    for item in by_component:
        baseline_rate = baseline_rates.get(item.component)
        if baseline_rate is not None:
            delta = item.rate - baseline_rate
            if delta > spike_threshold:
                regressions.append(
                    LeakageRegression(
                        component=item.component,
                        current_rate=item.rate,
                        baseline_rate=baseline_rate,
                        delta=delta,
                    )
                )
        elif item.rate > spike_threshold:
            regressions.append(
                LeakageRegression(
                    component=item.component,
                    current_rate=item.rate,
                    baseline_rate=0.0,
                    delta=item.rate,
                )
            )

    report = LeakageReport(
        slice_id=slice.slice_id,
        total=total,
        flagged=total_flagged,
        aggregate_rate=aggregate_rate,
        by_component=tuple(sorted(by_component, key=lambda r: r.component.value)),
        regressions=tuple(regressions),
    )
    if regressions:
        _LOG.warning(
            "leakage_rate_regression",
            slice_id=slice.slice_id,
            aggregate_rate=report.aggregate_rate,
            regressions=[
                {
                    "component": r.component.value,
                    "current_rate": r.current_rate,
                    "baseline_rate": r.baseline_rate,
                    "delta": r.delta,
                }
                for r in regressions
            ],
        )
    else:
        _LOG.info(
            "leakage_rate_estimated",
            slice_id=slice.slice_id,
            total=report.total,
            flagged=report.flagged,
            aggregate_rate=report.aggregate_rate,
        )
    return report


__all__ = [
    "DEFAULT_SPIKE_THRESHOLD",
    "estimate_leakage_rate",
]
