"""Dependency health checks for ``delphi doctor``.

A tiny, dependency-light harness: each check is a named zero-arg probe that
either returns a human-readable detail string or raises. ``run_checks`` turns
probes into :class:`CheckResult` rows (never raising), and ``format_report``
renders a PASS/FAIL report plus an overall ok flag. The probes themselves
(Postgres, Bedrock, Tavily, filesystem) live in the CLI so this module stays
hermetic and unit-testable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = ["CheckResult", "Probe", "format_report", "run_checks"]

Probe = Callable[[], str]


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one dependency check."""

    name: str
    ok: bool
    detail: str


def run_checks(checks: Sequence[tuple[str, Probe]]) -> list[CheckResult]:
    """Run each named probe, capturing success detail or the failure message."""
    results: list[CheckResult] = []
    for name, probe in checks:
        try:
            detail = probe()
        except Exception as exc:  # noqa: BLE001 - a doctor reports failures, never crashes
            results.append(CheckResult(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}"))
        else:
            results.append(CheckResult(name=name, ok=True, detail=detail))
    return results


def format_report(results: Sequence[CheckResult]) -> tuple[str, bool]:
    """Render results as a PASS/FAIL report; return ``(text, all_ok)``."""
    lines: list[str] = []
    all_ok = True
    for result in results:
        all_ok = all_ok and result.ok
        status = "PASS" if result.ok else "FAIL"
        lines.append(f"[{status}] {result.name}: {result.detail}")
    return "\n".join(lines), all_ok
