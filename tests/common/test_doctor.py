"""Unit tests for the delphi doctor health-check harness."""

from __future__ import annotations

from common.doctor import CheckResult, format_report, run_checks


def test_run_checks_captures_success_and_failure() -> None:
    def ok() -> str:
        return "all good"

    def boom() -> str:
        raise RuntimeError("no connection")

    results = run_checks([("alpha", ok), ("beta", boom)])
    assert results == [
        CheckResult(name="alpha", ok=True, detail="all good"),
        CheckResult(name="beta", ok=False, detail="RuntimeError: no connection"),
    ]


def test_format_report_all_pass() -> None:
    text, all_ok = format_report(
        [CheckResult("alpha", True, "up"), CheckResult("beta", True, "up")]
    )
    assert all_ok
    assert text == "[PASS] alpha: up\n[PASS] beta: up"


def test_format_report_any_fail_flips_overall() -> None:
    text, all_ok = format_report(
        [CheckResult("alpha", True, "up"), CheckResult("beta", False, "down")]
    )
    assert not all_ok
    assert "[FAIL] beta: down" in text


def test_format_report_empty() -> None:
    text, all_ok = format_report([])
    assert all_ok
    assert text == ""
