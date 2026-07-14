# Contributing to DELPHI

Thanks for your interest in contributing. DELPHI is a calibrated forecasting system, and
its correctness guarantees are unusual: most of the rules below exist to protect
**calibration, leakage discipline, and the evaluation harness**. Read
[CLAUDE.md](CLAUDE.md) (especially §2, the prime directives) before writing code — it is
the project primer and the ground truth for conventions.

## Development setup

DELPHI uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
git clone <repo-url>
cd delphi
uv sync --extra dev
uv run pre-commit install
```

Copy `.env.example` to `.env` for local configuration. Unit tests require no network,
no API keys, and no database (Postgres-marked integration tests are opt-in via
`-m postgres`).

## The hard gates

Every PR must pass, with no exceptions:

```bash
uv run pytest          # full suite, green
uv run ruff check .    # lint
uv run ruff format --check .
uv run pyright         # types
```

- **Tests land in the same PR as the code.** A change that adds or modifies behavior
  without adding or updating tests is incomplete and will not merge.
- **Never delete, skip, `xfail`, or weaken a test to make CI green.** Fix the code or
  surface the failure. Weakening the evaluation harness (`evaluation/`) to improve a
  number is the single worst offense in this codebase (CLAUDE.md §2.2).
- **Tests are hermetic and deterministic.** Seed all randomness, freeze the as-of clock,
  and mock every network/LLM/provider call. A flaky test is a failing test.
- **Coverage floor (CI-enforced):** ≥90% line-and-branch repo-wide; 100% on `evaluation/`.
  `core/pit/`, `core/forecast/`, and `core/registry/` are ratcheting toward 100% — do not
  let them regress.

## Non-negotiable design rules

- **No look-ahead.** Forecast-forming code reads the world only through the as-of facade
  (`core/pit/`). There is no `now()` in forecast code; the as-of time is always an
  explicit input. Anything that reads the world needs leakage tests and as-of tests.
- **The harness is the house.** `evaluation/`, the calibration split, and the guarded
  holdout may only ever be changed to become *stricter*, with new tests.
- **Calibration is learned on disjoint data only** — never on the holdout or live set.
- **`core/` stays domain-agnostic.** Domain assumptions live in `forecaster/`, `sources/`,
  `intake/`, `conductor/`. Don't fork `core/`; put domain coupling behind an interface.
- **No hardcoded secrets or model IDs.** Model IDs are pinned in `common/settings.py`.

## Pull requests

1. Fork and create a feature branch.
2. Keep changes small and focused; prefer the smallest change that works.
3. Update documentation (including CLAUDE.md, if you change architecture or conventions)
   in the same PR.
4. Make sure the full gate above is green locally before opening the PR.

If a change is ambiguous about look-ahead, the holdout, calibration, or test coverage,
open an issue to discuss it first.
