# Summary

<!-- What this PR changes and why. -->

## Checklist (hard gates — see CONTRIBUTING.md)

- [ ] Tests for every behavior change land **in this PR**
- [ ] `uv run pytest` green locally (hermetic — no network, keys, or DB needed)
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run pyright` passes
- [ ] No test deleted, skipped, or weakened to go green
- [ ] `evaluation/`, the calibration split, and the holdout untouched (or made strictly stricter, with new tests)
- [ ] Anything reading the world has leakage/as-of tests; no `now()` in forecast-forming code
- [ ] CLAUDE.md updated if architecture or conventions changed
