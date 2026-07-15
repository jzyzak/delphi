# DELPHI improvement program — pre-declared measurement protocol (July 2026)

This document is written **before** any scored run of the improved
configuration, per the trials-ledger discipline (CLAUDE.md §2.4). Every run
listed here will be reported, win or lose. No runs beyond those declared here
will be scored against guarded sets in this program.

## Changes under test (commit `129dc4a`)

- **Phase A (reasoning activation):** adaptive thinking + effort=high on all
  reasoning/estimation calls; Bayesian ensemble (12 draws; prior =
  reference-class base rate, per-draw evidence log-LRs in log-odds space);
  supervisor trigger loosened (spread 0.08) with a MEDIUM apply gate;
  anti-hedging prompts; eval-side alpha grid 0.25–3.0 and isotonic min-sample
  guard (fit on the calibration split only, §2.5).
- **Phase B (historical evidence):** GDELT (as-of-bounded news archive) +
  Wikipedia (revision pinned at/before as-of) + Tavily via a composite
  searcher (`DELPHI_EVIDENCE_PROVIDERS=tavily,gdelt,wikipedia`).

## Declared runs

| # | Purpose | Question set | Config | Cap | Est. ledger draw |
|---|---|---|---|---|---|
| 1 | Dev ablation: Phase A only | 2026-03-01 (dev; already exposed) | providers=tavily (evidence-blind) | 120 | ~53 |
| 2 | Dev ablation: Phase A+B | 2026-03-01 (dev) | providers=tavily,gdelt,wikipedia | 120 | ~53 |
| 3 | **Confirmatory** (headline) | freshest paired FB set with ≥200 resolved (never evaluated) | Phase A+B | 250 | ~110 |

Baseline for comparison: the July 2026 run (Brier 0.2056, n=160, commit
`a010507` config) already published in [README.md](README.md).

- All runs use `DELPHI_EVAL_RESOLVED_AFTER=2026-02-01` and
  `--with-leakage-audit`.
- Dev-set caveat: the 2026-03-01 set has been evaluated before (baseline); its
  ablation numbers are development signals, not headline claims. The
  confirmatory set is selected only by recency/size criteria, before seeing
  any scores on it.
- Ledger before program: 163/1000 consumed. Projected after: ~380/1000.
- Cost guard: if the ablation-measured per-question cost projects the program
  above ~$500, the confirmatory cap shrinks first (declared here, applied
  mechanically).

## Results

*(to be filled after each run — no edits above this line after run 1 starts)*
