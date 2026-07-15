# DELPHI retrospective benchmark — ForecastBench, July 2026

Retrospective evaluation of DELPHI on the public **ForecastBench** dataset
(Karger et al., forecastingresearch/forecastbench-datasets). Question in,
calibrated probability out; scored with proper scores against the dataset's
crowd baseline, leakage-audited, under the trials-ledger discipline of
CLAUDE.md §2.

## Headline results (held-out test split, n = 160)

| Metric | Value | 95% CI (question-level bootstrap) |
|---|---|---|
| **Brier score** | **0.2056** | [0.1816, 0.2310] |
| Log score | 0.6787 | [0.5341, 0.8414] |
| **Δ Brier vs crowd freeze baseline** | **−0.1338** | (negative = DELPHI better) |
| ECE (10-bin) | **0.0484** | — |
| MCE | 0.2743 | — |
| Leakage rate (LLM judge over traces) | **0.0000** (0/160 flagged) | — |
| Flagged-at-chance robustness | 1.0000 | — |

Per-domain Brier (n): acled 0.107 (24) · metaculus 0.129 (14) · wikipedia 0.157
(22) · dbnomics 0.232 (26) · fred 0.236 (16) · infer 0.253 (8) · yfinance 0.252
(21) · polymarket 0.261 (20) · manifold 0.302 (9).

Full report: [`forecastbench-report.md`](forecastbench-report.md). Pilot run
(10 questions, pipeline validation): [`pilot-forecastbench.md`](pilot-forecastbench.md).

## Methodology

- **Dataset.** ForecastBench question set `2026-03-01-llm.json` (500 binary
  questions, frozen 2026-02-19T00:00:00Z) with its paired resolution set
  `2026-03-01_resolution_set.json` (363 questions resolved as of the July 2026
  snapshot). Sources span prediction markets (Metaculus, Manifold, Polymarket,
  INFER) and data series (ACLED, FRED, DBnomics, Wikipedia, yfinance).
- **As-of discipline.** Every forecast is pinned to the question's
  `freeze_datetime` (2026-02-19). Evidence retrieval flows through the as-of
  facade: any retrieved item without a knowledge-time at/before the pin is
  discarded. Search results are content-addressed snapshots
  (`~/.delphi/snapshots`), so reruns replay identical evidence.
- **Training-cutoff guard.** Models have a January 2026 training cutoff. Only
  questions **resolved after 2026-02-01** were eligible
  (`DELPHI_EVAL_RESOLVED_AFTER`), and the question freeze itself (2026-02-19)
  postdates the cutoff, so no outcome in the scored set existed during model
  training.
- **Models.** `claude-opus-4-8` (workhorse: intake, base rate, decomposition,
  4-agent inside-view ensemble, leakage judge) and `claude-fable-5`
  (supervisor reconciliation, triggered only on material ensemble
  disagreement). ~10–12 LLM calls per question; no conductor (plain
  `Forecaster` chain).
- **Calibration split (§2.5).** Of 363 eligible questions, 320 forecasts were
  accepted (41 refused at intake as unresolvable-as-posed; 2 skipped on
  per-question errors, see below). The 320 accepted forecasts were split 50/50
  (seed 0): an isotonic recalibrator and extremization coefficient were fit on
  the 160 calibration questions ONLY and applied to the 160 held-out test
  questions; only the test half is scored. Calibration questions are never
  scored.
- **Baseline.** The dataset's `freeze_datetime_value` per question — the crowd
  forecast (market questions) or naive current-value forecast (data-series
  questions) at freeze time — scored on the identical question set.
- **Multi-horizon resolutions.** ForecastBench resolves data-series questions
  at several horizons; DELPHI's adapter keeps the last-listed resolved entry
  per question (deterministic).
- **Trials ledger (§2.4).** Program budget pre-declared at 1000. Draws:
  pilot 3 + main run 160 = **163 consumed**; one aborted run (crashed
  pre-scoring on an unhandled API safety refusal, since fixed) drew 0. No other
  guarded evaluations were run.
- **Skips.** 2 of 363 questions were excluded by per-question error handling:
  one Anthropic bio-safety-classifier refusal, one persistently malformed
  intake output.
- **Runtime.** Main run: 2h43m wall (2026-07-15 04:30–07:13 UTC), sequential.
- **Code.** Public repo `jzyzak/delphi` @ `a010507`; runner:
  `DELPHI_EVAL_RESOLVED_AFTER=2026-02-01 DELPHI_EVAL_MAX_QUESTIONS=400
  delphi eval --suite forecastbench --with-leakage-audit`.

## The most important caveat: this run was evidence-blind

Registry inspection shows **0 retrieved evidence items on all 498 recorded
forecasts**. The live search provider (Tavily) indexes the present-day web, so
essentially every result is dated after the 2026-02-19 as-of pin (or undated)
— and the as-of filter correctly discards all of it. Retrospective runs with a
live search engine cannot time-travel.

Consequences, stated plainly:

- These numbers measure DELPHI's **reasoning, ensembling, and calibration
  machinery without retrieval** — the models' internal knowledge as of their
  January 2026 cutoff plus the pipeline's aggregation/calibration discipline.
- The 0.0000 leakage rate is **structurally guaranteed** here (no evidence in,
  no evidence-borne leakage), not merely observed. It validates the as-of
  facade's fail-closed behavior; it does not exercise the leakage judge against
  contaminated evidence.
- The reliability table shows heavy mass at [0.5, 0.6) (117/160): without
  evidence, the pipeline hedges — and the hedge is *well calibrated*
  (mean prediction 0.522 vs outcome rate 0.547; overall ECE 0.048).
- Retrieval-enabled performance is measured by the **live benchmark loop**
  (`delphi bench live`), where as-of = now and search engages fully. Per
  CLAUDE.md §2.7, that live number is the only one we ultimately stand behind;
  this retrospective is a development-grade result.

## Other caveats

- **Crowd-baseline strength varies by source.** For data-series questions the
  freeze-value "crowd" is a naive persistence forecast, weaker than a true
  crowd; the aggregate Δ of −0.134 should not be read as beating prediction
  markets at their own game (on market-sourced questions alone, DELPHI's
  per-domain Briers of 0.13–0.30 bracket the aggregate).
- **Single question set, single time window.** One freeze date (2026-02-19),
  resolutions through early July 2026.
- **Metaculus's own API suite is pending.** Metaculus withholds resolution
  values and community aggregates from standard API tokens (verified July
  2026); elevated research access has been applied for. Metaculus questions
  are nonetheless represented here through ForecastBench's independent
  resolution pipeline (n=14 scored).
- **The recalibrator used here is eval-fitted.** The production live path
  currently runs an identity recalibrator; retrospective eval fits
  isotonic+extremization per §2.5 on its own calibration split.

## Reproducibility

- Registry (Postgres): 500 `question`, 498 `evidence_set`, 498 `forecast`
  append-only records with hash-chained integrity, as-of stamps, model IDs,
  and full workflow provenance; `trials_ledger` documents every guarded draw.
- Evidence snapshots: content-addressed under `~/.delphi/snapshots`
  (sha256 of provider/version/query/as-of).
- Exact environment knobs, dataset refs, and commit hash listed above; the
  question and resolution sets are public and immutable in the
  forecastingresearch/forecastbench-datasets repository.
