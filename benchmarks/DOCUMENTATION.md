# Benchmarking — Full Documentation (Metaculus + ForecastBench)

This document describes every feature built to benchmark DELPHI against
**Metaculus** and **ForecastBench**, across both evaluation paths:

- **Retrospective** — `delphi eval --suite metaculus|forecastbench`: fetch resolved
  questions, forecast each *as of* its pin, calibrate on a disjoint split, score
  the held-out split, and render proper scores + baselines + CIs + a leakage audit.
- **Live** — `delphi bench live --harvest|--score`: harvest genuinely-open
  questions, forecast them pinned to the harvest instant, and score them once they
  resolve. This is the only number we publish (CLAUDE.md §2.7).

It also documents the cross-cutting changes made to support these paths: the
append-only **trials ledger** write, the **benchmark-id metadata threading**
through the forecast chain, and the **benchmark resolution source**.

For the pieces this builds on, see:
[base adapter contract](base.py),
[the eval harness](../evaluation/report.py),
[calibration split](../evaluation/calibration_split.py),
[the forecast chain](../forecaster/chain.py),
[the leakage judge](../core/forecast/LEAKAGE_JUDGE_DOCUMENTATION.md), and
[the registry](../core/registry/DOCUMENTATION.md).

---

## Table of contents

1. [Mission and invariants](#1-mission-and-invariants)
2. [Architecture and data flow](#2-architecture-and-data-flow)
3. [Module map (what changed / what's new)](#3-module-map)
4. [Trials ledger (append-only debit)](#4-trials-ledger)
5. [Network fetchers](#5-network-fetchers)
   - 5.1 [HTTP foundation](#51-http-foundation)
   - 5.2 [MetaculusFetcher](#52-metaculusfetcher)
   - 5.3 [ForecastBenchFetcher](#53-forecastbenchfetcher)
6. [Retrospective suite loader](#6-retrospective-suite-loader)
7. [The `delphi eval` command](#7-the-delphi-eval-command)
8. [Live loop](#8-live-loop)
   - 8.1 [Benchmark-id metadata threading](#81-benchmark-id-metadata-threading)
   - 8.2 [Harvest job](#82-harvest-job)
   - 8.3 [Benchmark resolution source](#83-benchmark-resolution-source)
   - 8.4 [The `delphi bench live` command](#84-the-delphi-bench-live-command)
9. [Configuration and secrets](#9-configuration-and-secrets)
10. [Operator guide (how to run)](#10-operator-guide)
11. [Testing](#11-testing)
12. [Design decisions and deviations from the plan](#12-design-decisions-and-deviations)
13. [Known limitations and future work](#13-known-limitations-and-future-work)

---

## 1. Mission and invariants

Benchmarking DELPHI is not "run it and check accuracy." A benchmark result is only
trustworthy if it satisfies the prime directives simultaneously. Every feature
here is built to preserve them:

| Prime directive | How this feature honors it |
|---|---|
| **§2.1 No look-ahead** | Every fetched question carries an explicit `as_of`; all forecast reads go through the as-of facade; `assert_no_leakage` rejects any resolution dated before its question's `as_of`. Fetchers never call `now()` — pins come from the question's open/freeze time or an explicit `freeze_at`. |
| **§2.3 Proper score, not accuracy** | The suite loader always renders Brier + log, per-domain, with question-level bootstrap CIs, against mandatory baselines — never a bare number (`evaluation/report.py`). |
| **§2.4 Trials ledger is law** | Each guarded-set evaluation debits the append-only `trials_ledger` (now actually written on commit); silent re-runs draw down a budget. |
| **§2.5 Calibration on disjoint data** | Recalibration/extremization are fit only on a calibration split disjoint from the scored split; calibration questions are never scored. |
| **§2.6 Leakage-first** | The retrospective path collects per-forecast traces and exposes `delphi eval --leakage-audit`. |
| **§2.7 The live number is the only real one** | The live loop forecasts genuinely-open questions and scores them only on resolution — impossible to tune. |

---

## 2. Architecture and data flow

```mermaid
flowchart TD
  subgraph fetch [Network - benchmarks/fetchers]
    MetAPI[MetaculusFetcher] --> Recs["dict records"]
    FbRepo[ForecastBenchFetcher] --> Recs
  end
  Recs --> Adapter["from_records() - existing adapters"]

  subgraph retro [Retrospective - delphi eval]
    Adapter --> Split["assign_calibration_split (disjoint)"]
    Split --> Fit["fit_calibration_artifact (calibration split)"]
    Adapter --> Fc["forecast_fn: raw probability + traces"]
    Fit --> Apply["recalibrate then extremize (test split)"]
    Fc --> Apply
    Apply --> Scored["scored_records(forecasts, adapter)"]
    Scored --> Ctx["EvalContext"]
    Fc --> Traces["Traces"]
    Ctx --> Report["render_report + render_leakage_audit"]
  end

  subgraph live [Live - delphi bench live]
    Adapter -. "open questions" .-> Harvest["HarvestJob (conductor)"]
    Harvest --> Reg["RegistryStore (+benchmark_question_id)"]
    Reg --> Score["ScoreJob"]
    BenchRes["BenchmarkResolutionSource"] --> Score
    Score --> Metrics["LiveMetrics (Brier / log)"]
  end
```

**Two invariants shape the topology.** First, *fetch is separated from mapping*:
fetchers do all the network I/O and emit plain `dict` records; the existing
adapters (`MetaculusAdapter`, `ForecastBenchAdapter`) map those to pinned
`BenchmarkQuestion` / `BenchmarkResolution` objects deterministically. That keeps
adapters pure and lets the whole fetch path be tested with a mocked transport.
Second, *the harness is the house*: the suite loader assembles an `EvalContext` but
never scores; scoring happens inside `EvalHarness`, which debits the ledger.

---

## 3. Module map

New files:

| Path | Purpose |
|---|---|
| [`core/orchestration/migrations/0002_trials_ledger.sql`](../core/orchestration/migrations/0002_trials_ledger.sql) | Append-only `trials_ledger` table (§4). |
| [`benchmarks/fetchers/__init__.py`](fetchers/__init__.py) | Fetcher package exports. |
| [`benchmarks/fetchers/metaculus_api.py`](fetchers/metaculus_api.py) | `MetaculusFetcher` + `map_post`. |
| [`benchmarks/fetchers/forecastbench_repo.py`](fetchers/forecastbench_repo.py) | `ForecastBenchFetcher` + `map_question` / `map_resolutions`. |
| [`benchmarks/suites.py`](suites.py) | `build_eval_context`, `forecaster_fn`, `records_baseline`. |
| [`resolution/benchmark_source.py`](../resolution/benchmark_source.py) | `BenchmarkResolutionSource` + `BENCHMARK_QUESTION_ID_KEY`. |

Modified files:

| Path | Change |
|---|---|
| [`core/orchestration/budget.py`](../core/orchestration/budget.py) | `PostgresBudgetLedger.commit` now appends `grant.n` rows to `trials_ledger` (idempotent). |
| [`benchmarks/live_loop/harvest.py`](live_loop/harvest.py) | Threads the benchmark id into the recorded question's metadata. |
| [`intake/service.py`](../intake/service.py) | `intake(..., metadata=...)` merges caller metadata into the question. |
| [`forecaster/chain.py`](../forecaster/chain.py) | `forecast(..., metadata=...)` passes metadata to intake. |
| [`conductor/heuristic.py`](../conductor/heuristic.py) | `conduct(..., metadata=...)` passes metadata to the chain. |
| [`common/cli.py`](../common/cli.py) | `_default_eval_context` / `_default_live_context`; `eval` + `bench live` wired; `--suite` added to `bench live`. |

---

## 4. Trials ledger

**Problem it fixes.** `PostgresBudgetLedger` computes the firm-wide debited count as
`SELECT COUNT(*) FROM trials_ledger`, but that table was never created and never
written — so admission control had no persistent notion of how many trials had
been spent. Migration `0002_trials_ledger.sql` creates it:

```sql
CREATE TABLE IF NOT EXISTS trials_ledger (
    trial_id    BIGSERIAL   PRIMARY KEY,
    grant_id    TEXT        NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**How the debit works.** A guarded evaluation reserves `n = len(records)` slots
(`reserve_budget`), runs, then commits. `commit` now transitions the reservation
`reserved -> committed` and, only if that transition actually happened, appends
`grant.n` rows to `trials_ledger`:

```python
cur.execute("UPDATE budget_reservations SET status='committed' ... AND status='reserved'")
if cur.rowcount != 1:
    return  # already committed/released -> never double-debit
cur.execute("INSERT INTO trials_ledger (grant_id, recorded_at) "
            "SELECT %(grant_id)s, %(now)s FROM generate_series(1, %(n)s)")
```

**Invariants.**
- *Append-only*: the ledger is never updated or deleted, only inserted into — so it
  only ever grows. This is what makes method-overfitting (silent re-runs) *visible*
  rather than free (§2.4).
- *Idempotent*: the `rowcount != 1` guard means a repeated `commit` for the same
  grant appends nothing. `release` never writes to the ledger.
- *Unit = one question-evaluation*: reserving `len(records)` and committing `n` rows
  keeps `COUNT(*)` equal to the number of question-evaluations charged against the
  firm-wide cap (`settings.global_trials_budget`, default 100).

**In-memory path.** `InMemoryBudgetLedger` (tests, local dev) delegates the debited
count to an injected `trials_count` callable and does not itself persist trials —
its ledger is external by design. Only the Postgres ledger is the durable §2.4
record.

---

## 5. Network fetchers

### 5.1 HTTP foundation

Both fetchers take an injected [`common.http.client.HttpClient`](../common/http/client.py)
(httpx + tenacity: bounded retries on 429/5xx/transport errors, a polite rate gate,
and a merged `User-Agent`). This is the same client the Tavily provider uses, so
benchmarking inherits the same politeness and reliability. Because the client is
injected, tests supply an `httpx.Client(transport=httpx.MockTransport(handler))` and
never touch the network (CLAUDE.md §2.8).

The design rule is **fetch (network) is separate from map (pure)**: each fetcher
exposes standalone mapping functions (`map_post`, `map_question`, `map_resolutions`)
that turn raw provider JSON into adapter records with no I/O, so the mapping is unit
tested directly and the network path is tested with a mock transport.

### 5.2 MetaculusFetcher

File: [`benchmarks/fetchers/metaculus_api.py`](fetchers/metaculus_api.py).

```python
MetaculusFetcher(
    *, http: HttpClient,
    base_url: str = "https://www.metaculus.com/api",
    secrets: SecretProvider | None = None,
    api_token_secret: str = "metaculus-api-token",
)
.fetch(*, params=None, max_pages=1, binary_only=True, freeze_at=None) -> list[dict]
```

**Paging.** `fetch` GETs `{base_url}/posts/` with `params`, then follows the API's
`next` link up to `max_pages`. The `next` URL already carries the query string, so
`params` are only applied to the first request.

**Auth.** If a `SecretProvider` is given and the token secret resolves, requests
carry `Authorization: Token <token>`. Public question reads work without a token, so
a missing secret is tolerated (`SecretNotFoundError` -> no header), not fatal.

**Mapping (`map_post`).** Each Metaculus post is mapped to a `MetaculusAdapter`
record. Fields are read defensively from either the post or its nested `question`:

| Record key | Source | Notes |
|---|---|---|
| `id` | `post.id` / `question.id` | question_id becomes `metaculus:<id>`. |
| `title` | `post.title` | The question text. |
| `as_of` | `freeze_at` override, else `question.open_time` / `post.open_time` / `published_at` | If none derivable, the post is **skipped** (can't pin safely). |
| `question_type` | `question.type` | Default `binary`. |
| `domain` | first `projects.category[].slug` | Default `general`. |
| `resolution_criteria` | `question.description` | |
| `close_time` | `question.scheduled_close_time` | Optional. |
| `community` | `question.aggregations.recency_weighted.latest.centers[0]` | The crowd probability; retained for the consensus baseline. |
| `resolution` / `resolved_at` | `question.resolution` (`yes`->1.0, `no`->0.0) + `actual_resolve_time` | Only emitted for resolved binary questions; `annulled`/`ambiguous` are dropped. |

**Filtering.** `binary_only=True` (default) skips non-binary questions, because the
scored path only handles binary (0/1) outcomes. Set `binary_only=False` to keep
them (e.g. for inspection).

**As-of discipline.** For retrospective evaluation you typically fetch resolved
questions and let `as_of` come from each question's `open_time` (a moment the answer
was unknown). For the live harvest you pass `freeze_at=harvest_time` so every open
question is pinned to the harvest instant.

### 5.3 ForecastBenchFetcher

File: [`benchmarks/fetchers/forecastbench_repo.py`](fetchers/forecastbench_repo.py).

```python
ForecastBenchFetcher(
    *, http: HttpClient,
    base_url: str = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main",
)
.fetch(*, question_set: str, resolution_set: str | None = None, freeze_at=None) -> list[dict]
```

ForecastBench publishes a public dataset repo (no auth). `question_set` and
`resolution_set` may be repo-relative paths (joined onto `base_url`) or absolute
URLs. `fetch` reads the question document, optionally reads the resolution document,
joins them, and emits `ForecastBenchAdapter` records.

**Composite ids.** ForecastBench aggregates many upstream sources (Metaculus,
Manifold, Polymarket, ACLED, FRED, Wikipedia, ...), so a raw id is only unique
*within* its source. `_resolution_key(source, id)` forms `"<source>-<id>"` (e.g.
`metaculus-abc`), used as the record `id` and as the join key between questions and
resolutions. The adapter then makes the question_id `forecastbench:<source>-<id>`.

**Mapping (`map_question`).**

| Record key | Source | Notes |
|---|---|---|
| `id` | `<source>-<raw_id>` | Composite. |
| `question` | `question` / `title` | Text. |
| `as_of` | `freeze_at`, else `freeze_datetime`, else set-level `forecast_due_date` | Sentinels cleaned (see below). Skipped if none. |
| `domain` | `category` / `source` | Enables per-source/topic breakdowns. |
| `resolution_criteria` | `resolution_criteria` | |
| `close_time` | `resolution_date` / `market_info_close_datetime` | Optional; sentinels cleaned. |
| `freeze_value` | `freeze_datetime_value` | The crowd/market value frozen at `as_of` — carried for the baseline (see §6). |
| `resolved_value` / `resolved_at` / `resolution_source` | joined from the resolution doc | Only when a matching resolution exists. |

**`N/A` sentinel handling.** ForecastBench market questions carry the string
`"N/A"` for absent close/resolution dates. Passing that to the adapter's ISO parser
raises. `_clean_dt` maps `{"", "n/a", "na", "none", "null", "-"}` (case-insensitive)
to `None`, applied **per-candidate before `_first`** so that, e.g., a `"N/A"`
`freeze_datetime` correctly falls back to the set-level `forecast_due_date` rather
than being treated as a real value. This was validated against the live repo
(998 questions across 8 sources parse cleanly and pass `assert_no_leakage`).

**Resolutions (`map_resolutions`).** Indexes the resolution document by composite
id, reading `resolved_to` / `resolved_value` / `value` for the outcome and
`resolution_date` / `resolved_at` for the timestamp. Entries explicitly marked
`resolved: false`, or missing a value/date, are skipped so only genuinely-resolved
questions produce a resolution row.

---

## 6. Retrospective suite loader

File: [`benchmarks/suites.py`](suites.py). This is the generic, hermetic engine that
turns an adapter + a forecasting function into an `EvalContext`. Suite-specific
fetching and baseline construction live in the CLI (§7); the engine has no network
or LLM dependency, so it is fully unit-tested.

### The forecasting function seam

```python
@dataclass(frozen=True)
class QuestionForecast:
    accepted: bool
    raw_probability: float | None
    traces: tuple[Trace, ...] = ()

ForecastFn = Callable[[str, datetime], QuestionForecast]
```

`forecaster_fn(forecaster)` adapts a real [`Forecaster`](../forecaster/chain.py) into
a `ForecastFn`. Crucially it returns the **raw reconciled** probability
(`result.calibrated.raw_probability`) — *before* recalibration/extremization — so the
suite loader can fit and apply calibration itself on disjoint splits. The
forecaster's own recalibrator is intentionally not double-applied. It also builds
leakage-audit `Trace`s from the forecast's evidence (one `SEARCH` trace per evidence
snippet, pinned at the evidence's knowledge-time) and its rationale (one `SUPERVISOR`
trace). Refused questions return `accepted=False` and are excluded.

### `build_eval_context`

```python
build_eval_context(
    adapter, forecast_fn, *, harness, judge=None,
    calibration_fraction=0.5, holdout_ids=(), seed=0, extra_baselines=(),
) -> EvalContext
```

Steps:

1. **Select resolved binary questions.** Only questions with a `0.0`/`1.0`
   resolution that are present in the adapter's question set.
2. **Forecast each once.** Call `forecast_fn(text, as_of)`; keep the raw probability
   and traces for accepted questions. Raise if none are accepted.
3. **Split (disjoint).** `assign_calibration_split(accepted_ids, holdout_ids=...,
   fraction=..., seed=...)` returns the calibration subset; the remainder is the test
   (scored) subset. Raise if the split consumes every question (lower the fraction).
4. **Fit calibration on the calibration split only** (§2.5). `fit_calibration_artifact`
   fits an isotonic recalibrator + an extremization `alpha` from the calibration
   questions' `(raw_probability, outcome)`. If the split is empty, fall back to
   identity recalibration + `DEFAULT_ALPHA`.
5. **Apply to the test split.** For each test question:
   `calibrated = calibrate(recalibrator.apply(raw), alpha=alpha)`. This mirrors the
   chain's *recalibrate-then-extremize* order exactly, done in-memory so the
   forecaster doesn't have to be re-run.
6. **Score.** `scored_records(forecasts, adapter)` joins the calibrated test
   probabilities to resolutions (running `assert_no_leakage` first). Calibration
   questions are absent from `forecasts`, so they are never scored.
7. **Assemble.** Return `EvalContext(EvalInputs(records, baselines, traces), harness,
   judge)` — traces are those of the test set only.

The harness is *not* run here; `render_report` runs it (and debits the ledger).

### Baselines

Two helpers feed `extra_baselines`:

- **Crowd/market consensus.** For Metaculus, `consensus_baseline(adapter,
  price_key="community_prediction")` (from `benchmarks/market_consensus.py`) reads the
  community probability retained in each question's metadata.
- **`records_baseline(records, *, source, value_key="freeze_value")`.** For
  ForecastBench, the crowd value is dropped by the adapter, so this builds the
  baseline straight from the fetched records, keyed to `question_id` (`source:id`).

Report the *delta* vs each baseline, and specifically whether **DELPHI + consensus
beats consensus** (§2.10) — additive information is the honest, publishable claim.

---

## 7. The `delphi eval` command

File: [`common/cli.py`](../common/cli.py), `_default_eval_context(suite)`.

```
delphi eval --suite metaculus|forecastbench [--leakage-audit]
```

`_default_eval_context` is the production wiring (network + LLM + DB, so it is
`# pragma: no cover` and exercised via injected contexts in tests):

1. Build an `HttpClient` and `forecast_fn = forecaster_fn(_default_forecaster())`.
2. Dispatch on `--suite`:
   - `metaculus`: `MetaculusFetcher.fetch(params={"statuses": "resolved",
     "forecast_type": "binary", "limit": 100}, max_pages=5)` -> `MetaculusAdapter`;
     baseline = crowd consensus.
   - `forecastbench`: `ForecastBenchFetcher.fetch(question_set=..., resolution_set=...)`
     from `DELPHI_FORECASTBENCH_QUESTION_SET` / `..._RESOLUTION_SET` ->
     `ForecastBenchAdapter`; baseline = `records_baseline`.
   - anything else raises with the valid suite names.
3. Build a `LeakageJudge` (for `--leakage-audit`).
4. Pick the ledger: `PostgresBudgetLedger` when `DELPHI_PG_DSN` is set, else
   `InMemoryBudgetLedger(cap=settings.global_trials_budget, ...)`.
5. Return `build_eval_context(...)`.

`cmd_eval` then either renders the full report (`render_report`) or, with
`--leakage-audit`, the leakage audit over the collected traces (`render_leakage_audit`).
Per the Paleka et al. (2026) pitfalls, a retrospective score is not trustworthy until
its leakage rate is reported.

---

## 8. Live loop

The live loop already existed (`benchmarks/live_loop/`); two gaps were closed so it
can actually resolve real questions: the benchmark id was being discarded at harvest,
and the default resolution source resolved nothing.

### 8.1 Benchmark-id metadata threading

To resolve a harvested question later, the registry question must remember which
benchmark question it came from. The registry is immutable and intake normalizes
text, so matching by text is fragile. Instead an optional `metadata` mapping is
threaded through the forecast chain and merged into the recorded question's metadata:

```
HeuristicConductor.conduct(q, *, as_of, metadata=None)
  -> Forecaster.forecast(q, *, as_of, metadata=None)
    -> IntakeService.intake(text, *, as_of=None, metadata=None)
      -> QuestionInput.metadata  (merged with intake's own entities/sources)
```

All three parameters are **optional with a `None` default**, so every existing call
site is unaffected (backward-compatible; no test churn beyond one stub signature).

### 8.2 Harvest job

File: [`benchmarks/live_loop/harvest.py`](live_loop/harvest.py). `HarvestJob.run` now
threads three metadata keys per question:

```python
metadata = {
    BENCHMARK_QUESTION_ID_KEY: question.question_id,  # e.g. "metaculus:101"
    "benchmark_source": question.source,
    "benchmark_external_id": question.external_id,
}
self._conductor.conduct(question.text, as_of=question.as_of, metadata=metadata)
```

Everything else is unchanged: accepted forecasts are counted as `pending`, refusals
as `refused`, and each accepted question lands a complete registry record pinned to
the harvest time.

### 8.3 Benchmark resolution source

File: [`resolution/benchmark_source.py`](../resolution/benchmark_source.py).

```python
BenchmarkResolutionSource(resolutions: Sequence[BenchmarkResolution],
                          *, metadata_key="benchmark_question_id")
.resolve(question: Question) -> ResolvedOutcome | None
```

Implements the `ResolutionSource` protocol. It reads the benchmark id from the
question's metadata and looks up a matching `BenchmarkResolution`. A question is
resolvable iff it carries the id *and* a matching resolution exists; otherwise
`resolve` returns `None` and `ResolutionService` leaves it open (idempotent — safe to
re-run nightly). `resolved_at` comes from the benchmark, never `now()` (§2.1), and
the provenance label falls back to the question's intake-captured
`resolution_sources` when the benchmark doesn't supply one.

### 8.4 The `delphi bench live` command

File: [`common/cli.py`](../common/cli.py), `_default_live_context(suite)`. A `--suite`
argument was added to `bench live` (default `metaculus`).

```
delphi bench live --harvest --suite metaculus --tick <ISO-8601>
delphi bench live --score   --suite metaculus --tick <ISO-8601>
```

`_default_live_context`:

1. `harvest_time = now(UTC)`; read the set of already-harvested benchmark ids from
   the registry (dedup).
2. Fetch **open** questions with `freeze_at=harvest_time` (so `as_of` = the harvest
   instant), filter out ids already seen, and build the **native** adapter
   (`MetaculusAdapter` / `ForecastBenchAdapter`).
3. Fetch **resolved** questions and build a `BenchmarkResolutionSource` from their
   resolutions.
4. Wire `HarvestJob(conductor=_default_conductor())`,
   `ScoreJob(store, ResolutionService(store, BenchmarkResolutionSource(...)))`, and a
   `RunStateStore` (`PostgresRunStateStore` when `DELPHI_PG_DSN` is set, else in-memory).

`cmd_bench_live` wraps each run in `claim_and_run(step_id=f"live-{mode}:{tick}")`, so a
given `--tick` runs at most once (idempotent) — safe for cron/at-least-once schedulers.

**Why the native adapter, not `LiveHarvestAdapter`?** `LiveHarvestAdapter` prefixes
ids with `live:`, which would break the join to resolutions keyed on the native id
(`metaculus:<id>`). Using the native adapter with a `freeze_at` pin keeps the id
scheme consistent end-to-end (harvest metadata id == resolution id), which is what
makes `BenchmarkResolutionSource` resolve correctly.

---

## 9. Configuration and secrets

All optional; see [`.env.example`](../.env.example).

| Env var | Purpose |
|---|---|
| `DELPHI_SECRET_METACULUS_API_TOKEN` | Metaculus API token (logical secret `metaculus-api-token`). Public reads work without it. |
| `DELPHI_FORECASTBENCH_QUESTION_SET` | Repo-relative path or URL of the ForecastBench question set (required for `--suite forecastbench`). |
| `DELPHI_FORECASTBENCH_RESOLUTION_SET` | Optional resolution set (open questions have none yet). |
| `DELPHI_PG_DSN` | When set, the trials ledger and run-state persist to Postgres; otherwise in-memory. |
| `DELPHI_GLOBAL_TRIALS_BUDGET` | Firm-wide trials cap (default 100). |

Secrets resolve through `common/secrets.py` (env or AWS Secrets Manager); model/DB
settings through `common/settings.py`. No credentials are hardcoded (§7).

---

## 10. Operator guide

```bash
# Load credentials from .env into the environment
set -a && source .env && set +a

# Retrospective (development). Always leakage-audit before trusting a score.
delphi eval --suite metaculus
delphi eval --suite metaculus --leakage-audit
delphi eval --suite forecastbench          # set DELPHI_FORECASTBENCH_QUESTION_SET first

# Live (the publishable track record). Schedule nightly; --tick makes it idempotent.
delphi bench live --harvest --suite metaculus --tick "$(date -u +%FT%TZ)"
delphi bench live --score   --suite metaculus --tick "$(date -u +%FT%TZ)"
```

A single `delphi eval` run forecasts every resolved question through the full
ensemble pipeline, so it makes many LLM calls — start with a small `limit` and watch
cost. Metaculus MiniBench resolves fastest, so it is the recommended place to start
the live loop.

---

## 11. Testing

Everything ships with tests in the same change (§2.8); all network/LLM calls are
mocked and time is injected (hermetic, deterministic).

| Test file | Covers |
|---|---|
| [`tests/orchestration/test_budget.py`](../tests/orchestration/test_budget.py) | In-memory ledger cycle + fail-closed; Postgres `trials_ledger` append, idempotent commit, release-writes-nothing, debit-against-cap (`@pytest.mark.postgres`). |
| [`tests/benchmarks/fetchers/test_metaculus_api.py`](../tests/benchmarks/fetchers/test_metaculus_api.py) | `map_post` field mapping, yes/no/annulled resolution, binary filter, `freeze_at`, missing as-of; `fetch` pagination, auth header, missing-token tolerance. |
| [`tests/benchmarks/fetchers/test_forecastbench_repo.py`](../tests/benchmarks/fetchers/test_forecastbench_repo.py) | `map_resolutions`, `map_question`, composite ids, `freeze_value`, `N/A` sentinel handling, question/resolution join. |
| [`tests/benchmarks/test_suites.py`](../tests/benchmarks/test_suites.py) | Disjoint calibration/test split, calibration questions never scored, identity fallback, refusal exclusion, `records_baseline`, `forecaster_fn` trace building. |
| [`tests/benchmarks/test_live_loop.py`](../tests/benchmarks/test_live_loop.py) | Harvest threads the benchmark id into metadata (plus the pre-existing loop tests). |
| [`tests/resolution/test_benchmark_source.py`](../tests/resolution/test_benchmark_source.py) | Resolve via metadata id, unmatched/missing/non-string id, provenance fallback. |
| [`tests/benchmarks/test_e2e_benchmark.py`](../tests/benchmarks/test_e2e_benchmark.py) | **End-to-end**: fetch -> adapter -> calibrate -> score -> leakage-audit; harvest -> registry -> resolve -> score. |

Run the full gate (with a local Postgres for the marked tests):

```bash
DELPHI_PG_DSN="postgresql://postgres:delphi@localhost:5433/delphi" uv run pytest
uv run ruff check . && uv run pyright
```

At the time of writing: ruff clean, pyright clean, 892 passed / 5 skipped (the
skipped are the `live` smoke tests gated by `DELPHI_LIVE_SMOKE`), coverage ~93%.

---

## 12. Design decisions and deviations

- **Native adapter for the live harvest, not `LiveHarvestAdapter`** (see §8.4). The
  plan named `LiveHarvestAdapter`, but its `live:` id prefix breaks the resolution
  join; the native adapter with a `freeze_at` pin keeps ids consistent, which is the
  correctness-critical property.
- **`--suite` added to `bench live`.** The plan's `_default_live_context(suite)`
  needed a source of the suite name; a `--suite` flag (default `metaculus`) provides
  it.
- **Calibration applied in-memory instead of re-running the forecaster.** The plan
  described injecting the fitted recalibrator into a fresh `Forecaster` and
  re-forecasting the test split. Instead the loader forecasts once (raw), fits on the
  calibration split, and applies *recalibrate-then-extremize* to the test split's raw
  probabilities in memory. This is equivalent, far cheaper, and fully deterministic —
  and it still guarantees the recalibrator is fit only on data disjoint from the
  scored set (§2.5).
- **`records_baseline` helper** instead of routing the ForecastBench crowd value
  through the adapter. The adapter drops unknown keys, so the crowd value is read
  from the fetched records directly and passed as an `extra_baseline`.

---

## 13. Known limitations and future work

- **No credentialed end-to-end run bundled.** The full `delphi eval` / `bench live`
  run needs Anthropic + Tavily keys and Metaculus API access; it was validated
  hermetically and against the live ForecastBench repo (read-only), but the
  credentialed forecasting run is an operator step (§10).
- **Extremization `alpha` re-fitting is skipped on the applied path.** The suite
  loader applies the fitted isotonic map + `alpha` to raw probabilities; it does not
  re-run the forecaster to also re-derive per-run extremization. This is a
  deliberate cost/fidelity trade-off, not a leakage concern.
- **Calibration artifacts are not yet persisted.** Fitted artifacts live only for the
  duration of an eval run; persisting them (e.g. to the S3 artifacts bucket) would
  make eval reruns reproducible and let the chain consume a stable recalibrator.
- **Live-loop scheduling is manual.** The idempotent tick is in place; wiring it to a
  managed scheduler (EventBridge + Batch/Step Functions) is future work.
- **ForecastBench dataset paths are date-stamped** and supplied via env; a small
  "latest question set" discovery helper would remove that manual step.
- **Metaculus community prediction is read from `recency_weighted.latest.centers[0]`**;
  richer aggregation snapshots (e.g. as-of-freeze community values) would sharpen the
  crowd baseline.
