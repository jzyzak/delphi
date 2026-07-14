# DELPHI

General-purpose superforecaster: an arbitrary question about the future goes in,
a **calibrated** probability (or predictive distribution) comes out, decomposed,
evidence-backed, and honest about its own uncertainty. The product is an API
endpoint, not a chatbot. See [CLAUDE.md](CLAUDE.md) for the full design, mission,
and prime directives (no look-ahead, proper scoring, sacred holdout).

This README is the operator runbook: how to bring DELPHI up with **real** LLMs
(the direct Anthropic Claude API), real evidence retrieval (Tavily), and
Postgres, and how to verify end to end that everything works.

> **LLM transport:** DELPHI calls the **Anthropic Claude API directly** (a Claude
> Console key) by default. AWS Bedrock is retained as an opt-in path
> (`DELPHI_LLM_PROVIDER=bedrock`); see [Deploying on AWS](#deploying-on-aws-outline).

---

## What you need

- **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/).
- **Docker** (for local Postgres) — or any reachable PostgreSQL with the
  `pgvector` extension.
- **An Anthropic Claude API key** from the Claude Console
  (<https://console.anthropic.com/>) — see step 2.
- **A Tavily API key** for evidence retrieval (<https://tavily.com>).

DELPHI needs three external dependencies to form a real forecast: **Postgres**
(the registry/PIT spine), the **Claude API** (the model pool), and **Tavily**
(as-of evidence search). The `delphi doctor` command checks all three.

---

## Quick start (local)

### 1. Install + start Postgres

```bash
uv sync --extra dev          # install DELPHI + dev tooling
docker compose up -d         # start local Postgres (pgvector) on :5432
```

Database migrations apply automatically the first time DELPHI connects (each
store runs its own SQL on `connect(..., migrate=True)`); there is no separate
migration step.

### 2. Get a Claude API key

Create a key in the **Claude Console** (<https://console.anthropic.com/> ->
Settings -> API keys). No AWS account is required for the default transport.

The default model IDs are pinned in [common/settings.py](common/settings.py) as
**Claude API** ids: Claude Opus 4.8 (`claude-opus-4-8`) for the workhorse tier
and Claude Fable 5 (`claude-fable-5`) for the strongest tier. Override per tier
with `DELPHI_MODEL_OPUS` / `DELPHI_MODEL_FABLE` if needed.

### 3. Configure environment

```bash
cp .env.example .env
# edit .env: set DELPHI_SECRET_ANTHROPIC_API_KEY and DELPHI_SECRET_TAVILY_API_KEY
set -a && source .env && set +a     # export into your shell
```

### 4. Verify every dependency

```bash
uv run delphi doctor
```

`doctor` checks Postgres (connect + migrate), the LLM (a cheap structured Claude
API call per tier), Tavily (a probe search), and the snapshot directory
(writable). It prints `[PASS]`/`[FAIL]` per dependency and exits non-zero if any
fail. Get this green before going further.

### 5. Run the live smoke suite (the proof)

```bash
DELPHI_LIVE_SMOKE=1 uv run pytest tests/live -m live -v
```

This exercises the real path end to end: a Claude API round-trip per tier, Tavily
retrieval with as-of filtering + snapshot replay, a full forecast written to
Postgres (asserting no look-ahead and a complete registry record), the
conductor, and the published API. It is skipped by default and never runs in CI.

### 6. Form your first forecast

```bash
NOW=$(python -c "import datetime; print(datetime.datetime.now(datetime.UTC).isoformat())")

uv run delphi intake   "Will SpaceX reach orbit with Starship before 2027?" --as-of "$NOW"
uv run delphi forecast "Will SpaceX reach orbit with Starship before 2027?" --as-of "$NOW"
uv run delphi conductor "Will SpaceX reach orbit with Starship before 2027?" --as-of "$NOW"
```

Serve the OpenAI-compatible API and call it:

```bash
uv run delphi serve --check          # health round-trip, no socket bound
uv run delphi serve --host 127.0.0.1 --port 8080   # bind and serve

curl -s http://127.0.0.1:8080/v1/forecast \
  -H 'Content-Type: application/json' \
  -d "{\"question\": \"Will it rain in London tomorrow?\", \"as_of\": \"$NOW\"}" | python -m json.tool
```

The response is OpenAI-shaped with the full DELPHI envelope under the `delphi`
key: calibrated probability, confidence band, rationale, evidence provenance
(with knowledge-time stamps), calibration metadata, resolution criteria, and a
reproducibility handle.

---

## CLI reference

| Command | Purpose |
|---|---|
| `delphi doctor` | Check Postgres, the LLM (Claude API), Tavily, and snapshot dir. |
| `delphi intake "<q>" [--as-of TS]` | Show the normalized, resolvable question (or the refusal). |
| `delphi forecast "<q>" --as-of TS` | Form a calibrated forecast; writes a full registry record. |
| `delphi conductor "<q>" --as-of TS` | Forecast via the heuristic conductor; records a workflow trace. |
| `delphi resolve [--since TS] [--answers FILE]` | Resolve closed questions from a JSON answer key. |
| `delphi eval --suite metaculus\|forecastbench [--leakage-audit]` | Retrospective proper scores + baselines + CIs, or a leakage audit. |
| `delphi bench live --harvest\|--score [--suite S] [--tick TS] [--since TS]` | Nightly live loop: harvest open questions or resolve + score matured ones. |
| `delphi serve [--host H --port P] [--check]` | Serve the published API. |

`--answers` takes a JSON file mapping question ids to ground truth, e.g.:

```json
{ "q-abc123": { "value": 1.0, "resolved_at": "2027-01-01T00:00:00Z", "source": "official result", "label": "YES" } }
```

---

## Benchmarking (Metaculus + ForecastBench)

Two evaluation paths, both honoring the prime directives (proper scores +
baselines + CIs, leakage-first, calibration on a disjoint split, the trials
ledger). The **live number is the only real one** (CLAUDE.md §2.7); retrospective
scores are development-only and suspect until leakage-audited.

**Retrospective** (`delphi eval`) fetches resolved questions, forecasts each at
its as-of pin, fits recalibration on a disjoint calibration split, scores the
held-out split, and reports Brier/log + per-domain + baseline deltas + a
reliability diagram — never a bare score:

```bash
delphi eval --suite metaculus                # proper scores + baselines + CIs
delphi eval --suite metaculus --leakage-audit # leakage rate over forecast traces
delphi eval --suite forecastbench            # needs DELPHI_FORECASTBENCH_QUESTION_SET
```

**Live** (`delphi bench live`) harvests genuinely-open questions, forecasts them
(pinned to the harvest instant), and scores them once they resolve. Schedule the
two phases nightly; `--tick` makes each run idempotent:

```bash
delphi bench live --harvest --suite metaculus --tick "$(date -u +%FT%TZ)"
delphi bench live --score   --suite metaculus --tick "$(date -u +%FT%TZ)"
```

Config: an optional `DELPHI_SECRET_METACULUS_API_TOKEN` (public reads work
without it) and, for ForecastBench, `DELPHI_FORECASTBENCH_QUESTION_SET` /
`DELPHI_FORECASTBENCH_RESOLUTION_SET` (see `.env.example`). Metaculus MiniBench
resolves fastest, so it is the recommended place to start the live loop.

Full design and internals: [`benchmarks/DOCUMENTATION.md`](benchmarks/DOCUMENTATION.md).

---

## Cost + latency notes

The model pool is **tiered by capability class** so the expensive model is used
sparingly (CLAUDE.md §7). By default:

| Tier (env override) | Model | Used by |
|---|---|---|
| Workhorse (`DELPHI_MODEL_OPUS`) | Claude Opus 4.8 | the high-volume estimator ensemble; intake, base-rate / decomposition / inside-view, leakage judge |
| Strongest (`DELPHI_MODEL_FABLE`) | Claude Fable 5 | the supervisor / aggregation / meta-layer |

A single `delphi forecast` fans out an ensemble of estimator calls plus several
reasoning and one supervisor call, so it is materially more expensive than one
chat completion. Note the high-volume tier defaults to Opus 4.8 (not a cheap
Haiku-class model), so watch cost closely; set `DELPHI_MODEL_OPUS` to a cheaper
model if you want to economize the ensemble. `delphi conductor` (and the
`delphi_deep` API tier) add red-team and verifier passes. Start with a couple of
questions and watch your Claude API usage before scaling up. Evidence retrieval is
cached: every search is written to the snapshot store (`DELPHI_SNAPSHOT_DIR`,
default `~/.delphi/snapshots`), so re-running the same `(query, as-of)` costs
nothing and stays leakage-auditable.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `doctor` postgres FAIL: `DELPHI_PG_DSN not set` | Export the DSN (see `.env.example`); `docker compose up -d`. |
| `doctor` postgres FAIL: connection refused | Container not up or wrong port; `docker compose ps`, check `:5432`. |
| `doctor` llm FAIL: secret `anthropic-api-key` not found | `DELPHI_SECRET_ANTHROPIC_API_KEY` not exported (Claude Console key). |
| `doctor` llm FAIL: `AuthenticationError` (401) | Bad/revoked Claude API key. |
| `doctor` llm FAIL: `NotFoundError` (404) on model | A `DELPHI_MODEL_*` id isn't a valid Claude API model; check ids in `common/settings.py`. |
| `doctor` tavily FAIL: secret not found | `DELPHI_SECRET_TAVILY_API_KEY` not exported. |
| `forecast` refuses with `already_resolved` | The question closes before your `--as-of`; pick a genuinely open question or an earlier as-of. |
| Forecast returns few/no evidence items | Tavily returned undated hits; the as-of filter drops undated results as unsafe. Tavily's `news` topic (the default) supplies dates. |
| `eval --suite forecastbench` KeyError on `DELPHI_FORECASTBENCH_QUESTION_SET` | Set the question-set (and optional resolution-set) path/URL; see `.env.example`. |
| `eval` returns "no accepted forecasts" | Intake refused every fetched question, or none had binary resolutions; widen the fetch or check the suite. |

---

## Deploying on AWS (outline)

The local setup maps cleanly onto AWS; the code already supports it (no forks):

- **Postgres -> Amazon RDS for PostgreSQL** with `pgvector` enabled. Point
  `DELPHI_PG_DSN` at the RDS endpoint; migrations still auto-apply on connect.
- **Secrets -> AWS Secrets Manager.** `common/secrets.py` already ships
  `AwsSecretsManagerProvider`; store the Claude + Tavily keys under the logical
  names `anthropic-api-key` / `tavily-api-key` and resolve via that provider
  instead of env.
- **LLM: direct Claude API (default) or Bedrock (opt-in).** The default transport
  calls `api.anthropic.com` and works from any AWS compute with just the API key —
  no Bedrock setup needed. To switch to **Bedrock in-VPC** (on your AWS credits),
  set `DELPHI_LLM_PROVIDER=bedrock` + `DELPHI_AWS_REGION`, override the model ids
  with Bedrock-style ids (e.g. `anthropic.claude-opus-4-8`, or a `us.`/`global.`
  inference profile), and grant the task role `bedrock:InvokeModel` (plus
  `aws-marketplace:Subscribe` for first-time model access).
- **Compute.** `delphi serve` is a plain WSGI app; run it on App Runner / ECS
  Fargate / EC2 behind an ALB. Ensemble sweeps and backfilled scoring are a good
  fit for **AWS Batch on Spot** (CLAUDE.md §7) — graduate to Step Functions or
  Dagster only when complexity actually demands it.
- **Evidence lake.** Point the snapshot store at S3/Parquet for a durable,
  reproducible, leakage-auditable evidence archive.

Not yet wired (follow-ups): scheduling the nightly live loop on managed compute
(e.g. EventBridge + Batch/Step Functions), persisting fitted calibration
artifacts to the S3 artifacts bucket for reproducible eval reruns. Terraform for
the core stack lives in [`deploy/aws/`](deploy/aws/). The retrospective eval and
live loop themselves are wired for Metaculus and ForecastBench (see
[Benchmarking](#benchmarking-metaculus--forecastbench)).

---

## Contributing & license

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup and the hard gates (tests in the same PR, hermetic suite,
never weaken the evaluation harness). Security reports: see
[SECURITY.md](SECURITY.md).

DELPHI is released under the [Apache License 2.0](LICENSE).
