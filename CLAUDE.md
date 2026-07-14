# CLAUDE.md — Codename: DELPHI (rename freely)

> A general-purpose **superforecaster**: arbitrary question in, *calibrated* forecast out,
> with rationale and evidence provenance. This file is the primer loaded into every Claude
> Code session. Read it fully before doing anything. It is intentionally dense — keep it
> that way. If you change architecture or conventions, update this file in the same PR.
>
> The domain-agnostic `core/` is designed to be reusable beyond DELPHI and carries the
> machinery the system's correctness depends on. Do not fork it. See §5.

---

## 1. What this is (mission in one breath)

We are building a system that takes an arbitrary question about the future (or any
unresolved fact) and returns a **calibrated probability or predictive distribution**,
decomposed, evidence-backed, and honest about its own uncertainty. The published product
is an **API endpoint**, not a chatbot: question in, forecast out.

The thesis (internalize this — it shapes every decision):

- **The product is calibration, not cleverness.** A forecaster that says 70% must be right
  70% of the time. A confident wrong answer is worse than a hedged right one. Every metric
  is a *proper score*; accuracy is not a metric here.
- **The LLM is not the forecaster.** It is a *researcher, estimator, and reasoner* inside a
  pipeline whose correctness is enforced by deterministic machinery the models cannot game:
  the as-of facade, the aggregation/calibration math, the registry, and the eval harness.
- **The value is not a model.** The models in the pool are commodities, and we don't even
  expose which one answered. What makes the system worth trusting is: ruthless
  as-of/leakage discipline, a registry-backed *live track record* nobody can retroactively
  fake, per-domain calibration we can prove, and — eventually — a conductor trained on the
  corpus of scored forecasts. That last asset compounds with every resolved question.
- **Search quality dominates.** In LLM forecasting, the quality of the as-of evidence
  retrieved drives accuracy more than the choice of model. Invest in retrieval accordingly.

The two things that will silently destroy this product are **leakage** and
**method-overfitting**. They are the whole reason §2 exists.

---

## 2. Prime directives (NON-NEGOTIABLE — violating these is a critical bug)

1. **NO LOOK-AHEAD. EVER.** Every forecast is formed *as of an explicit timestamp*.
   Forecast-forming code reads the world **only** through the as-of facade (`core/pit/`),
   which refuses anything dated later than the as-of time. There is **no `now()` inside
   forecast code** — the as-of time is always an explicit input. A single leaked post-as-of
   fact turns every retrospective score into fiction. If you are ever unsure whether a code
   path can see the future, stop and ask.

2. **THE HOLDOUT IS SACRED / ADVERSARIAL SEPARATION.** Forecast code (`forecaster/`,
   `conductor/`, `sources/`) and the evaluation harness + holdout vault (`evaluation/`,
   the guarded set) are separated by design. Forecast/agent code **must never** modify,
   weaken, bypass, or read the internals of the harness, the calibration split, or the
   holdout to make a number look better. The harness is the house. Editing `evaluation/`
   to make a score improve is the single worst thing you can do in this repo. The harness
   may only ever be edited to make it **stricter**, with new tests.

3. **PROPER SCORE, NOT ACCURACY.** Optimize and report Brier (binary) / log / CRPS
   (distributions), with reliability diagrams and ECE, **per domain**, always against
   mandatory baselines (superforecaster median, market/crowd consensus, a strong
   off-the-shelf LLM) with **bootstrap CIs at the question level**. A score in isolation is
   meaningless. Never optimize a threshold/accuracy metric — it teaches overconfidence.

4. **THE TRIALS LEDGER IS LAW.** There is one global, append-only count of evaluations
   against any guarded set, enforced across all agents and experiments. Every retrospective
   score draws down a budget. Silently trying many pipeline/prompt/calibration variants and
   reporting the best is **method-overfitting — the central anti-pattern of this project**,
   and it is forbidden. The ledger exists to make it visible when it happens. If the ledger
   is bypassed or silently broken, every number we publish is meaningless.

5. **CALIBRATION IS LEARNED ON DISJOINT DATA ONLY.** Recalibration functions, extremization
   coefficients, and ensemble weights are fit **only** on a dedicated calibration split —
   never on the holdout, never on the live set. Leaking the holdout into calibration is a
   §2.2 violation wearing a lab coat.

6. **LEAKAGE-FIRST EVALUATION.** Before trusting any retrospective number, run the leakage
   judge and report the leakage rate and the worst-case (flagged-at-chance) robustness. A
   great score on a leaky benchmark is noise, and reporting it as signal is a critical bug.

7. **THE LIVE NUMBER IS THE ONLY REAL ONE.** The nightly live benchmark — forecast
   genuinely open questions, score them on resolution — cannot be tuned because the answers
   do not exist yet. It is the number we publish and the track record we sell. Retrospective
   benchmarks are for development only and are always suspect until leakage-audited.

8. **EVERYTHING IS TESTED. NO EXCEPTIONS.** Every module ships with unit tests **in the same
   PR** — no code merges without them. This is not "leakage and calibration tests only"; it
   is *every unit of behavior*: every function, branch, and error path, plus the edge cases
   (empty inputs, boundary timestamps, missing evidence, malformed questions, provider
   failures). Untested code is treated as broken code, whatever it appears to do. Concretely:
   - **Tests-in-the-same-change.** A feature and its tests land together. A PR that adds or
     changes behavior without adding or updating tests is incomplete and must not merge.
   - **CI is a hard gate.** `pytest` (full suite), `ruff`, and `pyright` must pass before
     merge. A red suite blocks merge. **Never** delete, skip, `xfail`, weaken, or comment out
     a test to make CI green — fix the code or surface the failure (see §11). Making a test
     pass by weakening it is a §2.2-class offense.
   - **Coverage floor.** CI enforces ≥90% line-and-branch coverage repo-wide and **100% on
     `evaluation/`**. The other correctness-critical modules — `core/pit/` (as-of facade),
     `core/forecast/` (aggregation, calibration, leakage judge), `core/registry/` — are
     ratcheting toward 100%: never let them regress, and close their gaps when touching them
     (much of the remainder is Postgres-marked paths that skip without a database). Coverage
     is a floor, not a target — a covered line is not a tested behavior; assert on outputs,
     not just that code ran.
   - **Determinism.** Tests are deterministic and hermetic: seed all randomness, freeze the
     as-of clock, and **mock every network/LLM/provider call** (no live API calls in unit
     tests). A flaky test is a failing test.
   - **The mandatory specialized tests still apply on top of this:** leakage tests for
     anything that reads the world, calibration tests for anything touching aggregation or
     recalibration, and as-of tests proving no path can read past the timestamp (§2.1).

When a task is ambiguous about **look-ahead, the holdout, calibration, or test coverage** —
stop and ask.

---

## 3. Forecast-formation discipline (how a forecast is actually made)

Every forecast flows through this sequence. Each stage is a module with a clean contract so
the conductor (§4) can rearrange, skip, or repeat it. **Do not shortcut the pipeline to a
single model call** — one model run is a sample, not a forecast.

1. **Reference class / base rate** — find reference classes and their frequencies. The
   anchor. Skipping this and reasoning purely inside-view is the classic failure.
2. **Decomposition** — break the question into estimable sub-questions and define how they
   recompose (Fermi estimate, scenario tree, structural model).
3. **Inside-view modeling** — case-specific reasoning that moves off the base rate,
   *justified by retrieved as-of evidence*.
4. **Ensemble** — produce many estimates: across method-agents (base-rate-heavy,
   inside-view-heavy, market-anchored, extrapolation), seeds, and pool models. Diversity is
   the point; decorrelated estimators are what make aggregation work.
5. **Aggregate + extremize** — combine in log-odds space, extremize toward the tails by a
   *learned-on-calibration-data* coefficient. For distributions, mix predictive densities /
   pool quantiles and report a full quantile set, never just a mean.
6. **Calibrate** — map through the recalibrator (isotonic/Platt) fit per §2.5.
7. **Uncertainty** — from ensemble spread + evidence quality; drives the reported band and
   confidence-aware routing (weak/poorly-calibrated domains anchor harder to consensus).

The **leakage judge** runs over the full trace (fixed pipeline *and* conductor-orchestrated
runs alike). The **registry** records the question, the as-of time, the evidence set with
knowledge-time stamps, the full workflow trace, model/version provenance, and the eventual
resolution — immutably. That record is the audit trail, the training corpus, and the track
record, all at once. It is non-negotiable that every forecast writes a complete one.

---

## 4. Orchestration: two stages, second one deferred on purpose

The forecast pipeline in §3 is a fixed workflow. On top of it sits an orchestration layer
(the `conductor/`) whose design follows the Conductor line of work — a learned
orchestrator that assembles and routes a model pool per question and can beat any fixed
scaffold. We adapt it, and we build it in two stages.

**Roles the conductor deploys:** Researcher, Reference-class, Thinker/Decomposer,
Worker/Estimator, Red-team (devil's advocate), Verifier (coherence + leakage, accept/revise),
Aggregator/Extremizer, Calibrator.

**Stage 1 — heuristic conductor (build first, ship first).** A hand-designed, deterministic,
auditable orchestration over those roles. It is a complete, strong product on its own, and
every forecast it produces lands in the registry as a scored
`(question, workflow, evidence, forecast, resolution, proper-score)` tuple. **You cannot
train a learned conductor until this corpus exists — so Stage 1 is also how we generate
Stage 2's training data.**

**Stage 2 — learned conductor (build last).** A Conductor-style model that emits a
natural-language workflow (subtasks, agent ids, access/visibility lists) over the role set,
trained with RL. Three **required adaptations** vs. the papers — get these wrong and the
system regresses:

- **Reward on proper-score improvement, not binary correctness.** A binary-reward conductor
  learns overconfidence. This is the crux.
- **As-of discipline inside the loop.** Every agent the conductor calls reads through the
  as-of facade. It may design any workflow; it may not grant access to the future.
- **Preserve provenance.** Record the full workflow trace to the registry — routing is
  never hidden from the audit trail (see §9).

Anonymize the pool ("Model 0, Model 1…") so the conductor learns forecasting strengths from
reward, not brand priors. **Gate rollout behind the holdout:** the learned conductor replaces
the heuristic in production only if it beats it on the guarded set *without* regressing
calibration or leakage. If it never does, that is a **fine outcome** — the heuristic
conductor is already the product. Stage 2 is upside, not a dependency.

---

## 5. The domain-agnostic `core/` (do not fork)

`core/` is domain-agnostic and designed to be reusable by other applications. It contains
the machinery whose correctness the whole system depends on:

- `core/pit/` — the as-of read facade + bitemporal store. The single read path (§2.1).
- `core/forecast/` — bayesian, ensemble, supervisor, calibration, uncertainty, asof_search,
  agentic_search, leakage_judge.
- `core/registry/` — immutable questions / forecasts / evidence / resolutions.
- `core/memory/` — forecast & experiment recall (pgvector): "have I forecast something like
  this, and how did it resolve?" (built + tested; not yet consumed by the forecast chain —
  see the README roadmap)
- `core/agents/` — agent template + role contracts.
- `core/orchestration/` — loops, schedulers, the conductor interface.

Rules: improve `core/` in place with tests; never add DELPHI-specific domain assumptions
into `core/` (those live in `forecaster/`, `sources/`, `intake/`, `conductor/`); a change
that would couple `core/` to one application's domain goes behind an interface, not into
a fork.

---

## 6. Repository layout

```
.
├── CLAUDE.md                  # this file — primer + prime directives
├── pyproject.toml
├── core/                      # >>> domain-agnostic — DO NOT fork <<< (§5)
├── intake/                    # question typing + normalization into resolvable objects; refusal
├── sources/                   # as-of evidence providers (implement core search Protocol)
├── forecaster/                # concrete general-question ForecastLLM (implements core Protocol)
├── conductor/                 # heuristic conductor + learned conductor + its RL training (§4)
├── resolution/                # ground-truth resolution once a question closes
├── evaluation/                # proper scoring, reliability diagrams, baselines, bootstrap CIs
│                              #   + trials ledger + guarded holdout  (HOUSE — see §2.2, §2.4)
├── benchmarks/                # ForecastBench / Metaculus / market-consensus / live adapters
├── api/                       # published OpenAI-compatible endpoint (DELPHI / DELPHI Deep)  (§9)
└── tests/                     # mirrors the tree 1:1; unit tests for EVERY module (§2.8),
                               #   + mandatory leakage / calibration / as-of tests on top
```

---

## 7. Stack (AWS-native, deliberately minimal)

- **Language/tooling:** Python, `uv`, `ruff`, `pyright`, `pytest`, `pre-commit`. Don't bypass
  the hooks.
- **Spine:** one Postgres (pgvector for memory; the bitemporal store lives here) rather than
  three specialized databases. S3 + Parquet for the evidence-snapshot lake so retrieval is
  reproducible and leakage-auditable.
- **Parallelism:** AWS Batch on Spot for ensemble sweeps and backfilled scoring — this is
  where breadth gets cheap. Graduate to Step Functions/Dagster *when* complexity demands it,
  not before. Premature orchestration bogs these projects down.
- **LLM layer:** Claude via Bedrock (in-VPC), **tiered by capability class** —
  a cheap fast model as the high-volume research/estimator workhorse, a mid model for
  structured estimation, the strongest for the conductor / meta-layer. The pool supports
  **provider opt-out** for compliance (also an enterprise feature; see §9).
  **Do not hardcode model IDs or pricing** — they change. Pin current IDs from the docs
  (https://docs.claude.com/en/docs_site_map.md) and confirm Bedrock availability before
  committing. Verify these when you set it up.

---

## 8. CLI / entry points (illustrative — wire to reality once `common/cli.py` exists)

```
delphi forecast "<question>" --as-of <ts>      # form a single forecast (writes to registry)
delphi intake "<question>"                     # show the normalized, resolvable form (or refusal)
delphi resolve --since <ts>                    # resolve closed questions, feed the registry
delphi eval --suite <name>                     # proper scores + reliability + baselines + CIs
delphi eval --leakage-audit                    # leakage rate + flagged-at-chance robustness
delphi bench live --harvest | --score          # the nightly live benchmark loop
delphi conductor train                         # Stage-2 RL on the registry corpus (§4)
delphi serve                                    # the published API (DELPHI / DELPHI Deep)
```

These assume a single entry point you haven't built yet — treat them as a spec, not a
promise, and reconcile them with `common/cli.py` when it lands.

---

## 9. The published API (product surface)

Two tiers: **DELPHI** (balanced latency/quality, everyday, shallower ensemble) and
**DELPHI Deep** (maximal ensemble + orchestration depth for high-stakes questions).
OpenAI-compatible surface so existing clients point at it with no SDK migration.

Every forecast returns: the **calibrated probability / predictive distribution**; a
**rationale** (decomposition, reference classes, key evidence, the red-team's strongest
counter); **evidence provenance** (sources with knowledge-time stamps); **calibration
metadata** (our historical reliability in this question's domain + an honest confidence
band); the **resolution criteria as DELPHI understood them**; and a **reproducibility
handle** (as-of time, model/version IDs, cache reference) sufficient to reproduce the
forecast exactly.

The emphasis: **expose the evidence and the track record.** A consumer of a probability
needs grounds to trust it, so every response carries provenance and calibration history.
Also ship: provider opt-out, per-request usage/cost reporting, and data-retention opt-out.

---

## 10. Common pitfalls (do not do these)

- Any read path that can see past the as-of time. This is bug #1 and it hides well.
- Fitting a recalibrator, extremization coefficient, or ensemble weight on the holdout/live
  set (§2.5). Overfitting in a lab coat.
- Silent re-runs against a benchmark + cherry-picking; ignoring the trials ledger (§2.4).
- Reporting a proper score without baselines, per-domain breakdown, CIs, and a leakage audit.
- Collapsing the §3 pipeline to a single model call and calling the output a "forecast."
- Rewarding the learned conductor on binary correctness instead of proper score (§4).
- Answering unresolvable questions instead of refusing at intake. That's punditry, not
  forecasting.
- Merging code without unit tests, or making CI green by skipping/weakening a test (§2.8).
- Unit tests that hit live networks/LLMs or depend on wall-clock time — flaky and non-hermetic.
- Hardcoded secrets or model IDs pulled from memory.
- Forking `core/`; adding domain assumptions into `core/`.
- Heavyweight agent frameworks, premature microservices, or a second database before
  Postgres is actually a bottleneck.
- Treating "our number beat the market once" as edge on a liquid market — check whether
  *DELPHI + consensus* beats consensus; additive information is the honest, publishable claim.

---

## 11. Working agreement for Claude (the coding agent)

- Read the relevant module and its tests before changing it. Prefer the smallest change that
  works. Leave `evaluation/`, the calibration split, and the holdout alone unless explicitly
  tasked to make the harness **stricter** — with new tests, never looser.
- **Write unit tests for everything, in the same change (§2.8).** Every function, branch, and
  error path, plus edge cases. No code is "done" until its tests exist and the full suite is
  green. On top of that: **leakage tests are mandatory** for anything that reads the world,
  **calibration tests** for anything touching aggregation or recalibration, and **as-of tests**
  for anything that reads the world through the facade. If you're unsure how to test something,
  that's a signal the design is wrong — make it testable (inject the clock, the provider, the
  model client) rather than skipping the test.
- Run the full suite before you call a task complete: `uv run pytest`, `ruff`, `pyright`, and
  don't bypass `pre-commit`. CI is a hard gate (§2.8); a red suite blocks merge.
- When a task is ambiguous about **look-ahead, the holdout, calibration, or test coverage** —
  stop and ask.
- Never disable a failing gate — including deleting, skipping, or weakening a test — to make a
  task "complete." Surface the failure instead. Weakening a test to go green is a §2.2 offense.
- Every forecast path must write a complete registry record (§3). No silent forecasts.
- Keep this file current when architecture or conventions change — in the same PR.

---

## 12. Build order (see the full design doc for detail)

Correctness mechanisms first, optimization last:

1 CLAUDE.md + prime directives → 2 as-of facade + bitemporal store → 3 registry →
4 intake → 5 evidence/search layer → 6 leakage judge → 7 pipeline (base-rate + decomp +
inside-view) → 8 ensemble + aggregation + extremization → 9 calibration + uncertainty →
10 resolution → 11 eval harness (strict) → 12 trials ledger + guarded holdout →
13 benchmark adapters → 14 heuristic conductor (ships; starts the corpus) → 15 live
benchmark loop → 16 published API → 17 learned conductor (train on the corpus) →
18 learned conductor (holdout-gated rollout).

Milestones 1–13 are an honest, well-governed fixed-pipeline forecaster — already
publishable. 14–16 make it a product with a live track record. 17–18 are the learned-
conductor payoff, and are pure upside gated behind the holdout.

---

*One thing still to settle: the public name (DELPHI vs. final brand). The §8 CLI now
exists at `common/cli.py` (entry point `delphi`); `intake`, `forecast`, `resolve`,
`conductor`, `eval`, `bench live`, `serve`, and `doctor` are wired. `eval --suite
metaculus|forecastbench` and `bench live` run against the Metaculus API and the
ForecastBench dataset repo via `benchmarks/fetchers/` + `benchmarks/suites.py`; the
learned conductor remains the main deferred milestone.*
