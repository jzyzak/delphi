# Registry — Full Documentation

This document describes **every component** of the experiment & strategy registry in `core/registry/`. For a one-page quick start, see [README.md](README.md). The registry builds on the PIT data layer (Prompt 01, `core/pit/`).

The registry is DELPHI's **immutable system of record**: the audit trail and reproducibility backbone every other layer logs to. Because both humans and (later) autonomous agents write to it, it is an **integrity surface**. If the record could be quietly rewritten, the entire evaluation regime could be gamed. Design priorities, in strict order: **append-only**, **tamper-evident**, **reproducible** — correctness and immutability over features.

---

## Table of contents

1. [Conceptual model](#1-conceptual-model)
2. [Module map and public API](#2-module-map-and-public-api)
3. [Reproducibility contract (`models.py`)](#3-reproducibility-contract-modelspy)
4. [Record models (`models.py`)](#4-record-models-modelspy)
5. [Lifecycle state machine (`models.py`)](#5-lifecycle-state-machine-modelspy)
6. [Hashing and fingerprinting (`fingerprint.py`)](#6-hashing-and-fingerprinting-fingerprintpy)
7. [The store: shared logic (`store.py`)](#7-the-store-shared-logic-storepy)
8. [Write API](#8-write-api)
9. [Query API](#9-query-api)
10. [Tamper-evidence and `verify_chain`](#10-tamper-evidence-and-verify_chain)
11. [Backends: in-memory and Postgres](#11-backends-in-memory-and-postgres)
12. [PostgreSQL schema (`migrations/`)](#12-postgresql-schema-migrations)
13. [Concurrency model](#13-concurrency-model)
14. [Error taxonomy](#14-error-taxonomy)
15. [Secret scanning](#15-secret-scanning)
16. [Testing (R1–R8 + §8)](#16-testing-r1r8--8)
17. [Design decisions and deferred work](#17-design-decisions-and-deferred-work)
18. [End-to-end walkthroughs](#18-end-to-end-walkthroughs)
19. [Setup and operational status](#19-setup-and-operational-status)
20. [Known limitations and improvement opportunities](#20-known-limitations-and-improvement-opportunities)

---

## 1. Conceptual model

The registry has four load-bearing ideas. Internalize these; every feature is a direct consequence.

### 1.1 Append-only, event-sourced

Nothing is ever mutated. The registry is an **event log**: each write appends one immutable record. There are three consequences:

- An **`Experiment`** captures the immutable *inputs* of a research trial (hypothesis, economic rationale, full reproducibility metadata). Its **`Result`** (harness output) and **`Decision`** (promote/reject/abandon) are **separate append-only events**, not fields you overwrite on the experiment.
- A strategy's **status is never a column you `UPDATE`**. It is the **fold** over its append-only `LifecycleEvent` stream. To "change" status you append another event.
- **Corrections are new records, never edits.** A later, better result is a new `Result`; a reversed decision is a new `Decision`. The old record stays visible forever. This is what makes the registry an honest audit trail: you can always reconstruct exactly what was known at any point.

### 1.2 Streams

Every record belongs to exactly one **stream**, identified by `stream_id`. There are two stream kinds:

| Stream kind | `stream_id` | Records it holds (in append order) |
|-------------|-------------|------------------------------------|
| `experiment` | the experiment's id | `Experiment` (genesis) → `Result`* → `Decision`* (interleaved) |
| `strategy` | the strategy's id | `Strategy` (genesis) → `StrategyVersion`* / `LifecycleEvent`* (interleaved) |

`*` = zero or more. The first record in a stream (`seq = 0`) is its **genesis** and opens the chain.

Streams are the unit of chaining and concurrency: each has its own hash chain and its own append lock, so unrelated work never contends.

### 1.3 Tamper-evidence (per-stream hash chains)

Every record stores:

- `record_hash` — a SHA-256 content hash of the record (including its position, payload, and append time).
- `prev_hash` — the `record_hash` of the **previous record in the same stream** (`NULL` at `seq = 0`).

This forms a **hash chain per stream**. Editing any stored record changes its `record_hash`, which no longer matches the `prev_hash` of the next record — and `verify_chain` recomputes everything and reports the exact break. Chains are **per-stream, not global**, so appends to different streams never serialize on a single lock (see §13).

### 1.4 Reproducibility is a write precondition

> "A result that can't be reproduced doesn't exist." (CLAUDE.md §2.6)

`record_experiment` **refuses** to record an experiment whose reproducibility bundle is incomplete. The bundle is PIT-native: rather than copying data, it records the `as_of` knowledge-time ceiling plus the universe spec, so the exact dataset is reconstructible from the bitemporal store (01/02). See §3.

### What this prevents

| Anti-pattern | How the registry prevents it |
|--------------|------------------------------|
| Quietly rewriting a bad result to look promotable | No UPDATE/DELETE path (store **and** DB); content-hash chain detects edits |
| "It scored well" with no way to reproduce | Reproducibility bundle is a hard write precondition |
| Silent re-runs to cherry-pick the best of N | Deterministic `trial_fingerprint` makes duplicate trials detectable |
| Forging a promotion | `Decision` must name the deciding component + version; it is chained, not a generic status flip |
| Losing the map of what failed | Failures (reject/abandon) are first-class and fully queryable |
| Leaking secrets into the audit log | Every payload is scanned and rejected if it looks like a credential |

---

## 2. Module map and public API

```
core/registry/
├── __init__.py            Public re-exports
├── models.py              Pydantic models: repro contract, records, lifecycle FSM
├── fingerprint.py         Canonical JSON, content/record hashing, trial_fingerprint
├── store.py               RegistryStore ABC + InMemory + Postgres + verify_chain + queries
├── migrations/
│   └── 0001_init.sql      registry_events table, chain constraints, indexes, triggers
├── README.md              Quick start (< 1 page)
└── DOCUMENTATION.md       This file

tests/registry/
├── __init__.py
├── conftest.py                   Builders (make_repro/make_experiment_input), clocks, fixtures
├── test_models.py                Repro validation + lifecycle FSM + input-model validation
├── test_fingerprint.py           Canonicalization + trial fingerprint determinism (R4)
├── test_store.py                 R1–R8 + query API + secrets + precondition branches
├── test_postgres_mocked.py       Postgres unit tests (no DB required)
└── test_postgres_integration.py  Live Postgres tests (DELPHI_PG_DSN) incl. DB-level R1
```

### Public exports (`core/registry/__init__.py`)

| Export | Type | Purpose |
|--------|------|---------|
| `DataSnapshot` | model | PIT-native dataset descriptor (`as_of` + `universe_spec`) |
| `EnvFingerprint` | model | Python/package versions + image digest (no secrets) |
| `ReproMetadata` | model | The complete reproducibility bundle |
| `ExperimentInput` / `Experiment` | models | Experiment input bundle / stored record |
| `ResultInput` / `Result` | models | Harness output input / stored record |
| `DecisionInput` / `Decision` | models | Promote/reject/abandon input / stored record |
| `StrategyInput` / `Strategy` | models | Strategy genesis input / stored record |
| `StrategyVersionInput` / `StrategyVersion` | models | Versioned strategy input / stored record |
| `LifecycleEventInput` / `LifecycleEvent` | models | Lifecycle transition input / stored record |
| `DecisionOutcome`, `ResultStatus`, `LifecycleState`, `LifecycleEventType` | type aliases | Literal enums |
| `fold_lifecycle`, `next_state` | functions | Lifecycle fold + transition function |
| `IllegalTransitionError` | exception | Illegal lifecycle transition |
| `canonical_json`, `content_hash`, `compute_record_hash`, `trial_fingerprint` | functions | Hashing primitives |
| `RegistryStore` | ABC | Storage-agnostic interface (all logic) |
| `InMemoryRegistryStore` | class | Deterministic reference backend |
| `PostgresRegistryStore` | class | Production Postgres backend |
| `RegistryEvent` | dataclass | One stored, chained log record (envelope) |
| `ChainVerification` | dataclass | Result of `verify_chain` |
| `validate_repro_metadata` | function | Repro write-precondition check |
| `RegistryError`, `RecordNotFoundError`, `IncompleteReproMetadataError`, `SecretInRecordError` | exceptions | Error taxonomy |
| `DuplicateTrialWarning` | warning | Marker for duplicate-trial logging |

---

## 3. Reproducibility contract (`models.py`)

Three frozen Pydantic models compose the bundle that every experiment must carry. All are immutable (`ConfigDict(frozen=True)`) and validate at construction.

### 3.1 `DataSnapshot`

PIT-native description of the dataset an experiment ran against — **no rows are copied**.

| Field | Type | Meaning |
|-------|------|---------|
| `as_of` | `datetime` (tz-aware UTC) | The PIT knowledge-time ceiling. Combined with the PIT store, fixes exactly which facts were visible. |
| `universe_spec` | `dict[str, Any]` | The question-universe filter used (from Prompt 02), e.g. `{"category": "elections", "region": "US"}`. |

Validation:
- `as_of` is normalized to UTC; **naive datetimes are rejected** (`ensure_utc`).
- `universe_spec` **must be non-empty** — an empty spec can't reconstruct a dataset.

Together `(as_of, universe_spec)` make the dataset reconstructible from the bitemporal store, so the snapshot is tiny and immutable.

### 3.2 `EnvFingerprint`

Describes *what ran*, referencing versions/digests only — **never credentials** (CLAUDE.md N3).

| Field | Type | Meaning |
|-------|------|---------|
| `python_version` | `str` (non-empty) | e.g. `"3.12.3"` |
| `packages` | `dict[str, str]` | name → version, e.g. `{"polars": "1.0.0"}` |
| `image_digest` | `str \| None` | Container image digest, e.g. `"sha256:..."` |

`python_version` is validated non-empty. There is deliberately nowhere natural to put a secret; the store additionally scans every payload (see §15).

### 3.3 `ReproMetadata`

The complete bundle. **All fields are required**; the critical strings must be non-empty.

| Field | Type | Meaning |
|-------|------|---------|
| `code_sha` | `str` (non-empty) | Git commit the experiment ran at |
| `dirty` | `bool` | Working-tree dirty flag (was uncommitted code present?) |
| `spec_kind` | `Literal["dsl", "code"]` | How the signal/strategy was expressed |
| `spec_hash` | `str` (non-empty) | Content hash of the serialized spec |
| `params` | `dict[str, Any]` | Strategy/signal parameters |
| `data_snapshot` | `DataSnapshot` | See §3.1 |
| `env` | `EnvFingerprint` | See §3.2 |
| `seeds` | `dict[str, int]` | All RNG seeds (determinism) |

Validation enforced at two layers:

1. **Construction** — Pydantic rejects blank `code_sha` / `spec_hash`, blank `python_version`, empty `universe_spec`, naive datetimes. You cannot build an incomplete bundle through the normal path.
2. **Store precondition** — `validate_repro_metadata(meta)` re-checks every required field and raises `IncompleteReproMetadataError` naming the first gap. This is **defense-in-depth**: it catches bundles smuggled in via `model_construct` (which bypasses validation). `record_experiment` calls it before anything is persisted.

`validate_repro_metadata` checks: `code_sha`, `spec_hash`, `env.python_version` non-empty; `data_snapshot` present with `as_of` set and non-empty `universe_spec`; `params` and `seeds` present.

---

## 4. Record models (`models.py`)

Every record kind has an **`*Input`** model (what callers submit) and a stored model (what queries return). The stored model subclasses the input and adds the store-generated identity and the `knowledge_time` (append time, tz-aware UTC). All are frozen.

This split matters: callers never set ids, fingerprints, or timestamps — the store does, deterministically and under its own clock. Stored records are immutable; assigning to a field raises `ValidationError`.

### 4.1 Experiment

`ExperimentInput`:

| Field | Type | Notes |
|-------|------|-------|
| `hypothesis` | `str` (non-empty) | Falsifiable mechanism statement |
| `economic_rationale` | `str` (non-empty) | Why an edge should exist (CLAUDE.md §10) |
| `author` | `str` (non-empty) | Human or agent id, e.g. `"agent.researcher"` |
| `niche` | `str` (non-empty) | e.g. `"us_elections"` |
| `repro` | `ReproMetadata` | The full bundle (§3) |
| `parent_experiment_id` | `str \| None` | Lineage link to a prior experiment |

`Experiment` (stored) adds: `experiment_id`, `trial_fingerprint`, `knowledge_time`.

### 4.2 Result

`ResultInput`:

| Field | Type | Notes |
|-------|------|-------|
| `experiment_id` | `str` | Stream it attaches to |
| `status` | `Literal["success", "failure", "error"]` | `error` = run didn't complete cleanly |
| `metrics` | `dict[str, Any]` | e.g. `{"brier": 0.18, "log_score": -0.45}` |
| `artifacts` | `dict[str, Any]` | References to plots, equity curves, etc. |

`Result` (stored) adds: `result_id`, `knowledge_time`. The harness math itself (proper scoring, multiple-testing corrections) is **out of scope** (Prompt 06); the registry only *stores* whatever metrics it's given.

### 4.3 Decision

`DecisionInput`:

| Field | Type | Notes |
|-------|------|-------|
| `experiment_id` | `str` | Stream it attaches to |
| `outcome` | `Literal["promote", "reject", "abandon"]` | The disposition |
| `deciding_component` | `str` (non-empty) | Who decided, e.g. `"gates.v1"` |
| `component_version` | `str` (non-empty) | e.g. `"1.0.0"` |
| `rationale` | `str` (non-empty) | Why |
| `evidence` | `dict[str, Any]` | Supporting evidence (gate outputs, etc.) |

`Decision` (stored) adds: `decision_id`, `knowledge_time`.

**Design intent:** a decision is *attributable*. There is deliberately **no** generic "set status = promoted" write. A promotion is a `Decision` record naming the gate and its version, chained into the experiment's stream — so promotions can't be forged and can always be traced to the deciding component.

### 4.4 Strategy / StrategyVersion

`StrategyInput`: `name`, `niche`, `author` (all non-empty). `Strategy` (stored) adds `strategy_id`, `knowledge_time`.

`StrategyVersionInput`:

| Field | Type | Notes |
|-------|------|-------|
| `strategy_id` | `str` | Strategy stream it attaches to |
| `version` | `int` (≥ 1) | Monotonic version number |
| `origin_experiment_id` | `str` | The experiment that produced this version |
| `spec_hash` | `str` | Spec content hash |
| `params` | `dict[str, Any]` | Parameters for this version |

`StrategyVersion` (stored) adds `strategy_version_id`, `knowledge_time`. `origin_experiment_id` is the anchor for lineage: a version resolves to its originating experiment and that experiment's full ancestry (§9, R7).

### 4.5 LifecycleEvent

`LifecycleEventInput`: `strategy_id`, `event` (`Literal["create", "promote", "retire"]`), `rationale` (optional). `LifecycleEvent` (stored) adds `lifecycle_event_id`, `seq` (its position in the strategy stream), `knowledge_time`.

---

## 5. Lifecycle state machine (`models.py`)

A strategy's status is **derived**, never stored. The state machine is intentionally strict and linear.

```
            create            promote            retire
  (none) ──────────▶ candidate ────────▶ promoted ────────▶ retired
```

The transition table (`_TRANSITIONS`):

| Current state | Legal event | Next state |
|---------------|-------------|-----------|
| `None` (empty stream) | `create` | `candidate` |
| `candidate` | `promote` | `promoted` |
| `promoted` | `retire` | `retired` |
| `retired` | — (terminal) | — |

Everything else is illegal, including the explicitly-tested **retire-before-promote** (`candidate` + `retire`), promoting twice, and re-creating.

### Functions

- **`next_state(current, event) -> LifecycleState`** — returns the resulting state, or raises `IllegalTransitionError` if the transition isn't in the table.
- **`fold_lifecycle(events) -> LifecycleState | None`** — folds an ordered list of event types into the current state, applying `next_state` step by step. Returns `None` for an empty stream. Raises `IllegalTransitionError` if the *sequence* is illegal.

### How the store uses it

- `create_strategy` opens the stream **and** appends the initial `create` event, so a fresh strategy is immediately `candidate`.
- `record_lifecycle_event` computes the current state by folding existing events, then calls `next_state(current, new_event)` to validate **before** persisting. An illegal transition raises and **nothing is written** — the rejected event never enters the log.
- `current_state(strategy_id)` returns the fold of the stored lifecycle events, which by construction always matches the validated history.

---

## 6. Hashing and fingerprinting (`fingerprint.py`)

The crux of both tamper-evidence and trial accounting is **canonicalization**: the same logical content must always serialize to the same bytes.

### 6.1 `canonical_json(obj) -> str`

Recursively canonicalizes, then `json.dumps` with `sort_keys=True`, tight separators, and UTF-8:

- **dict** → keys coerced to strings and sorted (so key *order* never matters, at any nesting depth).
- **datetime** → ISO-8601 string (`.isoformat()`); always UTC in practice because models normalize.
- **set** → sorted list (deterministic).
- **list / tuple** → elements canonicalized but **order preserved** (sequence is meaningful).
- scalars → unchanged.

Result: `canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})`, while `canonical_json([1, 2]) != canonical_json([2, 1])`.

### 6.2 `content_hash(obj) -> str`

SHA-256 hex digest of `canonical_json(obj)`. Deterministic and collision-resistant for our purposes; 64 hex chars.

### 6.3 `compute_record_hash(...) -> str`

The per-record tamper-evidence hash. Hashes a body of:

```
{stream_id, seq, record_kind, record_id, payload, prev_hash, knowledge_time}
```

Because `prev_hash` is included, the hash chains: editing any earlier record changes its `record_hash`, which then mismatches the `prev_hash` baked into the next record's hash. Editing the record itself changes its own `record_hash` vs. the recomputed value. Either way `verify_chain` catches it (§10).

### 6.4 `trial_fingerprint(meta) -> str` — the key accounting primitive

```python
trial_fingerprint(meta) = content_hash({
    "spec_hash":     meta.spec_hash,
    "params":        meta.params,
    "data_snapshot": {"as_of": meta.data_snapshot.as_of,
                      "universe_spec": meta.data_snapshot.universe_spec},
    "universe_spec": meta.data_snapshot.universe_spec,
})
```

A **trial** is identified by **spec + params + dataset + universe** — and **deliberately excludes** `code_sha`, `env`, `seeds`, and `dirty`.

This is a hard-to-reverse, intentional design choice:

- **Same trial across builds.** Re-running the identical spec/params/dataset on a new git commit or new seed yields the **same** fingerprint. That's the point: it's the same logical trial.
- **Anti-cherry-pick.** The later trials ledger (Prompt 06) uses the fingerprint to count distinct trials honestly and to detect an agent silently re-running the same trial to fish for a lucky result.
- **Sensitivity.** Any change to spec, params, `as_of`, or universe produces a **new** fingerprint.

The fingerprint is computed and stored at `record_experiment` time. `duplicate_experiment_ids(fp)` returns every experiment already recorded under that fingerprint; recording a duplicate is **logged** (structured `duplicate_trial_recorded` warning) but **never blocked** — the log is append-only and the trials ledger, not the registry, decides what duplication means.

---

## 7. The store: shared logic (`store.py`)

The architecture deliberately concentrates *all* behavior on the abstract base class, so the two backends can't drift apart.

```
RegistryStore (ABC)  ── all write semantics, hashing, chaining, fingerprinting,
                         lifecycle validation, queries, verify_chain
   │
   ├── InMemoryRegistryStore   implements 3 storage primitives (per-stream locks)
   └── PostgresRegistryStore   implements 3 storage primitives (advisory locks + triggers)
```

### Subclass contract — only three primitives

| Primitive | Responsibility |
|-----------|----------------|
| `_persist(...)` | Atomically append one record to a stream: determine `seq` and `prev_hash` from the current tail **under a per-stream lock**, stamp `knowledge_time` from `self._now()`, compute the chained hash, and store — never losing or reordering a concurrent same-stream append. |
| `_stream_events(stream_id)` | Return all events for a stream ordered by ascending `seq`. |
| `_events_of_kind(record_kind)` | Return all events of a kind across streams (for author/niche/outcome/dedup/lineage queries). |

Everything else — `record_experiment`, all queries, `verify_chain`, lifecycle folding — is implemented once on the base in terms of those three. A shared helper `_build_event(...)` computes the chained `record_hash` and assembles the immutable `RegistryEvent`; both backends call it inside their critical section.

### The clock

The base takes an injectable `clock: Callable[[], datetime]` (default `datetime.now(UTC)`), exposed via `_now()` which forces tz-aware UTC. Tests inject a deterministic `IncrementingClock`. The **only** wall-clock read that correctness depends on is the append `knowledge_time` (CLAUDE.md constraint), and it's injectable for determinism.

### `RegistryEvent` (the envelope)

A frozen dataclass — the physical, stored representation:

| Field | Meaning |
|-------|---------|
| `stream_id`, `stream_kind` | Which stream and its kind |
| `seq` | 0-based position within the stream |
| `record_kind` | One of the six kinds |
| `record_id` | Unique id of this record (e.g. `exp_…`, `res_…`) |
| `payload` | JSON-safe dict; the typed record body minus `knowledge_time` |
| `prev_hash` | Prior record's `record_hash` (`None` at `seq 0`) |
| `record_hash` | This record's content hash |
| `knowledge_time` | Append time (tz-aware UTC) |

The payload omits `knowledge_time`; queries reconstruct a typed model by merging `payload` + the event's `knowledge_time` (helpers `_to_experiment`, `_to_result`, …). Ids are generated as `"{prefix}_{uuid4().hex}"` (`exp`, `res`, `dec`, `strat`, `ver`, `lc`).

---

## 8. Write API

All writes append; none mutate. Each builds a JSON-safe payload, scans it for secrets (§15), and calls `_persist`.

### `record_experiment(exp: ExperimentInput) -> str`

1. `validate_repro_metadata(exp.repro)` — reject incomplete bundles (`IncompleteReproMetadataError`).
2. If `parent_experiment_id` is set, `get_experiment(parent)` — **a parent reference must resolve** (`RecordNotFoundError`), guaranteeing lineage always terminates.
3. Generate `experiment_id`; compute `trial_fingerprint(exp.repro)`.
4. Look up existing experiments with that fingerprint (`duplicate_experiment_ids`).
5. Build payload (`experiment_id`, `trial_fingerprint`, plus the input bundle), scan for secrets, `_persist` as the genesis (`seq 0`) of a new `experiment` stream.
6. If duplicates existed, emit a structured `duplicate_trial_recorded` warning. **Never blocks.**
7. Return `experiment_id`.

### `record_result(result: ResultInput) -> str`

Verifies the experiment exists, then appends a `result` record to that experiment's stream. Returns `result_id`.

### `record_decision(decision: DecisionInput) -> str`

Verifies the experiment exists, then appends an attributable `decision` record (carrying `deciding_component` + `component_version` + `rationale` + `evidence`) to the experiment's stream. Returns `decision_id`.

### `create_strategy(strategy: StrategyInput) -> str`

Opens a new `strategy` stream (genesis `Strategy` record) **and** appends the initial `create` `LifecycleEvent`, leaving the strategy in state `candidate`. Returns `strategy_id`.

### `record_strategy_version(version: StrategyVersionInput) -> str`

Verifies both the strategy and the `origin_experiment_id` exist, then appends a `strategy_version` record. Returns `strategy_version_id`.

### `record_lifecycle_event(event: LifecycleEventInput) -> str`

Verifies the strategy exists, folds existing lifecycle events to the current state, validates the transition with `next_state` (raises `IllegalTransitionError` and writes nothing if illegal), then appends a `lifecycle_event`. Returns `lifecycle_event_id`.

---

## 9. Query API

All queries are read-only and reconstruct typed models from events. Results that span streams are sorted deterministically by `(knowledge_time, id)`.

### Experiment retrieval & lineage

| Method | Returns |
|--------|---------|
| `get_experiment(experiment_id)` | The `Experiment`, or `RecordNotFoundError` |
| `results_for(experiment_id)` | `tuple[Result, ...]` in append order |
| `decisions_for(experiment_id)` | `tuple[Decision, ...]` in append order |
| `experiments_by_author(author)` | All experiments by that author/agent |
| `experiments_by_niche(niche)` | All experiments in a niche (**failures included**) |
| `experiments_by_outcome(outcome)` | Experiments whose **latest** decision has that outcome |
| `duplicate_experiment_ids(fingerprint)` | All experiment ids sharing a trial fingerprint (dedup view) |
| `experiment_children(experiment_id)` | Direct children (downward lineage) |
| `experiment_lineage(experiment_id)` | Ancestry chain **root → self** (upward), with cycle guard |

`experiments_by_outcome` groups all `decision` events by experiment, keeps the one with the **highest `seq`** (the latest decision wins, so a reversal supersedes the original), then filters by outcome. This is how **failures are first-class**: a rejected/abandoned experiment is as queryable as a promoted one, by both `niche` and `outcome` (R6).

`experiment_lineage` walks `parent_experiment_id` upward; because parent references are validated at write time, the walk always resolves. A `RegistryError` is raised if a cycle is ever detected (defensive).

### Strategy retrieval & lifecycle

| Method | Returns |
|--------|---------|
| `get_strategy(strategy_id)` | The `Strategy`, or `RecordNotFoundError` |
| `strategy_versions(strategy_id)` | `tuple[StrategyVersion, ...]` in append order |
| `lifecycle_events(strategy_id)` | `tuple[LifecycleEvent, ...]` in append order |
| `current_state(strategy_id)` | The folded `LifecycleState` (or `None`) |
| `strategies_by_state(state)` | All strategies whose folded state equals `state` |
| `strategy_version_ancestry(strategy_id, version)` | Full experiment ancestry behind a version (resolves origin experiment → `experiment_lineage`); `RecordNotFoundError` for an unknown version |

`strategy_version_ancestry` is the R7 link: it finds the version, resolves its `origin_experiment_id`, and returns that experiment's full ancestry — connecting a deployed strategy version back to the entire experiment chain that produced it.

---

## 10. Tamper-evidence and `verify_chain`

### `verify_chain(stream_id) -> ChainVerification`

Walks the stream in `seq` order and, for each record, checks three invariants:

1. **Dense, contiguous `seq`** — `ev.seq == index`. A gap or reorder is a break.
2. **Linkage** — `ev.prev_hash` equals the previous record's `record_hash` (`None` at `seq 0`).
3. **Content integrity** — `compute_record_hash(...)` recomputed from the stored fields equals the stored `record_hash`. A mismatch means the record's content was altered after the fact.

On the **first** failing record it returns:

```python
ChainVerification(
    stream_id=...,
    ok=False,
    broken_at_seq=...,        # exact position
    broken_record_id=...,     # exact record
    reason="record_hash does not match content (record was altered)"  # or linkage/seq reason
)
```

A clean (or empty/nonexistent) stream returns `ChainVerification(stream_id=..., ok=True)`.

This makes any retroactive edit **detectable and localizable**. In the test suite (R2) a stored record's payload is mutated in place without re-hashing; `verify_chain` reports the exact broken link. At the database layer, such an edit can't even happen — triggers forbid `UPDATE`/`DELETE` (§12) — so `verify_chain` is the backstop for in-memory/transport tampering and an independent integrity audit.

---

## 11. Backends: in-memory and Postgres

Both share all logic; they differ only in the three storage primitives and in durability/concurrency mechanics.

### `InMemoryRegistryStore`

The deterministic reference (tests, local dev). State:

- `_events: list[RegistryEvent]` — global append order.
- `_by_stream: dict[str, list[RegistryEvent]]` — per-stream order (= `seq` order).
- `_meta_lock` — guards the structures and the lock registry.
- `_stream_locks: dict[str, Lock]` — one lock per stream.

`_persist` acquires the **per-stream** lock, reads the tail to compute `seq`/`prev_hash`, builds the event, and appends under the meta lock. Per-stream locking means appends to different streams proceed concurrently while same-stream appends serialize to keep the chain valid (§13).

### `PostgresRegistryStore`

The production spine.

- `PostgresRegistryStore.connect(dsn, *, migrate=True, clock=None)` opens a `psycopg` connection and (by default) applies migrations.
- `apply_migrations()` runs every `core/registry/migrations/*.sql` in sorted order, then commits.
- `_persist` runs in a single transaction:
  1. `SELECT pg_advisory_xact_lock(hashtext(stream_id))` — a **per-stream** advisory lock, auto-released at commit, so only appends to the *same* stream serialize.
  2. `SELECT seq, record_hash … ORDER BY seq DESC LIMIT 1` — the current tail.
  3. Compute `seq`/`prev_hash`/hash; `INSERT` the row with the explicitly-computed `knowledge_time`.
- Reads (`_stream_events`, `_events_of_kind`) map rows back to `RegistryEvent`s; JSONB is parsed by `_parse_jsonb` (accepts dict or JSON string; rejects anything else with `TypeError`).
- Implements the context-manager protocol (`__enter__`/`__exit__` → `close()`).

Both backends were validated to produce identical behavior: the full registry suite (91 tests) passes against a live Postgres 16 at 98% coverage, including the Postgres read/write paths.

---

## 12. PostgreSQL schema (`migrations/0001_init.sql`)

A single append-only table, `registry_events`, is the source of truth.

### Columns

| Column | Type | Notes |
|--------|------|-------|
| `id` | `BIGINT` identity PK | Physical row id |
| `stream_id` | `TEXT` | Experiment or strategy id |
| `stream_kind` | `TEXT` | `CHECK in ('experiment','strategy')` |
| `seq` | `BIGINT` | `CHECK >= 0`; 0-based position in stream |
| `record_kind` | `TEXT` | `CHECK in` the six kinds |
| `record_id` | `TEXT` | Unique record id |
| `payload` | `JSONB` | Record body (no `knowledge_time`, no secrets) |
| `prev_hash` | `TEXT` | Prior record's hash (`NULL` at genesis) |
| `record_hash` | `TEXT` | This record's content hash |
| `knowledge_time` | `TIMESTAMPTZ` | Append time (UTC) |

### Constraints (chain integrity)

- `UNIQUE (stream_id, seq)` — **dense, unique positions per stream**. A racing append that computes the same `seq` fails this check and retries; no write is lost and the chain can't fork.
- `UNIQUE (record_id)` — record ids are globally unique.
- `CHECK ((seq = 0) = (prev_hash IS NULL))` — genesis (and only genesis) is unchained; every later record must carry a `prev_hash`.

### Indexes (query paths)

`(stream_id, seq)`, `(record_kind)`, and JSONB expression indexes on `payload->>'author'`, `'niche'`, `'trial_fingerprint'`, `'parent_experiment_id'`, and `'outcome'`.

### Append-only enforcement (DB level)

```sql
CREATE FUNCTION registry_reject_mutation() RETURNS TRIGGER ...
  RAISE EXCEPTION 'registry is append-only: UPDATE and DELETE are forbidden on %';
CREATE TRIGGER registry_events_no_update BEFORE UPDATE ... EXECUTE FUNCTION registry_reject_mutation();
CREATE TRIGGER registry_events_no_delete BEFORE DELETE ... EXECUTE FUNCTION registry_reject_mutation();
```

Even a privileged client issuing a raw `UPDATE`/`DELETE` is rejected (verified by integration tests R1). Append-only is enforced at **both** the store layer (no such method exists) and the database.

---

## 13. Concurrency model

The goal (CLAUDE.md N2): concurrent appends to **different** streams don't block each other and never lose a write; per-stream chains stay valid under concurrency.

- **Different streams → no contention.** Locks/advisory locks are keyed by `stream_id`, so unrelated experiments and strategies append in parallel.
- **Same stream → serialized.** Two appends to one stream are ordered so each sees the other's `record_hash` as its `prev_hash`. In-memory uses the per-stream `Lock`; Postgres uses the per-stream advisory xact lock plus `UNIQUE(stream_id, seq)` as a backstop (a loser retries rather than corrupting the chain).
- **No lost writes.** Every accepted append lands at a unique `seq`.

Tested by R8: 50 concurrent appends across distinct streams all persist with valid chains; 40 concurrent results to a *single* stream produce a contiguous `seq 0..40` and a passing `verify_chain`.

> Note: a single `psycopg` connection is not itself thread-safe, so true multi-threaded Postgres concurrency requires a connection per worker; the advisory-lock + unique-constraint design makes that safe. The integration test exercises many independent streams sequentially over one connection.

---

## 14. Error taxonomy

| Exception | Base(s) | Raised when |
|-----------|---------|-------------|
| `RegistryError` | `Exception` | Base for all registry errors (e.g. lineage cycle) |
| `RecordNotFoundError` | `RegistryError`, `KeyError` | A referenced experiment/strategy/version doesn't exist |
| `IncompleteReproMetadataError` | `RegistryError`, `ValueError` | Experiment write with an incomplete repro bundle |
| `SecretInRecordError` | `RegistryError`, `ValueError` | A payload appears to contain a credential |
| `IllegalTransitionError` | `ValueError` | An illegal/out-of-order lifecycle transition |

Multiple inheritance (e.g. `RecordNotFoundError(RegistryError, KeyError)`) lets callers catch either the registry-specific type or the familiar built-in.

---

## 15. Secret scanning

Before any record is persisted, `_assert_no_secrets(payload)` recursively walks the payload (dicts and lists). If a **key** (case-insensitive) contains any marker — `password`, `passwd`, `secret`, `token`, `credential`, `api_key`, `apikey`, `access_key`, `private_key`, `aws_secret` — and its value is truthy, it raises `SecretInRecordError` naming the path.

This enforces CLAUDE.md N3 (env fingerprints reference versions/digests, not credentials) structurally: a `params={"api_key": "..."}` or a `Decision.evidence={"aws_secret_access_key": "..."}` is rejected at write time, and nested-in-a-list secrets (`{"logs": [{"api_key": "leak"}]}`) are caught too. The audit log can never accumulate secrets.

---

## 16. Testing (R1–R8 + §8)

Tested to CLAUDE.md §8 (happy/boundary/failure, determinism, AAA, behavior-naming) plus the prompt's mandatory component cases. Time is injected; no network. Suite: **87 tests** in-memory (plus 4 Postgres integration), **97% coverage** (models 100%, fingerprint 100%, store 95% in-memory / 97% with Postgres).

| Case | What it proves | Where |
|------|----------------|-------|
| **R1 Immutability** | No `update`/`delete`/`set_status` method exists; stored records are frozen (assignment raises); corrections are new records; **DB rejects raw `UPDATE`/`DELETE`** | `test_store.py::TestImmutability`, `test_postgres_integration.py` |
| **R2 Tamper-evidence** | Clean chain verifies; an in-place edit to a stored record reports the exact broken link; empty stream is OK | `test_store.py::TestTamperEvidence` |
| **R3 Repro round-trip + precondition** | A fully-specified experiment round-trips (including `repro` and fingerprint); a smuggled-in incomplete bundle is rejected (`code_sha`, `spec_hash`, `universe_spec`, `data_snapshot`, `params/seeds`) | `test_store.py::TestReproRoundTripAndPrecondition`, `::TestReproPreconditionBranches` |
| **R4 Fingerprint / dedup** | Identical trials share a fingerprint and are flagged as duplicates; changing params/spec/as_of/universe yields a new fingerprint; key order is irrelevant; build/seed changes do **not** change it | `test_fingerprint.py`, `test_store.py::TestTrialDedup` |
| **R5 Lifecycle event-sourcing** | `create`→candidate; full fold to `retired`; retire-before-promote rejected and not persisted; `current_state` matches the fold; `strategies_by_state` | `test_models.py::TestLifecycleStateMachine`, `test_store.py::TestLifecycleEventSourcing` |
| **R6 Failures first-class** | Rejected/abandoned experiments queryable by outcome and niche; latest decision wins; failure results retrievable | `test_store.py::TestFailuresFirstClass` |
| **R7 Lineage** | Parent/child traversal; unknown parent rejected; a strategy version resolves to its origin experiment and full ancestry | `test_store.py::TestLineage` |
| **R8 Concurrency** | Concurrent appends to different streams all persist with valid chains; concurrent appends to one stream keep a contiguous, valid chain | `test_store.py::TestConcurrency` |

Plus: secret rejection (params, decision evidence, nested-in-list), unknown-reference rejection, `experiments_by_author`, `decisions_for` ordering, input-model validation (blank fields, `version >= 1`), and Postgres-mocked unit tests (seq/prev chaining, row mapping, migrations, JSONB parsing).

### Running

```bash
uv run pytest tests/registry                       # in-memory (fast, deterministic)
uv run ruff check . && uv run pyright              # lint + types (clean)

# Integration against a real DB (DB-level append-only enforcement, round-trip):
DELPHI_PG_DSN=postgresql://user:pass@localhost:5432/delphi \
  uv run pytest tests/registry -m postgres
```

---

## 17. Design decisions and deferred work

### Hard-to-reverse choices (flagged at design time)

1. **One unified event log, not six per-record tables.** The hash chain inside an experiment stream spans `Experiment → Result → Decision`; splitting those across tables would force a fragile cross-table chain and complicate `verify_chain`. A single append-only, hash-chained log is the correct event-sourcing primitive; the six "kinds" are discriminated rows with typed JSONB payloads. This deviates from the literal table list in the prompt and was flagged explicitly.
2. **`trial_fingerprint` excludes `code_sha`/`env`/`seeds`/`dirty`.** A re-run on a different build/seed is the *same* trial — which is exactly what enables honest trial counting and silent-rerun detection downstream (Prompt 06). Changing this later would redefine trial identity, so it was called out.
3. **Stream partitioning:** `stream_id = experiment_id` (experiment/result/decision) and `stream_id = strategy_id` (strategy/version/lifecycle). Per-stream chains + `UNIQUE(stream_id, seq)` + per-stream advisory locks deliver non-blocking cross-stream concurrency with no lost writes.

### Deliberately out of scope (deferred to later prompts)

| Deferred | Prompt |
|----------|--------|
| Trials ledger, multiple-testing math | 06 |
| Semantic / vector recall over the registry | 10 |
| Promotion gates (the bar a strategy must clear) | 07 |
| Research agents | 11 |

The registry only **stores** experiments and computes the `trial_fingerprint` those layers will consume. It records metrics and decisions; it does **not** compute or judge them.

---

## 18. End-to-end walkthroughs

### A. A failed experiment (the common case)

```python
from core.registry import (
    InMemoryRegistryStore, ExperimentInput, ResultInput, DecisionInput,
    ReproMetadata, DataSnapshot, EnvFingerprint,
)
from datetime import datetime, UTC

store = InMemoryRegistryStore()

repro = ReproMetadata(
    code_sha="9f1c2a", dirty=False, spec_kind="dsl", spec_hash="spec-abc",
    params={"lookback": 20, "threshold": 1.5},
    data_snapshot=DataSnapshot(
        as_of=datetime(2024, 6, 1, tzinfo=UTC),
        universe_spec={"category": "elections", "region": "US"},
    ),
    env=EnvFingerprint(python_version="3.12.3", packages={"polars": "1.0.0"}),
    seeds={"numpy": 7},
)

exp_id = store.record_experiment(ExperimentInput(
    hypothesis="Polling averages underestimate incumbent turnout in midterms.",
    economic_rationale="Sparse local polling underreacts to late registration data.",
    author="agent.researcher",
    niche="us_elections",
    repro=repro,
))

store.record_result(ResultInput(
    experiment_id=exp_id, status="failure", metrics={"brier": 0.31},
))
store.record_decision(DecisionInput(
    experiment_id=exp_id, outcome="reject",
    deciding_component="gates.v1", component_version="1.0.0",
    rationale="Brier score worse than baseline after trials correction.",
))

# The failure is first-class and queryable — it maps what doesn't work.
assert exp_id in {e.experiment_id for e in store.experiments_by_outcome("reject")}
assert exp_id in {e.experiment_id for e in store.experiments_by_niche("us_elections")}
assert store.verify_chain(exp_id).ok
```

### B. Iterating, then promoting a strategy

```python
# Parent → child experiment (lineage), then a strategy version off the child.
parent = store.record_experiment(ExperimentInput(..., repro=repro))
child  = store.record_experiment(ExperimentInput(..., repro=repro2,
                                                 parent_experiment_id=parent))

from core.registry import StrategyInput, StrategyVersionInput, LifecycleEventInput

sid = store.create_strategy(StrategyInput(
    name="polling-drift", niche="us_elections", author="agent.researcher",
))
assert store.current_state(sid) == "candidate"            # the [create] fold

store.record_strategy_version(StrategyVersionInput(
    strategy_id=sid, version=1, origin_experiment_id=child, spec_hash="spec-def",
))

# A version traces back to its full experiment ancestry (R7):
assert [e.experiment_id for e in store.strategy_version_ancestry(sid, 1)] == [parent, child]

# Lifecycle transitions are event-sourced and validated:
store.record_lifecycle_event(LifecycleEventInput(strategy_id=sid, event="promote"))
assert store.current_state(sid) == "promoted"
assert sid in {s.strategy_id for s in store.strategies_by_state("promoted")}
```

### C. Detecting a silent re-run (anti-cherry-pick)

```python
a = store.record_experiment(ExperimentInput(..., repro=repro))
b = store.record_experiment(ExperimentInput(..., repro=repro))   # same spec/params/dataset

fp = store.get_experiment(a).trial_fingerprint
assert store.get_experiment(b).trial_fingerprint == fp           # same TRIAL
assert set(store.duplicate_experiment_ids(fp)) == {a, b}         # both flagged
# (Prompt 06's trials ledger consumes this to count honestly.)
```

### D. Tamper detection

```python
chk = store.verify_chain(exp_id)         # clean stream
assert chk.ok

# If any stored record were edited in place without re-hashing,
# verify_chain would return ok=False with broken_at_seq / broken_record_id
# pointing at the exact altered record. At the Postgres layer the edit itself
# is impossible — UPDATE/DELETE triggers reject it.
```

---

## 19. Setup and operational status

**Bottom line: the registry is library-complete and needs no additional setup to use in-process.** What remains is operational hardening and downstream wiring, none of which blocks usage today.

### 19.1 Already done (no action needed)

| Item | Status |
|------|--------|
| Runtime dependencies (`pydantic`, `polars`, `structlog`, `psycopg[binary]`) | Declared in `pyproject.toml`; installed by `uv sync` |
| Package wiring | `registry` is in the hatch build targets, `pyright` `include`, and `coverage` `source` |
| Schema | `migrations/0001_init.sql`; **auto-applied** by `PostgresRegistryStore.connect(dsn)` (default `migrate=True`) |
| CI | CI runs `ruff`, `ruff format --check`, `pyright`, and `pytest` over the whole repo (registry included) |
| In-memory backend | Works out of the box; zero external services |
| Tests | 87 in-memory tests pass; 4 Postgres integration tests are **skipped** unless `DELPHI_PG_DSN` is set |

### 19.2 To use the in-memory backend

Nothing. `from core.registry import InMemoryRegistryStore` and go. This is the right choice for unit tests, local research loops, and any component that doesn't need durability.

### 19.3 To use the Postgres backend

1. Have a reachable **PostgreSQL 16** instance. The registry uses only core Postgres — **no `pgvector` extension is required** (that serves other DELPHI layers).
2. Provide a DSN and connect; migrations apply automatically:

```python
from core.registry import PostgresRegistryStore
store = PostgresRegistryStore.connect("postgresql://user:pass@host:5432/delphi")
```

3. To exercise the live integration suite (including DB-level UPDATE/DELETE rejection):

```bash
# Example local DB via Docker:
docker run -d --name delphi-pg -e POSTGRES_PASSWORD=delphi -e POSTGRES_DB=delphi \
  -p 5432:5432 postgres:16
DELPHI_PG_DSN=postgresql://postgres:delphi@localhost:5432/delphi \
  uv run pytest tests/registry -m postgres
```

### 19.4 Open setup items (not blocking, but worth doing to "finish")

These are gaps relative to the broader vision in `CLAUDE.md`, not defects in the registry:

1. **Postgres in CI.** CI does **not** start a Postgres service, so the DB-level append-only enforcement (R1 triggers) and the Postgres read/write paths are only validated locally. Add a `services: postgres` block and set `DELPHI_PG_DSN` so the `-m postgres` tests run on every push. This is the single highest-value setup item.
2. **Migration version ledger.** `apply_migrations()` re-executes every `*.sql` on each connect. Migration `0001` is written idempotently (`IF NOT EXISTS`, `CREATE OR REPLACE`, `DROP TRIGGER IF EXISTS`), so this is safe **today**, but there is no `schema_migrations` table tracking what has run. Before a second, non-idempotent migration lands, add a version ledger (record applied filenames, skip already-applied).
3. **`delphi` CLI surface.** The `delphi` entry point now exists (`common/cli.py`), but registry-specific commands (e.g. `delphi registry verify <stream>`, `delphi registry show <experiment>`) are not yet added to it.
4. **Repo-level secret scanning / pre-commit.** `CLAUDE.md §7` calls for a `pre-commit` config (ruff, pyright, secret scan). The registry has a *runtime* payload secret scan (§15), but there is no repo `.pre-commit-config.yaml`. This is a repo-wide gap, not registry-specific, but the two are complementary.
5. **Centralized config for the DSN.** The store takes a raw DSN string. When a shared settings object (pydantic) lands, route the DSN and secrets through it (Secrets Manager / env), rather than passing connection strings around.
6. **No downstream consumer yet.** The registry is a ready-to-import library; the orchestrator, agents (11), and gates (07) that will *write* to it are later prompts. "Finishing" in the product sense means those layers calling `record_experiment` / `record_decision` — out of scope here.

---

## 20. Known limitations and improvement opportunities

Honest accounting of where the current implementation trades simplicity for headroom. None of these are correctness bugs; they are scaling and hardening opportunities.

### 20.1 Scale and performance

| # | Limitation | Improvement |
|---|------------|-------------|
| 1 | **Queries filter in Python.** `experiments_by_author/niche/outcome`, `duplicate_experiment_ids`, `experiment_children`, and `strategies_by_state` fetch all events of a kind via `_events_of_kind` and filter in process. | Push predicates into SQL `WHERE` clauses — the JSONB expression indexes (`author`, `niche`, `outcome`, `trial_fingerprint`, `parent_experiment_id`) already exist to serve them. Add per-query Postgres methods rather than reusing `_events_of_kind`. |
| 2 | **Lineage does N round-trips.** `experiment_lineage` calls `get_experiment` once per ancestor; `strategy_version_ancestry` then chains into it. | Use a recursive CTE in Postgres to resolve an entire ancestry in one query. |
| 3 | **`strategies_by_state` re-folds every strategy.** It loads each strategy and re-reads its lifecycle stream. | Maintain a derived (but still non-authoritative) state projection / materialized view refreshed on append, keeping the fold as the source of truth. |
| 4 | **Unbounded result sets.** Queries return full tuples with no pagination or limit. | Add `limit`/`offset` (or keyset pagination on `seq`/`id`) to all list queries. |
| 5 | **`verify_chain` loads the whole stream.** Fine for small streams; expensive for very long ones. | Offer an incremental/streaming verify and a `verify_all()` background audit that checkpoints progress. |

### 20.2 Concurrency and durability

| # | Limitation | Improvement |
|---|------------|-------------|
| 6 | **Single connection isn't thread-safe.** True parallel appends across streams need a connection per worker; the Postgres integration test therefore only exercises *sequential* multi-stream correctness. | Provide a `psycopg_pool`-backed store so the per-stream advisory-lock design can be exercised under real parallelism. |
| 7 | **No retry/backoff helper.** The `UNIQUE(stream_id, seq)` backstop means a racing same-stream append must retry; today the advisory lock makes that rare, but there's no explicit retry wrapper. | Add bounded retry-on-unique-violation around `_persist` for pool-based concurrency. |

### 20.3 Integrity hardening

| # | Limitation | Improvement |
|---|------------|-------------|
| 8 | **No hash-scheme version.** `record_hash` embeds no algorithm/canonicalization version, so evolving the canonical form would break verification of old records. | Add a `hash_version` field included in the hashed body; verify with the version a record was written under. |
| 9 | **Per-stream chains only.** Chains detect edits and reordering *within* a stream and the DB triggers prevent row deletion, but there is no cross-stream / global anchor. A stream that was never created can't be "missing." | Optionally anchor periodic Merkle roots (or an append-only global counter) for tamper-evidence across the whole log. |
| 10 | **Heuristic secret scan.** §15 matches credential-like *key names*; it can miss a secret stored under an innocuous key and can false-positive on a legitimate key containing `token`. | Add value-pattern detection (e.g. AWS/key regexes) plus a small allowlist; consider failing closed with an override flag. |

### 20.4 Domain invariants and ergonomics

| # | Limitation | Improvement |
|---|------------|-------------|
| 11 | **`StrategyVersion.version` uniqueness isn't enforced.** Nothing prevents two `version=1` records on one strategy; `strategy_version_ancestry` would then resolve the first match ambiguously. | Fold existing versions in `record_strategy_version` and reject a duplicate version number (mirrors the lifecycle-transition validation pattern). |
| 12 | **Duplicate-trial result is implicit.** `record_experiment` logs a `duplicate_trial_recorded` warning but returns only the id; callers must re-query `duplicate_experiment_ids`. | Optionally return a small typed result (`experiment_id`, `is_duplicate`, `prior_ids`) so callers don't re-query. |
| 13 | **No optional ordering invariants.** A `Decision` can precede any `Result`; this is intentional (append-only, flexible), but some callers may want "result-before-decision" guarantees. | Offer opt-in invariant checks rather than hard-coding them. |
| 14 | **Observability is minimal.** Only duplicate trials are logged. | Emit structured logs/metrics on every append (counts per kind, chain length, verify outcomes) for monitoring (Prompt 12). |

### 20.5 Priority ordering (if picking up next)

1. **Postgres in CI** (§19.4 #1) — closes the biggest validation gap with the least effort.
2. **SQL query pushdown** (§20.1 #1–2) — needed before the registry holds many thousands of experiments.
3. **Migration version ledger** (§19.4 #2) — required before the second migration.
4. **`StrategyVersion` uniqueness** (§20.4 #11) — a small correctness-adjacent invariant worth adding early.
5. Everything else is genuine "later, at scale" hardening.
