# Experiment & Strategy Registry

> **Full documentation:** [DOCUMENTATION.md](DOCUMENTATION.md) — every feature, model,
> query, and design decision in depth. This page is the one-page quick start.

The registry is DELPHI's **immutable system of record**: the audit trail and
reproducibility backbone every other layer logs to. Because humans and (later)
autonomous agents both write to it, it is an **integrity surface** — if the
record could be quietly rewritten, the whole evaluation regime could be gamed.

Design priorities, in order: **append-only**, **tamper-evident**,
**reproducible**. Correctness and immutability over features.

## Record model

Six record kinds live in one append-only event log (`registry_events`). Each
record belongs to a **stream** and is content-hashed and chained to the prior
record in that stream.

| Kind | Stream | Role |
|------|--------|------|
| `Experiment` | `experiment` (`stream_id = experiment_id`) | Immutable *input* bundle: hypothesis, economic rationale, full reproducibility metadata, lineage |
| `Result` | same experiment stream | Append-only harness *output* (metrics, artifacts, status) |
| `Decision` | same experiment stream | Append-only promote / reject / abandon, naming the deciding component + version |
| `Strategy` | `strategy` (`stream_id = strategy_id`) | Strategy genesis record |
| `StrategyVersion` | same strategy stream | A concrete versioned instantiation, linked to its originating experiment |
| `LifecycleEvent` | same strategy stream | One lifecycle transition; the fold of these *is* the status |

An `Experiment` captures inputs only. Its `Result` and `Decision` are **separate
append-only events** — corrections are new records, never edits.

### Why one event log instead of six tables

The hash chain inside an experiment stream spans `Experiment → Result →
Decision`; splitting those across tables would force a fragile cross-table
chain. A single, append-only, hash-chained log is the correct event-sourcing
primitive and makes `verify_chain` a linear scan. The six "kinds" are
discriminated rows with typed JSONB payloads.

## Immutability & tamper-evidence contract

- **No UPDATE / DELETE path exists.** Not at the store layer (there is no such
  method) and not at the database layer — Postgres triggers (`registry_reject_mutation`)
  raise on any `UPDATE`/`DELETE`.
- **Per-stream hash chain.** Every record stores `prev_hash` (the prior
  record's hash in its stream) and `record_hash = sha256(canonical(stream_id,
  seq, record_kind, record_id, payload, prev_hash, knowledge_time))`. Per-stream
  (not one global) chains mean appends to different streams never serialize.
- **`verify_chain(stream_id)`** recomputes every hash and confirms linkage and
  dense `seq`. On a simulated edit it returns the exact broken link
  (`broken_at_seq`, `broken_record_id`, `reason`).
- **Concurrency.** A racing append to the same stream collides on
  `UNIQUE(stream_id, seq)` and retries — no lost write, chain stays valid.
  Postgres uses a per-stream `pg_advisory_xact_lock`; the in-memory reference
  uses per-stream locks.
- **No secrets.** Every payload is scanned before persistence; keys that look
  like credentials (`*_secret`, `*token*`, `api_key`, …) are rejected. Env
  fingerprints reference versions/digests only.

## Reproducibility is a write precondition

> "A result that can't be reproduced doesn't exist." (CLAUDE.md §2.6)

`record_experiment` **rejects** any experiment whose `ReproMetadata` bundle is
incomplete. Required fields:

| Field | Meaning |
|-------|---------|
| `code_sha` + `dirty` | git commit and working-tree dirty flag |
| `spec_kind` + `spec_hash` | `dsl`/`code` and the content hash of the serialized spec |
| `params` | strategy/signal parameters |
| `data_snapshot` | PIT-native: `as_of` knowledge-time ceiling + `universe_spec` (from 01/02), so the exact dataset is reconstructible |
| `env` | python + key package versions, container image digest — never credentials |
| `seeds` | all RNG seeds |

A parent reference (`parent_experiment_id`) must resolve, guaranteeing lineage
traversal always terminates.

## Trial fingerprint (honest accounting)

```python
trial_fingerprint(meta) = sha256(canonical(
    spec_hash, params, data_snapshot.as_of, universe_spec
))
```

A *trial* is identified by **spec + params + dataset + universe** — and
**deliberately excludes** `code_sha`, `env`, `seeds`, and `dirty`. Re-running the
*same* trial on a different build or seed yields the **same** fingerprint, so the
later trials ledger (prompt 06) counts honestly and a silent re-run to
cherry-pick is detectable. `duplicate_experiment_ids(fp)` is the dedup view;
recording a duplicate is logged (never blocked — the log is append-only).

## Strategy lifecycle (event-sourced)

Status is never a mutable column; it is the fold over append-only
`LifecycleEvent`s.

```
create → candidate ── promote → promoted ── retire → retired
```

`record_lifecycle_event` validates the transition against the current fold
before persisting; illegal/out-of-order transitions (e.g. retire-before-promote)
are rejected. `current_state(strategy_id)` always equals the fold.

## Query API (failures and lineage are first-class)

```python
store = InMemoryRegistryStore()                 # or PostgresRegistryStore.connect(dsn)

exp_id = store.record_experiment(ExperimentInput(...))
store.record_result(ResultInput(experiment_id=exp_id, status="failure", metrics={...}))
store.record_decision(DecisionInput(experiment_id=exp_id, outcome="reject",
                                    deciding_component="gates.v1",
                                    component_version="1.0.0", rationale="fails calibration gate"))

store.get_experiment(exp_id)
store.experiments_by_author("agent.researcher")
store.experiments_by_niche("us_elections")
store.experiments_by_outcome("reject")          # failures are as queryable as successes
store.results_for(exp_id); store.decisions_for(exp_id)
store.experiment_lineage(exp_id)                # root → self
store.experiment_children(exp_id)
store.strategy_versions(sid); store.lifecycle_events(sid)
store.current_state(sid); store.strategies_by_state("promoted")
store.strategy_version_ancestry(sid, version=1) # resolves to originating experiment + ancestry
store.verify_chain(exp_id)
```

## Backends

| Backend | Purpose |
|---------|---------|
| `InMemoryRegistryStore` | Deterministic reference; tests and local dev |
| `PostgresRegistryStore` | Production spine; DB-level append-only enforcement |

Both share all write/query/verify logic (it lives on `RegistryStore`); backends
implement only the atomic per-stream append and the read primitives, so they
behave identically.

## Schema & migrations

See [`migrations/0001_init.sql`](migrations/0001_init.sql): the `registry_events`
table, per-stream chain constraints (`UNIQUE(stream_id, seq)`, genesis-unchained
check), query-path expression indexes, and the UPDATE/DELETE-rejecting triggers.

## Testing

```bash
uv run pytest tests/registry
# integration (real DB-level enforcement) — requires a reachable Postgres:
DELPHI_PG_DSN=postgresql://user:pass@localhost:5432/delphi uv run pytest tests/registry -m postgres
```

Component tests R1–R8 cover immutability (store + DB), tamper-evidence,
reproducibility round-trip + precondition, fingerprint/dedup, lifecycle
event-sourcing, failures-first-class, lineage, and concurrency.

## Out of scope (deferred)

The trials-ledger / multiple-testing math (06), the semantic recall layer
(10), gates (07), and agents (11). This
layer only **stores** experiments and computes the `trial_fingerprint` they will
consume.
