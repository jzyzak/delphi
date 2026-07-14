-- Global append-only trials ledger (CLAUDE.md §2.4).
-- One row per committed guarded-set trial. COUNT(*) is the firm-wide debited
-- count that admission control (budget_reservations, migration 0001) draws
-- against. Append-only by construction: no UPDATE/DELETE is ever issued, so the
-- ledger only ever grows. This makes method-overfitting via silent re-runs
-- visible instead of free — the ledger only ever gets stricter, never looser.

CREATE TABLE IF NOT EXISTS trials_ledger (
    trial_id    BIGSERIAL   PRIMARY KEY,
    grant_id    TEXT        NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS trials_ledger_grant_idx
    ON trials_ledger (grant_id);
