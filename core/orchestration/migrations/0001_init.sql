-- Orchestration run state and global trials-budget reservations (prompt 16).
-- Reservations are admission control on top of harness/stats trials_ledger (06).

CREATE TABLE IF NOT EXISTS budget_reservations (
    grant_id    TEXT        PRIMARY KEY,
    n           INTEGER     NOT NULL CHECK (n > 0),
    status      TEXT        NOT NULL CHECK (status IN ('reserved', 'committed', 'released')),
    reserved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS budget_reservations_status_idx
    ON budget_reservations (status);

CREATE TABLE IF NOT EXISTS orchestration_runs (
    step_id       TEXT        PRIMARY KEY,
    loop_name     TEXT        NOT NULL,
    tick_at       TIMESTAMPTZ NOT NULL,
    status        TEXT        NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS orchestration_runs_loop_tick_idx
    ON orchestration_runs (loop_name, tick_at);
