-- Async forecast jobs for the published API (the App Runner 120s request cap
-- means long forecasts must run out-of-request; see api/jobs.py).
--
-- idempotency_key is UNIQUE but nullable: Postgres unique indexes admit any
-- number of NULLs, so keyless submissions never collide while a client-supplied
-- key maps to exactly one job (duplicate-spend protection).

CREATE TABLE IF NOT EXISTS forecast_jobs (
    job_id          TEXT        PRIMARY KEY,
    idempotency_key TEXT        UNIQUE,
    status          TEXT        NOT NULL
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    request         JSONB       NOT NULL,
    result          JSONB,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS forecast_jobs_status_idx
    ON forecast_jobs (status);
