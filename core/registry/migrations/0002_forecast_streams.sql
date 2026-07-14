-- Forecast taxonomy (DELPHI §3/§5): widen the append-only event log to carry
-- the question stream kind and its record kinds (question / evidence_set /
-- forecast / resolution) alongside the existing experiment and strategy streams.
--
-- The 0001 CHECK constraints were declared inline on their columns, so Postgres
-- named them deterministically (<table>_<column>_check). Drop-and-recreate them
-- with the widened value sets. This is additive: existing rows still satisfy the
-- new constraints, and the hash-chain / append-only triggers are untouched.

ALTER TABLE registry_events DROP CONSTRAINT IF EXISTS registry_events_stream_kind_check;
ALTER TABLE registry_events ADD CONSTRAINT registry_events_stream_kind_check
    CHECK (stream_kind IN ('experiment', 'strategy', 'question'));

ALTER TABLE registry_events DROP CONSTRAINT IF EXISTS registry_events_record_kind_check;
ALTER TABLE registry_events ADD CONSTRAINT registry_events_record_kind_check
    CHECK (record_kind IN (
        'experiment', 'result', 'decision',
        'strategy', 'strategy_version', 'lifecycle_event',
        'question', 'evidence_set', 'forecast', 'resolution'
    ));

-- Query-path index for per-domain calibration lookups (CLAUDE.md §2.3).
CREATE INDEX IF NOT EXISTS registry_events_domain_idx
    ON registry_events ((payload ->> 'domain'));
