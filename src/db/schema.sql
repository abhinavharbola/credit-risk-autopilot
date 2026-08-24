-- pipeline_state: single source of truth for the orchestration clock.
-- version column enables optimistic concurrency so cron and manual advance
-- can never double-advance (4.6a).
CREATE TABLE IF NOT EXISTS pipeline_state (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    current_batch   INTEGER NOT NULL DEFAULT 0,
    version         INTEGER NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT single_row CHECK (id = 1)
);

-- predictions: one row per scored prediction. label lands here as a nullable
-- column (decision made explicitly in 6a) rather than a separate table.
CREATE TABLE IF NOT EXISTS predictions (
    id                  BIGSERIAL PRIMARY KEY,
    batch_id            INTEGER NOT NULL,
    model_alias         TEXT NOT NULL,       -- 'production' or 'challenger'
    model_version       TEXT NOT NULL,       -- MLflow registered model version
    features            JSONB NOT NULL,
    predicted_prob      DOUBLE PRECISION NOT NULL,
    predicted_label     INTEGER NOT NULL,
    true_label          INTEGER,             -- NULL until delayed label release
    label_released_at   TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_batch ON predictions (batch_id);
CREATE INDEX IF NOT EXISTS idx_predictions_alias ON predictions (model_alias);

-- champion_history: N-hop lineage, not just current/previous (4.3).
-- fingerprint stores the Evidently drift report json used for the
-- rollback reference staleness check (4.1).
CREATE TABLE IF NOT EXISTS champion_history (
    id                      BIGSERIAL PRIMARY KEY,
    model_version           TEXT NOT NULL,
    promoted_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    holdout_metrics         JSONB NOT NULL,
    window_metrics          JSONB NOT NULL,   -- metrics on the drifted window it was gated against
    drift_fingerprint        JSONB NOT NULL,   -- Evidently report at promotion time
    reference_stale          BOOLEAN NOT NULL DEFAULT FALSE,
    rolled_back_at           TIMESTAMPTZ,
    rolled_back_to_version   TEXT
);

-- audit_log: every governance decision, not just promotions.
-- includes rejected challengers, drift checks, rollback checks, and
-- label-release events on predictions (6a).
CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL PRIMARY KEY,
    event_type      TEXT NOT NULL,   -- 'gate_evaluation', 'promotion', 'rollback',
                                      -- 'label_release', 'drift_check', 'clock_advance'
    event_payload   JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_type ON audit_log (event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log (created_at);

INSERT INTO pipeline_state (id, current_batch, version)
VALUES (1, 0, 0)
ON CONFLICT (id) DO NOTHING;
