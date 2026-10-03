CREATE TABLE IF NOT EXISTS pipeline_state (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    current_batch   INTEGER NOT NULL DEFAULT 0,
    version         INTEGER NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT single_row CHECK (id = 1)
);

CREATE TABLE IF NOT EXISTS predictions (
    id                  BIGSERIAL PRIMARY KEY,
    batch_id            INTEGER NOT NULL,
    model_alias         TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    features            JSONB NOT NULL,
    predicted_prob      DOUBLE PRECISION NOT NULL,
    predicted_label     INTEGER NOT NULL,
    true_label          INTEGER,
    label_released_at   TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_batch ON predictions (batch_id);
CREATE INDEX IF NOT EXISTS idx_predictions_alias ON predictions (model_alias);

CREATE TABLE IF NOT EXISTS champion_history (
    id                       BIGSERIAL PRIMARY KEY,
    model_version            TEXT NOT NULL,
    promoted_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    holdout_metrics          JSONB NOT NULL,
    window_metrics           JSONB NOT NULL,
    drift_fingerprint        JSONB NOT NULL,
    reference_stale          BOOLEAN NOT NULL DEFAULT FALSE,
    rolled_back_at           TIMESTAMPTZ,
    rolled_back_to_version   TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL PRIMARY KEY,
    event_type      TEXT NOT NULL,
    event_payload   JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_type ON audit_log (event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log (created_at);

INSERT INTO pipeline_state (id, current_batch, version)
VALUES (1, 0, 0)
ON CONFLICT (id) DO NOTHING;
