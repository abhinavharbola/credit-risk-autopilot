"""Bulk read/write helpers for predictions, labels, audit_log, champion_history,
and pipeline_state. Every function that touches N rows does it in one query
(4.7), never a Python loop of N round trips. Every function takes an open
SQLAlchemy Connection so callers control the transaction boundary.
"""

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection


def write_audit_log(conn: Connection, event_type: str, payload: dict[str, Any]) -> None:
    """Every governance decision goes through here: gate evaluations
    (promoted and rejected), promotions, rollbacks, drift checks, label
    releases, clock advances. Not just the promote branch.
    """
    conn.execute(
        text(
            "INSERT INTO audit_log (event_type, event_payload) "
            "VALUES (:event_type, CAST(:payload AS JSONB))"
        ),
        {"event_type": event_type, "payload": json.dumps(payload)},
    )


def insert_predictions_bulk(conn: Connection, rows: list[dict[str, Any]]) -> None:
    """rows: batch_id, model_alias, model_version, features (dict),
    predicted_prob, predicted_label. One batched INSERT, not a loop.
    """
    if not rows:
        return
    payload = [
        {
            "batch_id": r["batch_id"],
            "model_alias": r["model_alias"],
            "model_version": r["model_version"],
            "features": json.dumps(r["features"]),
            "predicted_prob": r["predicted_prob"],
            "predicted_label": r["predicted_label"],
        }
        for r in rows
    ]
    conn.execute(
        text(
            "INSERT INTO predictions "
            "(batch_id, model_alias, model_version, features, predicted_prob, predicted_label) "
            "VALUES (:batch_id, :model_alias, :model_version, CAST(:features AS JSONB), "
            ":predicted_prob, :predicted_label)"
        ),
        payload,
    )


def release_labels_bulk(
    conn: Connection, batch_id: int, id_to_label: dict[int, int]
) -> None:
    """Bulk-updates true_label for a whole batch's worth of predictions in a
    single UPDATE ... FROM VALUES, then writes ONE audit_log event describing
    the release (per-row updates are not individually logged, the release
    itself is the auditable event per schema note 6a).
    """
    if not id_to_label:
        return

    # ids/labels are cast to int before interpolation: this VALUES list can't be
    # parameterized as a bind list in plain SQLAlchemy text(), so we don't trust
    # them as raw strings even though they originate from our own pipeline.
    values_clause = ", ".join(
        f"({int(pid)}, {int(label)})" for pid, label in id_to_label.items()
    )
    conn.execute(
        text(
            f"""
            UPDATE predictions AS p
            SET true_label = v.label, label_released_at = now()
            FROM (VALUES {values_clause}) AS v(id, label)
            WHERE p.id = v.id
            """
        )
    )

    write_audit_log(
        conn,
        event_type="label_release",
        payload={"batch_id": batch_id, "n_labels_released": len(id_to_label)},
    )


def get_predictions_for_batch(
    conn: Connection, batch_id: int, model_alias: str | None = None
) -> list[dict[str, Any]]:
    query = "SELECT * FROM predictions WHERE batch_id = :batch_id"
    params: dict[str, Any] = {"batch_id": batch_id}
    if model_alias is not None:
        query += " AND model_alias = :model_alias"
        params["model_alias"] = model_alias
    result = conn.execute(text(query), params)
    return [dict(row._mapping) for row in result]


def insert_champion_history(conn: Connection, row: dict[str, Any]) -> int:
    """row: model_version, holdout_metrics (dict), window_metrics (dict),
    drift_fingerprint (dict). Returns the new champion_history id.
    """
    result = conn.execute(
        text(
            """
            INSERT INTO champion_history
                (model_version, holdout_metrics, window_metrics, drift_fingerprint)
            VALUES
                (:model_version, CAST(:holdout_metrics AS JSONB),
                 CAST(:window_metrics AS JSONB), CAST(:drift_fingerprint AS JSONB))
            RETURNING id
            """
        ),
        {
            "model_version": row["model_version"],
            "holdout_metrics": json.dumps(row["holdout_metrics"]),
            "window_metrics": json.dumps(row["window_metrics"]),
            "drift_fingerprint": json.dumps(row["drift_fingerprint"]),
        },
    )
    return result.scalar_one()


def get_champion_history(conn: Connection) -> list[dict[str, Any]]:
    result = conn.execute(text("SELECT * FROM champion_history ORDER BY promoted_at ASC"))
    return [dict(row._mapping) for row in result]


def get_latest_champion(conn: Connection) -> dict[str, Any] | None:
    result = conn.execute(
        text(
            "SELECT * FROM champion_history "
            "WHERE rolled_back_at IS NULL "
            "ORDER BY promoted_at DESC LIMIT 1"
        )
    )
    row = result.first()
    return dict(row._mapping) if row else None


def mark_reference_stale(conn: Connection, champion_history_id: int, stale: bool) -> None:
    """Flags whether the stored rollback reference (fingerprint + window
    metrics) has diverged from live traffic and can no longer be trusted for
    rollback comparisons (4.1).
    """
    conn.execute(
        text("UPDATE champion_history SET reference_stale = :stale WHERE id = :id"),
        {"stale": stale, "id": champion_history_id},
    )


def record_rollback(conn: Connection, champion_history_id: int, rolled_back_to_version: str) -> None:
    conn.execute(
        text(
            """
            UPDATE champion_history
            SET rolled_back_at = now(), rolled_back_to_version = :rolled_back_to_version
            WHERE id = :id
            """
        ),
        {"rolled_back_to_version": rolled_back_to_version, "id": champion_history_id},
    )


def get_audit_log(
    conn: Connection, event_type: str | None = None, limit: int = 500
) -> list[dict[str, Any]]:
    query = "SELECT * FROM audit_log"
    params: dict[str, Any] = {"limit": limit}
    if event_type:
        query += " WHERE event_type = :event_type"
        params["event_type"] = event_type
    query += " ORDER BY created_at DESC LIMIT :limit"
    result = conn.execute(text(query), params)
    return [dict(row._mapping) for row in result]


def get_pipeline_state(conn: Connection) -> dict[str, Any]:
    result = conn.execute(text("SELECT * FROM pipeline_state WHERE id = 1"))
    return dict(result.one()._mapping)


def advance_pipeline_state(conn: Connection, expected_version: int) -> bool:
    """The single writer for pipeline_state (4.6a). Advances current_batch by
    1 and bumps version, but only if the row's version still matches
    expected_version (optimistic concurrency). If another caller already
    advanced it, this is a no-op and returns False, so a duplicate or
    overlapping invocation (cron racing a manual run) never double-advances.
    """
    result = conn.execute(
        text(
            """
            UPDATE pipeline_state
            SET current_batch = current_batch + 1, version = version + 1, updated_at = now()
            WHERE id = 1 AND version = :expected_version
            RETURNING current_batch, version
            """
        ),
        {"expected_version": expected_version},
    )
    row = result.first()
    return row is not None
