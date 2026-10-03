import json
from typing import Any, Sequence

from sqlalchemy import text
from sqlalchemy.engine import Connection


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _dumps(value: Any) -> str:
    return json.dumps(value, default=_to_builtin)


def write_audit_log(conn: Connection, event_type: str, payload: dict[str, Any]) -> None:
    conn.execute(
        text(
            "INSERT INTO audit_log (event_type, event_payload) "
            "VALUES (:event_type, CAST(:payload AS JSONB))"
        ),
        {"event_type": event_type, "payload": _dumps(payload)},
    )


def insert_predictions_bulk(conn: Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    payload = [
        {
            "batch_id": r["batch_id"],
            "model_alias": r["model_alias"],
            "model_version": r["model_version"],
            "features": _dumps(r["features"]),
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


def release_labels_bulk(conn: Connection, batch_id: int, id_to_label: dict[int, int]) -> None:
    if not id_to_label:
        return

    ids = list(id_to_label.keys())
    labels = list(id_to_label.values())

    conn.execute(
        text(
            """
            UPDATE predictions AS p
            SET true_label = v.label, label_released_at = now()
            FROM (
                SELECT * FROM unnest(
                    CAST(:ids AS bigint[]), CAST(:labels AS integer[])
                ) AS v(id, label)
            ) AS v
            WHERE p.id = v.id
            """
        ),
        {"ids": ids, "labels": labels},
    )

    write_audit_log(
        conn,
        event_type="label_release",
        payload={"batch_id": batch_id, "n_labels_released": len(id_to_label)},
    )


def get_labeled_predictions(
    conn: Connection,
    model_alias: str = "production",
    exclude_batch_ids: Sequence[int] = (),
    limit: int | None = None,
) -> list[dict[str, Any]]:
    query = (
        "SELECT features, true_label FROM predictions "
        "WHERE true_label IS NOT NULL AND model_alias = :model_alias "
        "AND batch_id <> ALL(CAST(:excluded AS integer[])) "
        "ORDER BY batch_id DESC, id DESC"
    )
    params: dict[str, Any] = {
        "model_alias": model_alias,
        "excluded": [int(b) for b in exclude_batch_ids],
    }
    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = int(limit)
    result = conn.execute(text(query), params)
    return [dict(row._mapping) for row in result]


def get_predictions_for_batch(
    conn: Connection, batch_id: int, model_alias: str | None = None
) -> list[dict[str, Any]]:
    query = "SELECT * FROM predictions WHERE batch_id = :batch_id"
    params: dict[str, Any] = {"batch_id": batch_id}
    if model_alias is not None:
        query += " AND model_alias = :model_alias"
        params["model_alias"] = model_alias
    query += " ORDER BY id"
    result = conn.execute(text(query), params)
    return [dict(row._mapping) for row in result]


def insert_champion_history(conn: Connection, row: dict[str, Any]) -> int:
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
            "model_version": str(row["model_version"]),
            "holdout_metrics": _dumps(row["holdout_metrics"]),
            "window_metrics": _dumps(row["window_metrics"]),
            "drift_fingerprint": _dumps(row["drift_fingerprint"]),
        },
    )
    return result.scalar_one()


def get_champion_history(conn: Connection) -> list[dict[str, Any]]:
    result = conn.execute(text("SELECT * FROM champion_history ORDER BY id ASC"))
    return [dict(row._mapping) for row in result]


def get_latest_champion(conn: Connection) -> dict[str, Any] | None:
    result = conn.execute(
        text(
            "SELECT * FROM champion_history "
            "WHERE rolled_back_at IS NULL "
            "ORDER BY id DESC LIMIT 1"
        )
    )
    row = result.first()
    return dict(row._mapping) if row else None


def mark_reference_stale(conn: Connection, champion_history_id: int, stale: bool) -> None:
    conn.execute(
        text("UPDATE champion_history SET reference_stale = :stale WHERE id = :id"),
        {"stale": stale, "id": champion_history_id},
    )


def record_rollback(
    conn: Connection, champion_history_id: int, rolled_back_to_version: str
) -> None:
    conn.execute(
        text(
            """
            UPDATE champion_history
            SET rolled_back_at = now(), rolled_back_to_version = :rolled_back_to_version
            WHERE id = :id
            """
        ),
        {"rolled_back_to_version": str(rolled_back_to_version), "id": champion_history_id},
    )


def get_audit_log(
    conn: Connection,
    event_type: str | None = None,
    limit: int = 500,
    payload_equals: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM audit_log WHERE TRUE"
    params: dict[str, Any] = {"limit": limit}
    if event_type:
        query += " AND event_type = :event_type"
        params["event_type"] = event_type
    for i, (key, value) in enumerate((payload_equals or {}).items()):
        query += f" AND event_payload ->> :pk{i} = :pv{i}"
        params[f"pk{i}"] = key
        params[f"pv{i}"] = value
    query += " ORDER BY created_at DESC, id DESC LIMIT :limit"
    result = conn.execute(text(query), params)
    return [dict(row._mapping) for row in result]


def get_last_gate_batch(conn: Connection) -> int | None:
    result = conn.execute(
        text(
            "SELECT (event_payload ->> 'batch')::integer FROM audit_log "
            "WHERE event_type = 'gate_evaluation' AND event_payload ->> 'batch' IS NOT NULL "
            "ORDER BY id DESC LIMIT 1"
        )
    )
    row = result.first()
    return int(row[0]) if row is not None and row[0] is not None else None


def get_pipeline_state(conn: Connection) -> dict[str, Any]:
    result = conn.execute(text("SELECT * FROM pipeline_state WHERE id = 1"))
    return dict(result.one()._mapping)


def advance_pipeline_state(conn: Connection, expected_version: int) -> bool:
    result = conn.execute(
        text(
            """
            UPDATE pipeline_state
            SET current_batch = current_batch + 1, version = version + 1, updated_at = now()
            WHERE id = 1 AND version = :expected_version
              AND id IN (SELECT id FROM pipeline_state WHERE id = 1 FOR UPDATE SKIP LOCKED)
            RETURNING current_batch, version
            """
        ),
        {"expected_version": expected_version},
    )
    return result.first() is not None
