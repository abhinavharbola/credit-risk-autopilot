from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sqlalchemy.engine import Connection

from src.db.repository import (
    get_champion_history,
    get_latest_champion,
    insert_champion_history,
    mark_reference_stale,
    record_rollback,
    write_audit_log,
)
from src.drift.detect import check_fingerprint_staleness, compute_fingerprint
from src.gate.evaluate import bootstrap_delta_ci
from src.model.features import TARGET
from src.model.train import score
from src.utils.config import MODEL_NAME, PRODUCTION_ALIAS


def set_production_alias(version: str) -> None:
    mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, str(version))


def current_production_version() -> str | None:
    try:
        info = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except Exception:
        return None
    return str(info.version)


def load_model_version(version: str) -> Any:
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


def reconcile_production_alias(conn: Connection) -> bool:
    latest = get_latest_champion(conn)
    if latest is None:
        return False
    expected = str(latest["model_version"])
    actual = current_production_version()
    if actual == expected:
        return False
    set_production_alias(expected)
    write_audit_log(
        conn,
        "alias_reconciled",
        {"expected_version": expected, "found_version": actual},
    )
    return True


def promote_challenger(
    conn: Connection,
    challenger_version: str,
    holdout_metrics: dict[str, float],
    window_metrics: dict[str, float],
    window_df: pd.DataFrame,
    training_pool_df: pd.DataFrame,
    fingerprint_kwargs: dict[str, Any] | None = None,
) -> int:
    fingerprint = compute_fingerprint(window_df, training_pool_df, **(fingerprint_kwargs or {}))

    champion_history_id = insert_champion_history(
        conn,
        {
            "model_version": challenger_version,
            "holdout_metrics": holdout_metrics,
            "window_metrics": window_metrics,
            "drift_fingerprint": fingerprint,
        },
    )
    write_audit_log(
        conn,
        "promotion",
        {
            "model_version": challenger_version,
            "champion_history_id": champion_history_id,
            "window_metrics": window_metrics,
        },
    )
    set_production_alias(challenger_version)
    return champion_history_id


def find_previous_champion(
    champion_history: list[dict[str, Any]], current_champion_history_id: int
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in champion_history
        if row["id"] < current_champion_history_id and row["rolled_back_at"] is None
    ]
    return max(candidates, key=lambda r: r["id"]) if candidates else None


def check_rollback(
    conn: Connection,
    live_batch_df: pd.DataFrame,
    live_prob,
    live_metrics: dict[str, float],
    training_pool_df: pd.DataFrame,
    gate_config: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    metric_name = gate_config["primary_metric"]
    current = get_latest_champion(conn)
    if current is None:
        return {
            "rollback_triggered": False,
            "rollback_executed": False,
            "degradation_flagged": False,
            "reference_stale": False,
            "reason": "no champion recorded",
        }

    live_fingerprint = compute_fingerprint(
        live_batch_df,
        training_pool_df,
        ks_pvalue_threshold=gate_config["drift_ks_pvalue_threshold"],
    )
    is_stale = check_fingerprint_staleness(
        current["drift_fingerprint"],
        live_fingerprint,
        drift_share_delta_threshold=gate_config["staleness_drift_share_delta_threshold"],
        drifted_column_disagreement_threshold=gate_config[
            "staleness_drifted_column_disagreement_threshold"
        ],
    )
    mark_reference_stale(conn, current["id"], is_stale)

    if is_stale:
        result = {
            "rollback_triggered": False,
            "rollback_executed": False,
            "degradation_flagged": False,
            "reference_stale": True,
            "reason": (
                "drift fingerprint at promotion no longer matches the live regime, "
                "so the stored reference metric is not comparable"
            ),
        }
        write_audit_log(conn, "rollback_check", {**result, "champion_history_id": current["id"]})
        return result

    stored_metric = current["window_metrics"].get(metric_name)
    live_metric = live_metrics.get(metric_name)
    drop = None if stored_metric is None or live_metric is None else stored_metric - live_metric
    degradation_flagged = drop is not None and drop >= gate_config["rollback_metric_drop_threshold"]

    previous = None
    ci_lower = ci_upper = None
    rollback_triggered = False
    if degradation_flagged:
        previous = find_previous_champion(get_champion_history(conn), current["id"])
        if previous is not None:
            previous_prob = score(load_model_version(previous["model_version"]), live_batch_df)
            ci_lower, ci_upper = bootstrap_delta_ci(
                live_batch_df[TARGET].to_numpy(),
                live_prob,
                previous_prob,
                metric_name,
                gate_config["decision_threshold"],
                gate_config["bootstrap_resamples"],
                seed,
                gate_config["significance_alpha"],
            )
            rollback_triggered = ci_lower > 0

    if not degradation_flagged:
        reason = "live metric within threshold of the reference metric"
    elif previous is None:
        reason = "degradation flagged but no earlier champion is available to roll back to"
    elif rollback_triggered:
        reason = "earlier champion significantly outperforms current champion on the live batch"
    else:
        reason = "degradation flagged but earlier champion is not significantly better on the live batch"

    payload = {
        "champion_history_id": current["id"],
        "stored_metric": stored_metric,
        "live_metric": live_metric,
        "drop": drop,
        "degradation_flagged": degradation_flagged,
        "candidate_version": previous["model_version"] if previous else None,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "rollback_triggered": rollback_triggered,
        "reference_stale": False,
        "reason": reason,
    }
    write_audit_log(conn, "rollback_check", payload)

    if rollback_triggered:
        record_rollback(conn, current["id"], previous["model_version"])
        write_audit_log(
            conn,
            "rollback",
            {
                "rolled_back_from": current["model_version"],
                "rolled_back_to": previous["model_version"],
                "champion_history_id": current["id"],
                "drop": drop,
            },
        )
        set_production_alias(previous["model_version"])

    return {
        "rollback_triggered": rollback_triggered,
        "rollback_executed": rollback_triggered,
        "degradation_flagged": degradation_flagged,
        "reference_stale": False,
        "reason": reason,
        "drop": drop,
    }
