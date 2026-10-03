from typing import Any

import pandas as pd
from sqlalchemy.engine import Connection

from src.data.drift_injection import inject_drift
from src.db.repository import (
    get_labeled_predictions,
    get_last_gate_batch,
    get_predictions_for_batch,
    insert_predictions_bulk,
    release_labels_bulk,
    write_audit_log,
)
from src.drift.detect import check_retrain_trigger
from src.gate.evaluate import compute_metric, evaluate_gate
from src.model.features import ALL_COLUMNS, FEATURES, TARGET
from src.model.train import score as score_model
from src.model.train import train_challenger
from src.orchestration.promote import (
    check_rollback,
    promote_challenger,
    reconcile_production_alias,
)
from src.utils.config import MODEL_NAME, PRODUCTION_ALIAS
from src.utils.model_cache import AliasedModelCache

BASE_POOL_SAMPLE_SEED = 123

_production_cache = AliasedModelCache(MODEL_NAME, PRODUCTION_ALIAS)


def score_batch_with_production(
    conn: Connection,
    batch_df: pd.DataFrame,
    batch_id: int,
    model,
    model_version: str,
    decision_threshold: float,
) -> None:
    probs = score_model(model, batch_df)
    feature_records = batch_df[FEATURES].to_dict(orient="records")
    rows = [
        {
            "batch_id": batch_id,
            "model_alias": PRODUCTION_ALIAS,
            "model_version": model_version,
            "features": features,
            "predicted_prob": float(p),
            "predicted_label": int(p >= decision_threshold),
        }
        for features, p in zip(feature_records, probs)
    ]
    insert_predictions_bulk(conn, rows)


def release_due_labels(
    conn: Connection, due_batch_id: int, due_batch_df: pd.DataFrame
) -> list[dict[str, Any]]:
    rows = get_predictions_for_batch(conn, due_batch_id, model_alias=PRODUCTION_ALIAS)
    if len(rows) != len(due_batch_df):
        raise ValueError(
            f"prediction count ({len(rows)}) for batch {due_batch_id} doesn't "
            f"match due_batch_df ({len(due_batch_df)}) - order/window mismatch, "
            "labels would be released against the wrong rows"
        )
    id_to_label = {
        row["id"]: int(true_label) for row, true_label in zip(rows, due_batch_df[TARGET])
    }
    release_labels_bulk(conn, due_batch_id, id_to_label)
    for row, true_label in zip(rows, due_batch_df[TARGET]):
        row["true_label"] = int(true_label)
    return rows


def build_expanded_training_pool(
    conn: Connection,
    base_training_pool_df: pd.DataFrame,
    gate_config: dict[str, Any],
    exclude_batch_ids: tuple[int, ...] = (),
) -> pd.DataFrame:
    sample_size = gate_config["retrain_base_pool_sample"]
    if len(base_training_pool_df) > sample_size:
        base_sample = base_training_pool_df.sample(
            n=sample_size, random_state=BASE_POOL_SAMPLE_SEED
        )
    else:
        base_sample = base_training_pool_df

    labeled_rows = get_labeled_predictions(
        conn,
        exclude_batch_ids=exclude_batch_ids,
        limit=gate_config["retrain_max_labeled_rows"],
    )
    if not labeled_rows:
        return base_sample.reset_index(drop=True)

    incremental_df = pd.DataFrame([row["features"] for row in labeled_rows])
    incremental_df[TARGET] = [row["true_label"] for row in labeled_rows]
    incremental_df = incremental_df[ALL_COLUMNS]

    return pd.concat([base_sample[ALL_COLUMNS], incremental_df], ignore_index=True)


def retrain_and_gate(
    conn: Connection,
    labeled_batch_df: pd.DataFrame,
    production_model,
    training_pool_df: pd.DataFrame,
    gate_config: dict[str, Any],
    run_name: str,
    batch_index: int | None = None,
) -> dict[str, Any]:
    challenger_run_id, challenger_version, challenger_model = train_challenger(
        training_pool_df, run_name=run_name
    )

    y_true = labeled_batch_df[TARGET].to_numpy()
    champion_prob = score_model(production_model, labeled_batch_df)
    challenger_prob = score_model(challenger_model, labeled_batch_df)

    gate_result = evaluate_gate(y_true, champion_prob, challenger_prob, gate_config)

    write_audit_log(
        conn,
        event_type="gate_evaluation",
        payload={
            "batch": batch_index,
            "challenger_run_id": challenger_run_id,
            "challenger_version": challenger_version,
            **gate_result.to_dict(),
        },
    )

    return {
        "gate_result": gate_result,
        "challenger_version": challenger_version,
        "challenger_model": challenger_model,
        "challenger_prob": challenger_prob,
        "champion_prob": champion_prob,
        "y_true": y_true,
    }


def _cooldown_elapsed(conn: Connection, current_batch: int, cooldown_batches: int) -> bool:
    last_gate_batch = get_last_gate_batch(conn)
    return last_gate_batch is None or current_batch - last_gate_batch >= cooldown_batches


def _retrain_stage(
    conn: Connection,
    current_batch: int,
    due_batch_index: int,
    due_batch_df: pd.DataFrame,
    production_model,
    training_pool_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    expanded_training_df = build_expanded_training_pool(
        conn, training_pool_df, gate_config, exclude_batch_ids=(due_batch_index,)
    )
    gate_outcome = retrain_and_gate(
        conn,
        due_batch_df,
        production_model,
        expanded_training_df,
        gate_config,
        run_name=f"challenger-batch-{current_batch}",
        batch_index=current_batch,
    )
    gate_result = gate_outcome["gate_result"]
    stage: dict[str, Any] = {"gate": gate_result.to_dict(), "promoted_champion_history_id": None}

    if not gate_result.promote:
        return stage

    metric_name = gate_config["primary_metric"]
    holdout_prob = score_model(gate_outcome["challenger_model"], holdout_df)
    holdout_metric_value = compute_metric(
        holdout_df[TARGET].to_numpy(),
        holdout_prob,
        metric_name,
        gate_config["decision_threshold"],
    )
    stage["promoted_champion_history_id"] = promote_challenger(
        conn,
        gate_outcome["challenger_version"],
        {metric_name: holdout_metric_value},
        {metric_name: gate_result.challenger_metric},
        due_batch_df,
        training_pool_df,
        fingerprint_kwargs={"ks_pvalue_threshold": gate_config["drift_ks_pvalue_threshold"]},
    )
    return stage


def _rollback_stage(
    conn: Connection,
    due_batch_df: pd.DataFrame,
    production_model,
    training_pool_df: pd.DataFrame,
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    metric_name = gate_config["primary_metric"]
    live_prob = score_model(production_model, due_batch_df)
    live_metric_value = compute_metric(
        due_batch_df[TARGET].to_numpy(),
        live_prob,
        metric_name,
        gate_config["decision_threshold"],
    )
    return check_rollback(
        conn,
        due_batch_df,
        live_prob,
        {metric_name: live_metric_value},
        training_pool_df,
        gate_config,
    )


def run_tick(
    conn: Connection,
    current_batch: int,
    raw_batches: list[pd.DataFrame],
    training_pool_df: pd.DataFrame,
    config: dict[str, Any],
    holdout_df: pd.DataFrame,
) -> dict[str, Any]:
    gate_config = config["gate"]

    reconcile_production_alias(conn)
    production_model, production_version = _production_cache.get()

    batch_df = inject_drift(raw_batches[current_batch], current_batch, config)
    score_batch_with_production(
        conn,
        batch_df,
        current_batch,
        production_model,
        production_version,
        gate_config["decision_threshold"],
    )

    due_batch_index = current_batch - config["delayed_labels"]["delay_batches"]
    has_due_batch = due_batch_index >= 0

    drift_detected, fingerprint = check_retrain_trigger(
        batch_df,
        training_pool_df,
        gate_config["retrain_drift_share_threshold"],
        ks_pvalue_threshold=gate_config["drift_ks_pvalue_threshold"],
    )
    retrain_triggered = (
        drift_detected
        and has_due_batch
        and _cooldown_elapsed(conn, current_batch, gate_config["retrain_cooldown_batches"])
    )
    write_audit_log(
        conn,
        event_type="drift_check",
        payload={
            "batch": current_batch,
            "drift_detected": drift_detected,
            "retrain_triggered": retrain_triggered,
            "fingerprint": fingerprint,
        },
    )

    result: dict[str, Any] = {
        "batch": current_batch,
        "drift_detected": drift_detected,
        "retrain_triggered": retrain_triggered,
    }
    if not has_due_batch:
        return result

    due_batch_df = inject_drift(raw_batches[due_batch_index], due_batch_index, config)
    release_due_labels(conn, due_batch_index, due_batch_df)
    result["labels_released_for_batch"] = due_batch_index

    promoted_champion_history_id = None
    if retrain_triggered:
        stage = _retrain_stage(
            conn,
            current_batch,
            due_batch_index,
            due_batch_df,
            production_model,
            training_pool_df,
            holdout_df,
            gate_config,
        )
        result["gate"] = stage["gate"]
        promoted_champion_history_id = stage["promoted_champion_history_id"]
        if promoted_champion_history_id is not None:
            result["promoted_champion_history_id"] = promoted_champion_history_id

    if promoted_champion_history_id is not None:
        skip_reason = (
            "skipped: this tick promoted a challenger, live window is "
            "the same window window_metrics was just computed from"
        )
        write_audit_log(
            conn,
            event_type="rollback_check",
            payload={
                "champion_history_id": promoted_champion_history_id,
                "rollback_triggered": False,
                "skipped": True,
                "reason": skip_reason,
            },
        )
        result["rollback"] = {
            "rollback_triggered": False,
            "rollback_executed": False,
            "skipped": True,
            "reason": skip_reason,
        }
    else:
        result["rollback"] = _rollback_stage(
            conn, due_batch_df, production_model, training_pool_df, gate_config
        )

    return result
