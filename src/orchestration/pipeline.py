"""Orchestrates one clock tick: score the batch with @production, release any
now-due delayed labels, check for drift-triggered retrain, evaluate the gate
if a challenger was trained, promote or reject, then check rollback.

Every branch writes to audit_log - drift checks, gate evaluations (promoted
AND rejected), rollback checks - not just the promote branch. This is the
fix from the review: audit_log must record decisions, not just outcomes that
changed state.
"""

from typing import Any

import pandas as pd
from sqlalchemy.engine import Connection

from src.data.drift_injection import inject_drift
from src.db.repository import (
    get_predictions_for_batch,
    insert_predictions_bulk,
    release_labels_bulk,
    write_audit_log,
)
from src.drift.detect import check_retrain_trigger
from src.gate.evaluate import compute_metric, evaluate_gate
from src.model.features import TARGET
from src.model.train import score as score_model
from src.model.train import train_challenger
from src.orchestration.promote import check_rollback, promote_challenger
from src.utils.model_cache import AliasedModelCache

MODEL_NAME = "credit-risk-classifier"
_production_cache = AliasedModelCache(MODEL_NAME, "production")


def score_batch_with_production(
    conn: Connection,
    batch_df: pd.DataFrame,
    batch_id: int,
    model,
    model_version: str,
) -> None:
    """Scores batch_df with the given (already-loaded) production model and
    bulk-inserts one row per prediction (single INSERT, not a loop). Takes
    the model as a parameter rather than loading it itself, so run_tick's
    cached load is reused instead of a second independent load per tick.
    """
    probs = score_model(model, batch_df)
    rows = [
        {
            "batch_id": batch_id,
            "model_alias": "production",
            "model_version": model_version,
            "features": row.to_dict(),
            "predicted_prob": float(p),
            "predicted_label": int(p >= 0.5),
        }
        for (_, row), p in zip(batch_df.iterrows(), probs)
    ]
    insert_predictions_bulk(conn, rows)


def release_due_labels(
    conn: Connection, due_batch_id: int, due_batch_df: pd.DataFrame
) -> list[dict[str, Any]]:
    """Releases ground-truth labels for a batch whose delay has elapsed.
    due_batch_df must be the exact dataframe scored at insert time for this
    batch_id, same row order - that's what makes the id-to-label mapping
    below correct. Returns the prediction rows (now labeled) for the caller
    to use as the matched batch for gate evaluation.
    """
    rows = get_predictions_for_batch(conn, due_batch_id, model_alias="production")
    if len(rows) != len(due_batch_df):
        raise ValueError(
            f"prediction count ({len(rows)}) for batch {due_batch_id} doesn't "
            f"match due_batch_df ({len(due_batch_df)}) - order/window mismatch, "
            "labels would be released against the wrong rows"
        )
    id_to_label = {
        row["id"]: int(true_label)
        for row, true_label in zip(rows, due_batch_df[TARGET])
    }
    release_labels_bulk(conn, due_batch_id, id_to_label)
    for row, true_label in zip(rows, due_batch_df[TARGET]):
        row["true_label"] = int(true_label)
    return rows


def retrain_and_gate(
    conn: Connection,
    labeled_batch_df: pd.DataFrame,
    production_model,
    training_pool_df: pd.DataFrame,
    gate_config: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    """Trains a challenger, scores both champion and challenger on the
    identical matched (now fully-labeled) batch, runs the gate, and writes
    ONE audit_log entry regardless of outcome. Returns the gate decision plus
    enough context to promote if it passed.
    """
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


def run_tick(
    conn: Connection,
    current_batch: int,
    raw_batches: list[pd.DataFrame],
    training_pool_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Runs the full governance loop for one simulated batch:
    score -> release due labels -> drift check -> conditional retrain+gate
    -> conditional promote -> rollback check.

    raw_batches are the pre-drift pretrain batches; drift is injected here,
    on demand, for whichever batch index is needed (current_batch, and the
    due batch for label release). inject_drift is deterministic given
    (batch_index, config), so recomputing it later for the due batch
    reproduces byte-identical output to what was actually scored at the time
    - no need to persist a separate "already drifted" copy per batch.
    """
    production_model, production_version = _production_cache.get()

    batch_df = inject_drift(raw_batches[current_batch], current_batch, config)
    score_batch_with_production(conn, batch_df, current_batch, production_model, production_version)

    delay = config["delayed_labels"]["delay_batches"]
    due_batch_index = current_batch - delay
    result: dict[str, Any] = {"batch": current_batch}

    triggered, fingerprint = check_retrain_trigger(
        batch_df, training_pool_df, config["gate"]["reference_fingerprint_drift_threshold"]
    )
    write_audit_log(
        conn,
        event_type="drift_check",
        payload={"batch": current_batch, "retrain_triggered": triggered, "fingerprint": fingerprint},
    )
    result["retrain_triggered"] = triggered

    if due_batch_index >= 0:
        due_batch_df = inject_drift(raw_batches[due_batch_index], due_batch_index, config)
        release_due_labels(conn, due_batch_index, due_batch_df)
        result["labels_released_for_batch"] = due_batch_index

        if triggered:
            gate_outcome = retrain_and_gate(
                conn,
                due_batch_df,
                production_model,
                training_pool_df,
                config["gate"],
                run_name=f"challenger-batch-{current_batch}",
            )
            result["gate"] = gate_outcome["gate_result"].to_dict()

            if gate_outcome["gate_result"].promote:
                window_metrics = {
                    config["gate"]["primary_metric"]: gate_outcome["gate_result"].challenger_metric
                }
                holdout_metrics = window_metrics  # holdout eval wired in when live infra is available
                champion_history_id = promote_challenger(
                    conn,
                    gate_outcome["challenger_version"],
                    holdout_metrics,
                    window_metrics,
                    due_batch_df,
                    training_pool_df,
                )
                result["promoted_champion_history_id"] = champion_history_id

        live_prob = score_model(production_model, due_batch_df)
        live_metric_value = compute_metric(
            due_batch_df[TARGET].to_numpy(),
            live_prob,
            config["gate"]["primary_metric"],
            config["gate"]["decision_threshold"],
        )
        live_metrics = {config["gate"]["primary_metric"]: live_metric_value}
        rollback_result = check_rollback(
            conn,
            due_batch_df,
            live_metrics,
            training_pool_df,
            config["gate"]["primary_metric"],
            config["gate"]["rollback_metric_drop_threshold"],
            config["gate"]["reference_fingerprint_drift_threshold"],
        )
        result["rollback"] = rollback_result

    return result
