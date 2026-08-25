"""MLflow alias-based promotion and rollback, with the 4.1 fix built in: the
rollback reference stored at promotion time is the challenger's performance
on the drifted window it was gated against (window_metrics), plus a drift
fingerprint of that window, never the frozen holdout. The reference's own
fingerprint is checked for staleness before any rollback comparison is
trusted.
"""

from typing import Any

import mlflow
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
from src.gate.evaluate import bootstrap_metric_ci
from src.model.features import TARGET

MODEL_NAME = "credit-risk-classifier"


def promote_challenger(
    conn: Connection,
    challenger_version: str,
    holdout_metrics: dict[str, float],
    window_metrics: dict[str, float],
    window_df,
    training_pool_df,
) -> int:
    """Aliases challenger_version as @production. Stores window_metrics (not
    holdout metrics) as the rollback reference, alongside a drift fingerprint
    of the window computed against the canonical training reference.
    Writes an audit_log entry regardless - promotion is itself an event.
    """
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "production", challenger_version)

    fingerprint = compute_fingerprint(window_df, training_pool_df)

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
        event_type="promotion",
        payload={
            "champion_history_id": champion_history_id,
            "model_version": challenger_version,
            "holdout_metrics": holdout_metrics,
            "window_metrics": window_metrics,
        },
    )
    return champion_history_id


def find_previous_champion(
    history: list[dict[str, Any]], current_champion_history_id: int
) -> dict[str, Any] | None:
    """Pure selection logic, split out from check_rollback so it's directly
    testable: the most recent entry strictly before the current one that was
    never itself rolled back. Supports N-hop rollback (4.3) - this just has
    to find the right hop, not walk the whole chain by hand.
    """
    candidates = [
        h
        for h in history
        if h["id"] < current_champion_history_id and h["rolled_back_at"] is None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda h: h["id"])


def check_rollback(
    conn: Connection,
    live_batch_df,
    live_prob,
    live_metrics: dict[str, float],
    training_pool_df,
    metric_name: str,
    decision_threshold: float,
    drop_threshold: float,
    fingerprint_drift_threshold: float,
    bootstrap_resamples: int = 2000,
    significance_alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Two-step rollback check (4.1), plus a significance check (this pass):

      1. Has the stored reference window's fingerprint diverged materially
         from live traffic? If so, the reference is no longer apples-to-apples
         comparable. Rather than trust a stale comparison, flag it and
         suppress the rollback decision this cycle (conservative choice -
         a genuinely bad model stays live a bit longer, but nothing gets
         silently reverted against numbers we already know are unreliable).
      2. If the reference is still fresh, compare live_metrics against the
         stored window_metrics - a same-distribution comparison, never
         live-vs-pristine-holdout.
      3. If step 2's point estimate looks like a real drop, confirm it with
         a bootstrap CI on the live batch before actually rolling back - a
         raw point-estimate threshold on ~200 rows and a handful of
         positives swings on sampling noise alone (a real run showed
         rollback_triggered=True on drops that were pure noise, with
         nothing to actually roll back to yet). Requires the CI's upper
         (optimistic) bound to still fall below the drop threshold, not
         just the single point estimate - the same rigor the promotion
         gate already applies via McNemar/bootstrap on the other side.
    """
    current = get_latest_champion(conn)
    if current is None:
        return {"rollback_triggered": False, "reason": "no champion on record"}

    live_fingerprint = compute_fingerprint(live_batch_df, training_pool_df)
    is_stale = check_fingerprint_staleness(
        current["drift_fingerprint"], live_fingerprint, fingerprint_drift_threshold
    )

    if is_stale:
        mark_reference_stale(conn, current["id"], stale=True)
        write_audit_log(
            conn,
            event_type="rollback_check",
            payload={
                "champion_history_id": current["id"],
                "rollback_triggered": False,
                "reference_stale": True,
                "reason": (
                    "reference window fingerprint diverged from live traffic, "
                    "rollback check suppressed until re-baselined against a "
                    "fresh labeled window"
                ),
            },
        )
        return {"rollback_triggered": False, "reference_stale": True}

    mark_reference_stale(conn, current["id"], stale=False)

    stored_metric = current["window_metrics"].get(metric_name)
    live_metric = live_metrics.get(metric_name)
    drop = (
        stored_metric - live_metric
        if stored_metric is not None and live_metric is not None
        else None
    )
    point_estimate_flagged = drop is not None and drop >= drop_threshold

    rollback_triggered = False
    ci_lower = ci_upper = None
    if point_estimate_flagged:
        ci_lower, ci_upper = bootstrap_metric_ci(
            live_batch_df[TARGET].to_numpy(),
            live_prob,
            metric_name,
            decision_threshold,
            bootstrap_resamples,
            seed,
            significance_alpha,
        )
        rollback_triggered = ci_upper < (stored_metric - drop_threshold)

    write_audit_log(
        conn,
        event_type="rollback_check",
        payload={
            "champion_history_id": current["id"],
            "stored_metric": stored_metric,
            "live_metric": live_metric,
            "drop": drop,
            "point_estimate_flagged": point_estimate_flagged,
            "bootstrap_ci_lower": ci_lower,
            "bootstrap_ci_upper": ci_upper,
            "rollback_triggered": rollback_triggered,
            "reference_stale": False,
        },
    )

    if rollback_triggered:
        history = get_champion_history(conn)
        previous = find_previous_champion(history, current["id"])
        if previous is not None:
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(
                MODEL_NAME, "production", previous["model_version"]
            )
            record_rollback(conn, current["id"], previous["model_version"])
            write_audit_log(
                conn,
                event_type="rollback",
                payload={
                    "rolled_back_from": current["model_version"],
                    "rolled_back_to": previous["model_version"],
                    "drop": drop,
                },
            )

    return {"rollback_triggered": rollback_triggered, "reference_stale": False, "drop": drop}
