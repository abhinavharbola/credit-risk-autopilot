"""Evidently-based drift detection, current 0.7.x API only
(`from evidently import Report`, `from evidently.presets import DataDriftPreset`).
Reused for both the retrain trigger and the 4.1 rollback-reference fingerprint
check, so both always compare against the same canonical reference (the
training pool) and produce fingerprints that are comparable across time.

NOTE: Evidently's raw report dict is deep and has shifted shape across minor
versions. _reduce_to_fingerprint() below has not been run against a live
Evidently install in this environment (no network here) - verify the exact
metric_id / value keys against the installed 0.7.x version before relying on
this in a real run, per the build prompt's own caution about not trusting
tutorial code blindly.
"""

from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.model.features import FEATURES


def compute_fingerprint(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Runs DataDriftPreset comparing current_df to reference_df, reduces the
    result to a compact JSON-serializable summary: overall drift share plus a
    per-column drift score. Always call with the SAME reference_df (the
    training pool) so fingerprints computed at different times stay
    apples-to-apples comparable.
    """
    cols = columns or FEATURES
    report = Report([DataDriftPreset()])
    result = report.run(current_data=current_df[cols], reference_data=reference_df[cols])
    return _reduce_to_fingerprint(result.dict())


def _reduce_to_fingerprint(raw: dict[str, Any]) -> dict[str, Any]:
    """Pulls just the two numbers this project's rollback/retrain logic needs
    out of Evidently's raw dict, so downstream code never depends on
    Evidently's internal shape beyond this one function.
    """
    drift_share = None
    column_scores: dict[str, float] = {}

    for metric in raw.get("metrics", []):
        metric_id = str(metric.get("metric_id", "") or metric.get("metric", ""))
        value = metric.get("value")

        if "drifted_columns" in metric_id.lower() or "dataset_drift" in metric_id.lower():
            if isinstance(value, dict) and "share" in value:
                drift_share = value["share"]
            elif isinstance(value, (int, float)):
                drift_share = float(value)

        if "valuedrift" in metric_id.lower() or "driftscore" in metric_id.lower():
            column = metric.get("column_name") or metric.get("parameters", {}).get("column_name")
            if column and isinstance(value, (int, float)):
                column_scores[column] = float(value)

    return {"drift_share": drift_share, "column_drift_scores": column_scores}


def check_retrain_trigger(
    current_batch: pd.DataFrame,
    reference_df: pd.DataFrame,
    drift_share_threshold: float,
) -> tuple[bool, dict[str, Any]]:
    """Retrain trigger: has the current batch drifted meaningfully from the
    training reference? Reuses compute_fingerprint so the retrain trigger and
    the rollback-reference fingerprint check are always computed identically.
    """
    fingerprint = compute_fingerprint(current_batch, reference_df)
    triggered = (
        fingerprint["drift_share"] is not None
        and fingerprint["drift_share"] >= drift_share_threshold
    )
    return triggered, fingerprint


def check_fingerprint_staleness(
    fingerprint_at_promotion: dict[str, Any],
    fingerprint_now: dict[str, Any],
    threshold: float,
) -> bool:
    """4.1 fix: the stored rollback reference can itself go stale (e.g. the
    temporary concept-drift window active at promotion time ends and
    reverts). Compares two fingerprints computed against the same canonical
    reference; if they've materially diverged, the stored window is no
    longer representative of live traffic.
    """
    share_then = fingerprint_at_promotion.get("drift_share") or 0.0
    share_now = fingerprint_now.get("drift_share") or 0.0
    if abs(share_now - share_then) > threshold:
        return True

    scores_then = fingerprint_at_promotion.get("column_drift_scores", {})
    scores_now = fingerprint_now.get("column_drift_scores", {})
    shared_cols = set(scores_then) & set(scores_now)
    if not shared_cols:
        return False

    max_col_delta = max(abs(scores_now[c] - scores_then[c]) for c in shared_cols)
    return max_col_delta > threshold
