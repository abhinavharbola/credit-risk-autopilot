"""Evidently-based drift detection, current 0.7.x API only
(`from evidently import Report`, `from evidently.presets import DataDriftPreset`).
Reused for both the retrain trigger and the 4.1 rollback-reference fingerprint
check, so both always compare against the same canonical reference (the
training pool) and produce fingerprints that are comparable across time.

_reduce_to_fingerprint() has been verified against a real
evidently==0.7.21 report.dict() output (see its docstring for the actual
observed schema). An earlier version of this function was written against
guessed key names and silently returned drift_share=None on every call,
because it matched on a metric_id key that doesn't exist in the real output
- see that function's docstring for the corrected field to match on.
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

    Verified against a live evidently==0.7.21 report.dict() output (not
    guessed from docs, which only cover the pre-1.0 API shape). Real shape:

        {
          "metrics": [
            {
              "id": "...",
              "metric_name": "DriftedColumnsCount(drift_share=0.5)",
              "config": {"type": "evidently:metric_v2:DriftedColumnsCount", ...},
              "value": {"count": 1.0, "share": 0.333...}
            },
            {
              "metric_name": "ValueDrift(column=DebtRatio,method=K-S p_value,...)",
              "config": {"type": "evidently:metric_v2:ValueDrift", "column": "DebtRatio", ...},
              "value": 4.57e-66
            },
            ...
          ]
        }

    config["type"] is the stable field to match on (versioned identifier
    string), not metric_name (a human-readable string whose exact format
    isn't a documented contract) and not metric_id (doesn't exist - this was
    the actual bug: matching on a key that was never present meant every
    check silently matched nothing and drift_share stayed None forever).

    Note: ValueDrift's default value is a K-S test p-value - LOWER means
    MORE drift, opposite of an intuitive "drift score" scale. This project
    only thresholds on drift_share (DriftedColumnsCount's share of columns
    that individually crossed their drift test), so that inversion doesn't
    affect the retrain trigger. The per-column p-values are still useful for
    the staleness check as a "has this column's relationship to the
    reference changed" signal - an absolute delta between two snapshots at
    different times is meaningful either way, regardless of which direction
    "more drift" points.
    """
    drift_share = None
    column_p_values: dict[str, float] = {}

    for metric in raw.get("metrics", []):
        metric_type = metric.get("config", {}).get("type", "")
        value = metric.get("value")

        if metric_type == "evidently:metric_v2:DriftedColumnsCount":
            if isinstance(value, dict) and "share" in value:
                drift_share = value["share"]

        elif metric_type == "evidently:metric_v2:ValueDrift":
            column = metric.get("config", {}).get("column")
            if column and isinstance(value, (int, float)):
                column_p_values[column] = float(value)

    return {"drift_share": drift_share, "column_drift_scores": column_p_values}


class DriftFingerprintError(RuntimeError):
    """Raised when compute_fingerprint's output doesn't have a usable
    drift_share. This used to fail silently (drift_share stayed None,
    check_retrain_trigger just treated that as "not triggered") - which is
    exactly how a schema mismatch against a future Evidently version could
    disable retraining for an entire run with no error anywhere. Governance
    decisions should fail loud, not fail open.
    """


def check_retrain_trigger(
    current_batch: pd.DataFrame,
    reference_df: pd.DataFrame,
    drift_share_threshold: float,
) -> tuple[bool, dict[str, Any]]:
    """Retrain trigger: has the current batch drifted meaningfully from the
    training reference? Reuses compute_fingerprint so the retrain trigger and
    the rollback-reference fingerprint check are always computed identically.

    Raises DriftFingerprintError if drift_share couldn't be extracted, rather
    than silently treating an unreadable report as "no drift" (see 0.7.21
    regression this project already hit once, pinned as a fixture in
    tests/test_drift_detect.py).
    """
    fingerprint = compute_fingerprint(current_batch, reference_df)
    if fingerprint["drift_share"] is None:
        raise DriftFingerprintError(
            "compute_fingerprint returned drift_share=None - Evidently's "
            "report shape has likely changed and _reduce_to_fingerprint no "
            "longer matches it. Refusing to silently treat this as "
            "'no drift'; see src/drift/detect.py docstring."
        )
    triggered = fingerprint["drift_share"] >= drift_share_threshold
    return triggered, fingerprint


def check_fingerprint_staleness(
    fingerprint_at_promotion: dict[str, Any],
    fingerprint_now: dict[str, Any],
    drift_share_delta_threshold: float,
    column_pvalue_delta_threshold: float,
) -> bool:
    """4.1 fix: the stored rollback reference can itself go stale (e.g. the
    temporary concept-drift window active at promotion time ends and
    reverts). Compares two fingerprints computed against the same canonical
    reference; if they've materially diverged, the stored window is no
    longer representative of live traffic.

    Takes two separate thresholds (previously a single shared value): the
    aggregate drift_share delta and the per-column p-value delta are
    different units answering different questions, and tuning one used to
    silently move the other.
    """
    share_then = fingerprint_at_promotion.get("drift_share") or 0.0
    share_now = fingerprint_now.get("drift_share") or 0.0
    if abs(share_now - share_then) > drift_share_delta_threshold:
        return True

    scores_then = fingerprint_at_promotion.get("column_drift_scores", {})
    scores_now = fingerprint_now.get("column_drift_scores", {})
    shared_cols = set(scores_then) & set(scores_now)
    if not shared_cols:
        return False

    max_col_delta = max(abs(scores_now[c] - scores_then[c]) for c in shared_cols)
    return max_col_delta > column_pvalue_delta_threshold
