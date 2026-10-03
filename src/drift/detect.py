from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.model.features import FEATURES

DEFAULT_KS_PVALUE_THRESHOLD = 0.05
_DISTANCE_MARKERS = ("distance", "divergence", "psi")


class DriftFingerprintError(RuntimeError):
    pass


def compute_fingerprint(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    columns: list[str] | None = None,
    ks_pvalue_threshold: float = DEFAULT_KS_PVALUE_THRESHOLD,
) -> dict[str, Any]:
    cols = columns or FEATURES
    definition = DataDefinition(numerical_columns=list(cols))
    report = Report([DataDriftPreset(method="ks", threshold=ks_pvalue_threshold)])
    result = report.run(
        current_data=Dataset.from_pandas(current_df[cols], data_definition=definition),
        reference_data=Dataset.from_pandas(reference_df[cols], data_definition=definition),
    )
    return _reduce_to_fingerprint(result.dict())


def _column_is_drifted(value: float, method: str, threshold: float) -> bool:
    if any(marker in method.lower() for marker in _DISTANCE_MARKERS):
        return value > threshold
    return value < threshold


def _reduce_to_fingerprint(raw: dict[str, Any]) -> dict[str, Any]:
    drift_share = None
    column_scores: dict[str, float] = {}
    drifted_columns: list[str] = []

    for metric in raw.get("metrics", []):
        config = metric.get("config", {})
        metric_type = config.get("type", "")
        value = metric.get("value")

        if metric_type == "evidently:metric_v2:DriftedColumnsCount":
            if isinstance(value, dict) and "share" in value:
                drift_share = value["share"]

        elif metric_type == "evidently:metric_v2:ValueDrift":
            column = config.get("column")
            if column and isinstance(value, (int, float)):
                column_scores[column] = float(value)
                threshold = config.get("threshold", DEFAULT_KS_PVALUE_THRESHOLD)
                if _column_is_drifted(float(value), str(config.get("method", "")), threshold):
                    drifted_columns.append(column)

    return {
        "drift_share": drift_share,
        "column_drift_scores": column_scores,
        "drifted_columns": sorted(drifted_columns),
    }


def check_retrain_trigger(
    current_batch: pd.DataFrame,
    reference_df: pd.DataFrame,
    drift_share_threshold: float,
    ks_pvalue_threshold: float = DEFAULT_KS_PVALUE_THRESHOLD,
) -> tuple[bool, dict[str, Any]]:
    fingerprint = compute_fingerprint(
        current_batch, reference_df, ks_pvalue_threshold=ks_pvalue_threshold
    )
    if fingerprint["drift_share"] is None:
        raise DriftFingerprintError(
            "compute_fingerprint returned drift_share=None: Evidently's report "
            "shape has likely changed and _reduce_to_fingerprint no longer "
            "matches it"
        )
    return fingerprint["drift_share"] >= drift_share_threshold, fingerprint


def check_fingerprint_staleness(
    fingerprint_at_promotion: dict[str, Any],
    fingerprint_now: dict[str, Any],
    drift_share_delta_threshold: float,
    drifted_column_disagreement_threshold: float,
) -> bool:
    share_then = fingerprint_at_promotion.get("drift_share") or 0.0
    share_now = fingerprint_now.get("drift_share") or 0.0
    if abs(share_now - share_then) > drift_share_delta_threshold:
        return True

    drifted_then = fingerprint_at_promotion.get("drifted_columns")
    drifted_now = fingerprint_now.get("drifted_columns")
    if drifted_then is None or drifted_now is None:
        return False

    universe = set(fingerprint_at_promotion.get("column_drift_scores", {})) | set(
        fingerprint_now.get("column_drift_scores", {})
    )
    if not universe:
        return False

    disagreement = len(set(drifted_then) ^ set(drifted_now)) / len(universe)
    return disagreement > drifted_column_disagreement_threshold
