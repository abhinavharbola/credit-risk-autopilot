"""Recession-scenario drift injection.

Persistent feature drift: shift+scale DebtRatio and
RevolvingUtilizationOfUnsecuredLines upward, shrink MonthlyIncome, from a
configured batch onward, and it never turns off.

Temporary concept drift: blend delinquent-borrower rows toward the
non-delinquent centroid, active only within a fixed batch window, then reverts.

Both are seeded per-batch (seed, batch_index) so any batch's drift is
independently reproducible regardless of call order, and parameters are read
from config/drift_params.yaml, never hardcoded here.
"""

import numpy as np
import pandas as pd

from src.model.features import TARGET


def _batch_rng(seed: int, batch_index: int) -> np.random.Generator:
    return np.random.default_rng(seed + batch_index)


def apply_persistent_drift(
    batch_df: pd.DataFrame, batch_index: int, params: dict
) -> pd.DataFrame:
    """Applies the persistent shift/scale if batch_index >= start_batch.
    No-op otherwise. Deterministic, no randomness needed for this component.
    """
    p = params["persistent_drift"]
    if batch_index < p["start_batch"]:
        return batch_df

    out = batch_df.copy()
    for col, spec in p["columns"].items():
        shift = spec.get("shift", 0.0)
        scale = spec.get("scale", 1.0)
        if col == "MonthlyIncome":
            # shift is a fractional shrink for income, not additive
            out[col] = out[col] * (1 + shift) * scale
        else:
            out[col] = (out[col] + shift) * scale
    return out


def apply_temporary_concept_drift(
    batch_df: pd.DataFrame, batch_index: int, params: dict, seed: int
) -> pd.DataFrame:
    """Blends delinquent-borrower rows toward the non-delinquent centroid on the
    configured columns, only while start_batch <= batch_index < end_batch.
    Reverts automatically outside the window since it's simply not applied.
    """
    p = params["temporary_concept_drift"]
    if not (p["start_batch"] <= batch_index < p["end_batch"]):
        return batch_df

    out = batch_df.copy()
    columns = p["columns"]
    blend_ratio = p["blend_ratio"]

    non_delinquent_centroid = out.loc[out[TARGET] == 0, columns].mean()
    delinquent_mask = out[TARGET] == 1

    out.loc[delinquent_mask, columns] = (
        out.loc[delinquent_mask, columns] * (1 - blend_ratio)
        + non_delinquent_centroid * blend_ratio
    )
    return out


def inject_drift(batch_df: pd.DataFrame, batch_index: int, params: dict) -> pd.DataFrame:
    """Applies persistent drift then temporary concept drift, in that order,
    for a single batch. Both are governed entirely by params (seed, batch
    boundaries, shift/scale/blend values) so the exact drift shown to a
    reviewer is always reproducible by re-running with the same config.
    """
    seed = params["seed"]
    out = apply_persistent_drift(batch_df, batch_index, params)
    out = apply_temporary_concept_drift(out, batch_index, params, seed)
    return out
