import pandas as pd

from src.model.features import TARGET


def apply_persistent_drift(
    batch_df: pd.DataFrame, batch_index: int, params: dict
) -> pd.DataFrame:
    p = params["persistent_drift"]
    if batch_index < p["start_batch"]:
        return batch_df

    out = batch_df.copy()
    for col, spec in p["columns"].items():
        shift = spec.get("shift", 0.0)
        scale = spec.get("scale", 1.0)
        if col == "MonthlyIncome":
            out[col] = out[col] * (1 + shift) * scale
        else:
            out[col] = (out[col] + shift) * scale
    return out


def apply_temporary_concept_drift(
    batch_df: pd.DataFrame, batch_index: int, params: dict
) -> pd.DataFrame:
    p = params["temporary_concept_drift"]
    if not (p["start_batch"] <= batch_index < p["end_batch"]):
        return batch_df

    delinquent_mask = batch_df[TARGET] == 1
    non_delinquent_mask = batch_df[TARGET] == 0
    if not delinquent_mask.any() or not non_delinquent_mask.any():
        return batch_df

    out = batch_df.copy()
    columns = p["columns"]
    blend_ratio = p["blend_ratio"]

    out[columns] = out[columns].astype(float)
    non_delinquent_centroid = out.loc[non_delinquent_mask, columns].mean()

    out.loc[delinquent_mask, columns] = (
        out.loc[delinquent_mask, columns] * (1 - blend_ratio)
        + non_delinquent_centroid * blend_ratio
    )
    return out


def inject_drift(batch_df: pd.DataFrame, batch_index: int, params: dict) -> pd.DataFrame:
    out = apply_persistent_drift(batch_df, batch_index, params)
    return apply_temporary_concept_drift(out, batch_index, params)
