"""Frozen holdout carve-out, training-pool-only imputation, and pretrain batch
construction. Order matters: carve_holdout() runs before fit_imputation_medians()
so holdout rows never influence the medians (leakage guard, build prompt section 2).
"""

import numpy as np
import pandas as pd

from src.model.features import IMPUTE_COLUMNS, TARGET


def carve_holdout(
    df: pd.DataFrame, holdout_frac: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train_pool and a frozen holdout, stratified on the target so
    the rare positive class is represented in both. Called before any
    imputation or scaling parameter is fit on the data.
    """
    rng = np.random.default_rng(seed)

    holdout_idx = []
    for label in df[TARGET].unique():
        class_idx = df.index[df[TARGET] == label].to_numpy().copy()
        rng.shuffle(class_idx)
        n_holdout = int(len(class_idx) * holdout_frac)
        holdout_idx.extend(class_idx[:n_holdout])

    holdout = df.loc[sorted(holdout_idx)].reset_index(drop=True)
    train_pool = df.drop(index=holdout_idx).reset_index(drop=True)
    return train_pool, holdout


def fit_imputation_medians(train_pool: pd.DataFrame) -> dict[str, float]:
    """Compute medians for IMPUTE_COLUMNS from the training pool only."""
    return {col: float(train_pool[col].median()) for col in IMPUTE_COLUMNS}


def apply_imputation(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Fill IMPUTE_COLUMNS using medians fit elsewhere (never fit on df itself
    if df might be the holdout or a live/simulated batch).
    """
    out = df.copy()
    for col, median in medians.items():
        out[col] = out[col].fillna(median)
    return out


def build_pretrain_batches(
    train_pool: pd.DataFrame, batch_size: int, seed: int = 42
) -> list[pd.DataFrame]:
    """Split the imputed training pool into fixed-size batches, shuffled once
    with a fixed seed so batch composition is reproducible. Last partial batch
    is dropped to keep batch size constant for drift injection math.
    """
    rng = np.random.default_rng(seed)
    shuffled = train_pool.sample(frac=1, random_state=rng.integers(0, 2**32 - 1))
    shuffled = shuffled.reset_index(drop=True)

    n_batches = len(shuffled) // batch_size
    batches = [
        shuffled.iloc[i * batch_size : (i + 1) * batch_size].reset_index(drop=True)
        for i in range(n_batches)
    ]
    return batches
