import numpy as np
import pandas as pd

from src.model.features import IMPUTE_COLUMNS, TARGET


def carve_holdout(
    df: pd.DataFrame, holdout_frac: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    holdout_idx = []
    for label in df[TARGET].unique():
        class_idx = df.index[df[TARGET] == label].to_numpy().copy()
        rng.shuffle(class_idx)
        n_holdout = int(len(class_idx) * holdout_frac)
        holdout_idx.extend(class_idx[:n_holdout])

    holdout = df.loc[sorted(holdout_idx)].reset_index(drop=True)
    remainder = df.drop(index=holdout_idx).reset_index(drop=True)
    return remainder, holdout


def carve_stream(
    df: pd.DataFrame, stream_frac: float = 0.5, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_pool, stream_pool = carve_holdout(df, holdout_frac=stream_frac, seed=seed)
    return base_pool, stream_pool


def fit_imputation_medians(base_pool: pd.DataFrame) -> dict[str, float]:
    return {col: float(base_pool[col].median()) for col in IMPUTE_COLUMNS}


def apply_imputation(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col, median in medians.items():
        out[col] = out[col].fillna(median)
    return out


def build_pretrain_batches(
    stream_pool: pd.DataFrame, batch_size: int, seed: int = 42
) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    shuffled = stream_pool.sample(frac=1, random_state=rng.integers(0, 2**32 - 1))
    shuffled = shuffled.reset_index(drop=True)

    n_batches = len(shuffled) // batch_size
    return [
        shuffled.iloc[i * batch_size : (i + 1) * batch_size].reset_index(drop=True)
        for i in range(n_batches)
    ]
