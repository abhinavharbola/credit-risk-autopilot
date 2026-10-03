import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.model.features import ALL_COLUMNS, TARGET
from src.utils.config import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_FILENAME = "cs-training.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
BATCHES_PATH = PROCESSED_DIR / "pretrain_batches.pkl"
TRAINING_POOL_PATH = PROCESSED_DIR / "training_pool.pkl"
HOLDOUT_PATH = PROCESSED_DIR / "holdout.pkl"

EXPECTED_POSITIVE_RATE = 0.067
POSITIVE_RATE_TOLERANCE = 0.01


def resolve_raw_file(raw_dir: Path = RAW_DIR) -> Path:
    expected = raw_dir / RAW_FILENAME
    if expected.exists():
        return expected

    candidates = sorted(raw_dir.glob("*training*.csv")) if raw_dir.exists() else []
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"no training csv found in {raw_dir}/ - this project does not "
        "download Give Me Some Credit automatically (it is a Kaggle "
        "competition dataset and the API returns 403 without accepting the "
        "competition rules first). Download it manually from "
        f"https://www.kaggle.com/c/GiveMeSomeCredit/data and place {RAW_FILENAME} "
        f"at {expected}"
    )


def load_raw(path: Path | None = None) -> pd.DataFrame:
    resolved_path = path or resolve_raw_file()
    df = pd.read_csv(resolved_path)
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols)

    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"raw dataset missing expected columns: {missing}")

    return df[ALL_COLUMNS]


def confirm_positive_rate(df: pd.DataFrame) -> float:
    rate = df[TARGET].mean()
    if abs(rate - EXPECTED_POSITIVE_RATE) > POSITIVE_RATE_TOLERANCE:
        raise ValueError(
            f"positive rate {rate:.4f} outside expected "
            f"{EXPECTED_POSITIVE_RATE} +/- {POSITIVE_RATE_TOLERANCE}"
        )
    return rate


def load_pickled(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run scripts/run_demo_loop.py once to prepare "
            "the processed data, or dvc pull it"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_processed_data() -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    return (
        load_pickled(BATCHES_PATH),
        load_pickled(TRAINING_POOL_PATH),
        load_pickled(HOLDOUT_PATH),
    )


if __name__ == "__main__":
    raw = load_raw()
    rate = confirm_positive_rate(raw)
    print(f"loaded {len(raw)} rows, positive rate {rate:.4f}")
