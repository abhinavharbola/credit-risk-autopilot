"""Download Give Me Some Credit, confirm the positive rate, hand off a raw
dataframe. No imputation here, imputation medians must be fit on the training
pool only, after the holdout split (see split.py and the leakage guard in the
build prompt). Doing it here would leak holdout rows into the medians.
"""

from pathlib import Path

import pandas as pd

from src.model.features import ALL_COLUMNS, TARGET

RAW_DIR = Path("data/raw")
RAW_FILE = RAW_DIR / "give_me_some_credit.csv"

EXPECTED_POSITIVE_RATE = 0.067
POSITIVE_RATE_TOLERANCE = 0.01


def download_dataset(dest: Path = RAW_FILE) -> Path:
    """Fetch the Kaggle dataset via kagglehub. Requires KAGGLE_USERNAME /
    KAGGLE_KEY (or a configured kaggle.json) in the environment. No-ops if the
    file already exists locally.
    """
    if dest.exists():
        return dest

    import kagglehub

    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = kagglehub.dataset_download("GiveMeSomeCredit/cs-training")
    source_csv = next(Path(downloaded_path).glob("*.csv"))
    dest.write_bytes(source_csv.read_bytes())
    return dest


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Load the raw CSV, drop the unnamed index column Kaggle ships it with,
    and keep missing values as-is. Does not impute, does not fit anything.
    """
    df = pd.read_csv(path)
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols)

    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"raw dataset missing expected columns: {missing}")

    return df[ALL_COLUMNS]


def confirm_positive_rate(df: pd.DataFrame) -> float:
    """Confirm the ~6.7% positive rate holds. Raises if it drifts outside
    tolerance, since a silent change here would invalidate the whole
    class-imbalance narrative the project is built around.
    """
    rate = df[TARGET].mean()
    if abs(rate - EXPECTED_POSITIVE_RATE) > POSITIVE_RATE_TOLERANCE:
        raise ValueError(
            f"positive rate {rate:.4f} outside expected "
            f"{EXPECTED_POSITIVE_RATE} +/- {POSITIVE_RATE_TOLERANCE}"
        )
    return rate


if __name__ == "__main__":
    path = download_dataset()
    raw = load_raw(path)
    rate = confirm_positive_rate(raw)
    print(f"loaded {len(raw)} rows, positive rate {rate:.4f}")
