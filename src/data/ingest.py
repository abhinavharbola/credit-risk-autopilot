"""Loads Give Me Some Credit from data/raw/, confirms the positive rate, hands
off a raw dataframe. No imputation here, imputation medians must be fit on
the training pool only, after the holdout split (see split.py and the
leakage guard in the build prompt). Doing it here would leak holdout rows
into the medians.

This is a Kaggle COMPETITION dataset (not a plain public dataset), which
requires accepting the competition rules on kaggle.com and isn't reachable
via the dataset API even with valid credentials (that path returns a 403).
So this project does not download it automatically - place cs-training.csv
in data/raw/ yourself:

  1. https://www.kaggle.com/c/GiveMeSomeCredit/data
  2. accept the competition rules if prompted
  3. download the data (zip contains cs-training.csv, cs-test.csv,
     sampleEntry.csv, Data Dictionary.xls)
  4. extract cs-training.csv into data/raw/ (the other three files aren't
     used - cs-test.csv has no labels, so it's not useful for this project)
"""

from pathlib import Path

import pandas as pd

from src.model.features import ALL_COLUMNS, TARGET

RAW_DIR = Path("data/raw")
RAW_FILE = RAW_DIR / "cs-training.csv"

EXPECTED_POSITIVE_RATE = 0.067
POSITIVE_RATE_TOLERANCE = 0.01


def resolve_raw_file(raw_dir: Path = RAW_DIR) -> Path:
    """Finds the training file in data/raw/. Prefers the exact expected
    filename; falls back to any csv with "training" in its name in case the
    Kaggle zip was extracted under a slightly different name. Raises with
    clear setup instructions if nothing matches, rather than a bare
    FileNotFoundError.
    """
    if RAW_FILE.exists():
        return RAW_FILE

    candidates = sorted(raw_dir.glob("*training*.csv")) if raw_dir.exists() else []
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"no training csv found in {raw_dir}/ - this project does not "
        "download Give Me Some Credit automatically (it's a Kaggle "
        "competition dataset, not a plain dataset, and the API returns 403 "
        "without accepting the competition rules first). Download it "
        "manually from https://www.kaggle.com/c/GiveMeSomeCredit/data and "
        f"place cs-training.csv at {RAW_FILE}"
    )


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV, drop the unnamed index column Kaggle ships it with,
    and keep missing values as-is. Does not impute, does not fit anything.
    """
    resolved_path = path or resolve_raw_file()
    df = pd.read_csv(resolved_path)
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
    raw = load_raw()
    rate = confirm_positive_rate(raw)
    print(f"loaded {len(raw)} rows, positive rate {rate:.4f}")
