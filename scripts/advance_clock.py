"""Entrypoint both a human and the cron workflow call. Thin CLI wrapper: load
data/config, open a connection, delegate to
src.orchestration.clock.claim_and_run_tick - the actual single-writer
implementation (4.6a). This script has no governance logic of its own.
"""

import pickle
import sys
from pathlib import Path

# Same issue as run_demo_loop.py: running this file directly only puts
# scripts/ on sys.path, not the repo root. Insert it before any src import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.connection import get_connection
from src.orchestration.clock import claim_and_run_tick
from src.utils.config import load_yaml

RAW_BATCHES_PATH = Path("data/processed/pretrain_batches.pkl")
TRAINING_POOL_PATH = Path("data/processed/training_pool.pkl")


def load_pickled(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run the data prep step first "
            "(src/data/ingest.py + src/data/split.py) to produce it"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def build_config() -> dict:
    drift_params = load_yaml("config/drift_params.yaml")
    gate_config = load_yaml("config/gate_config.yaml")
    return {**drift_params, "gate": gate_config}


def main() -> int:
    raw_batches = load_pickled(RAW_BATCHES_PATH)
    training_pool_df = load_pickled(TRAINING_POOL_PATH)
    config = build_config()

    with get_connection() as conn:
        result = claim_and_run_tick(conn, raw_batches, training_pool_df, config)

    if result is None:
        print("tick already claimed by another caller, no work done")
        return 0

    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
