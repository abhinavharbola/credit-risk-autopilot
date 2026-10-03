import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest import load_processed_data
from src.db.connection import get_connection
from src.orchestration.clock import claim_and_run_tick
from src.utils.config import load_pipeline_config
from src.utils.logging import configure_logging


def main() -> int:
    configure_logging()
    raw_batches, training_pool_df, holdout_df = load_processed_data()
    config = load_pipeline_config()

    with get_connection() as conn:
        result = claim_and_run_tick(conn, raw_batches, training_pool_df, config, holdout_df)

    if result is None:
        print("tick already claimed by another caller, no work done")
        return 0

    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
