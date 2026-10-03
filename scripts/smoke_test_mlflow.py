import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow

from src.model.train import EXPERIMENT_NAME, ensure_experiment


def main() -> int:
    tracking_uri = mlflow.get_tracking_uri()
    print(f"tracking URI: {tracking_uri}")

    if "dagshub" not in tracking_uri.lower():
        print(
            "\nWARNING: this doesn't look like a DagsHub URL - MLFLOW_TRACKING_URI "
            "probably isn't set. Check that .env exists in the current directory "
            "and has MLFLOW_TRACKING_URI filled in, then run this again.\n"
        )
        return 1

    ensure_experiment()
    print(f"experiment '{EXPERIMENT_NAME}' ready")

    with mlflow.start_run(run_name="smoke-test") as run:
        mlflow.log_param("smoke_test", True)
        print(f"run created successfully: {run.info.run_id}")

    print("MLflow connectivity OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
