"""Quick MLflow connectivity check. Creates one tiny run and exits - use this
to confirm the tracking server works BEFORE running the full demo loop,
since that takes 10-15+ minutes. Safe to delete after use, not part of the
pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow

from src.model.train import EXPERIMENT_NAME, _ensure_experiment  # noqa: E402
# ^ import order: loads .env as an import side-effect, must come after sys.path setup above

tracking_uri = mlflow.get_tracking_uri()
print(f"tracking URI: {tracking_uri}")

if "dagshub" not in tracking_uri.lower():
    print(
        "\nWARNING: this doesn't look like a DagsHub URL - MLFLOW_TRACKING_URI "
        "probably isn't set. Check that .env exists in the current directory "
        "and has MLFLOW_TRACKING_URI filled in, then run this again.\n"
    )
    sys.exit(1)

_ensure_experiment()
print(f"experiment '{EXPERIMENT_NAME}' ready")

with mlflow.start_run(run_name="smoke-test") as run:
    mlflow.log_param("smoke_test", True)
    print(f"run created successfully: {run.info.run_id}")

print("MLflow connectivity OK")

