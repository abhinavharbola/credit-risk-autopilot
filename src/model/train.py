"""Trains a challenger model on the current training pool and logs it to
MLflow. Model quality is intentionally not the point (section 5) - a plain
logistic regression is enough; the governance loop around it is the
deliverable. Registers the model and aliases it @challenger (never the
deprecated stage-based API).
"""

import numpy as np
import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.features import FEATURES, TARGET

# Loaded here explicitly, not just as a side-effect of importing
# src.db.connection elsewhere. This module talks to MLflow directly and
# needs MLFLOW_TRACKING_URI/USERNAME/PASSWORD from .env regardless of
# whether anything database-related has been imported yet - a standalone
# script that only imports this module (e.g. scripts/smoke_test_mlflow.py)
# would otherwise silently fall back to MLflow's local default tracking
# store instead of erroring, which is exactly the trap that produced a
# false-positive "MLflow connectivity OK" smoke test result.
load_dotenv()

MODEL_NAME = "credit-risk-classifier"
EXPERIMENT_NAME = "credit-risk-governance-v2"


def _ensure_experiment() -> None:
    """Explicitly get-or-creates a named experiment instead of relying on
    MLflow's implicit default experiment (id "0") existing. Without this,
    every start_run() call depends on that default experiment still being
    valid server-side - if it's ever deleted, archived, or otherwise
    unavailable (e.g. from cleanup done in the tracking server's UI),
    CreateRun fails with an opaque RestException that gives no hint the
    actual problem is the experiment, not the run. Called on every
    train_challenger() call rather than cached behind a flag: set_experiment
    is cheap and idempotent, and this way the code self-heals if the
    experiment is ever deleted again later, rather than trusting a
    one-time check.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)


def train_challenger(train_df: pd.DataFrame, run_name: str) -> tuple[str, str, Pipeline]:
    """Trains and logs a challenger. Returns (run_id, registered_version, model)."""
    _ensure_experiment()

    X = train_df[FEATURES]
    y = train_df[TARGET]

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X, y)

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("n_train_rows", len(train_df))
        mlflow.log_param("positive_rate", float(y.mean()))

        model_info = mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=MODEL_NAME
        )

        run_id = run.info.run_id
        version = model_info.registered_model_version

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(MODEL_NAME, "challenger", version)

    return run_id, version, model


def score(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Predicted probability of SeriousDlqin2yrs == 1."""
    return model.predict_proba(df[FEATURES])[:, 1]
