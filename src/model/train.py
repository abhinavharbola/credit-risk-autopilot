"""Trains a challenger model on the current training pool and logs it to
MLflow. Model quality is intentionally not the point (section 5) - a plain
logistic regression is enough; the governance loop around it is the
deliverable. Registers the model and aliases it @challenger (never the
deprecated stage-based API).
"""

import numpy as np
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.features import FEATURES, TARGET

MODEL_NAME = "credit-risk-classifier"


def train_challenger(train_df: pd.DataFrame, run_name: str) -> tuple[str, str, Pipeline]:
    """Trains and logs a challenger. Returns (run_id, registered_version, model)."""
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
