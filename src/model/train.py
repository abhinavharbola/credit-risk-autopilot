import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.features import FEATURES, TARGET
from src.utils.config import MODEL_NAME

load_dotenv()

EXPERIMENT_NAME = "credit-risk-governance-v2"


def ensure_experiment() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)


def train_challenger(train_df: pd.DataFrame, run_name: str) -> tuple[str, str, Pipeline]:
    ensure_experiment()

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
        version = str(model_info.registered_model_version)

    return run_id, version, model


def score(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, 1]
