"""FastAPI serving layer. Loads whatever is currently aliased @production,
cached with explicit invalidation: every request checks the alias's current
version via a cheap metadata call, and only reloads the actual model artifact
if the version changed. This is what "picks up a promotion/rollback without
a redeploy" means in practice.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import pandas as pd

from src.model.features import FEATURES
from src.utils.model_cache import AliasedModelCache

MODEL_NAME = "credit-risk-classifier"

_cache = AliasedModelCache(MODEL_NAME, "production")

app = FastAPI(title="Credit Risk Governance - Serving")


class PredictionRequest(BaseModel):
    """Field names use Python-safe identifiers; `alias` maps to the dataset's
    actual hyphenated column names (e.g. NumberOfTime30-59DaysPastDueNotWorse
    isn't a valid Python identifier). populate_by_name lets callers send
    either form.
    """

    RevolvingUtilizationOfUnsecuredLines: float = Field(ge=0)
    age: int = Field(ge=18, le=120)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(ge=0)
    MonthlyIncome: float = Field(ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(ge=0)
    NumberOfTimes90DaysLate: int = Field(ge=0)
    NumberRealEstateLoansOrLines: int = Field(ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: int = Field(ge=0)

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    predicted_prob: float
    predicted_label: int
    model_version: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, str]:
    try:
        _, version = _cache.get()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"model_name": MODEL_NAME, "production_version": version}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Scores a single applicant. Not persisted to the predictions table -
    that table's batch_id semantics belong to the simulated governance loop
    (src/orchestration/pipeline.py), not ad-hoc live requests. Extend this
    endpoint to write through if serving real traffic later.
    """
    try:
        model, version = _cache.get()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    row = request.model_dump(by_alias=True)
    df = pd.DataFrame([row])[FEATURES]
    prob = float(model.predict(df)[0])

    return PredictionResponse(
        predicted_prob=prob,
        predicted_label=int(prob >= 0.5),
        model_version=version,
    )
