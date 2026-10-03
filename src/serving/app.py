from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import pandas as pd

from src.model.train import score
from src.utils.config import MODEL_NAME, PRODUCTION_ALIAS, load_yaml
from src.utils.logging import configure_logging, span
from src.utils.model_cache import AliasedModelCache

configure_logging()

CACHE_TTL_SECONDS = 30.0

_cache = AliasedModelCache(MODEL_NAME, PRODUCTION_ALIAS, ttl_seconds=CACHE_TTL_SECONDS)

_DECISION_THRESHOLD = load_yaml("config/gate_config.yaml")["decision_threshold"]

app = FastAPI(title="Credit Risk Governance - Serving")


class PredictionRequest(BaseModel):

    RevolvingUtilizationOfUnsecuredLines: float = Field(ge=0)
    age: int = Field(ge=18, le=120)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse"
    )
    DebtRatio: float = Field(ge=0)
    MonthlyIncome: float = Field(ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(ge=0)
    NumberOfTimes90DaysLate: int = Field(ge=0)
    NumberRealEstateLoansOrLines: int = Field(ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse"
    )
    NumberOfDependents: int = Field(ge=0)

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

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
    try:
        model, version = _cache.get()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    with span("predict", model_version=version):
        row = request.model_dump(by_alias=True)
        df = pd.DataFrame([row])
        prob = float(score(model, df)[0])

    return PredictionResponse(
        predicted_prob=prob,
        predicted_label=int(prob >= _DECISION_THRESHOLD),
        model_version=version,
    )
