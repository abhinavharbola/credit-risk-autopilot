"""Runs the full demo loop against real infra (build order step 10):
bootstrap -> drift -> retrain -> promote -> further drift -> rollback.

Single command that ties everything together:
  1. download + prep data (ingest, split, leakage-safe impute, batch)
  2. persist batches to data/processed/ so advance_clock.py / cron can reuse
     them later without re-running data prep
  3. apply the DB schema (idempotent)
  4. train + promote an initial "bootstrap" champion (no gate - nothing to
     compare against yet)
  5. run N ticks of the governance loop end to end
  6. print a summary: retrain triggers, promotions, rollbacks

Requires .env populated: DATABASE_URL, MLFLOW_TRACKING_URI/USERNAME/PASSWORD,
and Kaggle credentials for the dataset download (KAGGLE_USERNAME/KAGGLE_KEY
env vars, or a configured ~/.kaggle/kaggle.json).
"""

import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd

from src.data.ingest import confirm_positive_rate, download_dataset, load_raw
from src.data.split import (
    apply_imputation,
    build_pretrain_batches,
    carve_holdout,
    fit_imputation_medians,
)
from src.db.connection import get_connection, run_migrations
from src.db.repository import insert_champion_history, write_audit_log
from src.gate.evaluate import compute_metric
from src.model.features import TARGET
from src.model.train import score, train_challenger
from src.orchestration.clock import claim_and_run_tick
from src.utils.config import load_yaml

PROCESSED_DIR = Path("data/processed")
BATCH_SIZE = 200
# 25 batches: covers bootstrap, the persistent-drift onset at batch 10, the
# temporary concept-drift window (15-20), and enough runway past it to see
# whether a rollback is needed once that window reverts.
N_TICKS = 25
MODEL_NAME = "credit-risk-classifier"


def prepare_data() -> tuple[list, pd.DataFrame, pd.DataFrame]:
    print("downloading + loading raw data...")
    path = download_dataset()
    raw = load_raw(path)
    rate = confirm_positive_rate(raw)
    print(f"loaded {len(raw)} rows, positive rate {rate:.4f}")

    print("carving holdout (leakage guard: before any imputation is fit)...")
    train_pool, holdout = carve_holdout(raw, holdout_frac=0.15, seed=42)
    medians = fit_imputation_medians(train_pool)
    train_pool = apply_imputation(train_pool, medians)
    holdout = apply_imputation(holdout, medians)

    print("building pretrain batches...")
    batches = build_pretrain_batches(train_pool, batch_size=BATCH_SIZE, seed=42)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "pretrain_batches.pkl", "wb") as f:
        pickle.dump(batches, f)
    with open(PROCESSED_DIR / "training_pool.pkl", "wb") as f:
        pickle.dump(train_pool, f)
    with open(PROCESSED_DIR / "holdout.pkl", "wb") as f:
        pickle.dump(holdout, f)

    print(
        f"prepared {len(batches)} batches of {BATCH_SIZE} rows, "
        f"{len(train_pool)} train pool rows, {len(holdout)} holdout rows"
    )
    return batches, train_pool, holdout


def bootstrap_champion(train_pool_df, holdout_df) -> None:
    """Trains and promotes the very first model directly - there's no prior
    champion to gate against yet, so the three-gate logic doesn't apply here.
    """
    print("training bootstrap champion...")
    _run_id, version, model = train_challenger(train_pool_df, run_name="bootstrap-champion")

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "production", version)

    holdout_prob = score(model, holdout_df)
    holdout_metric = compute_metric(holdout_df[TARGET].to_numpy(), holdout_prob, "auc_pr", 0.5)

    with get_connection() as conn:
        champion_history_id = insert_champion_history(
            conn,
            {
                "model_version": version,
                "holdout_metrics": {"auc_pr": holdout_metric},
                # no drifted window exists yet at bootstrap time, so the
                # holdout metric stands in as the initial rollback reference
                "window_metrics": {"auc_pr": holdout_metric},
                "drift_fingerprint": {"drift_share": 0.0, "column_drift_scores": {}},
            },
        )
        write_audit_log(
            conn,
            event_type="promotion",
            payload={
                "champion_history_id": champion_history_id,
                "model_version": version,
                "reason": "bootstrap - first champion, no prior model to gate against",
            },
        )
    print(f"bootstrap champion promoted: version {version}, holdout AUC-PR {holdout_metric:.4f}")


def run_ticks(batches, train_pool_df, n_ticks: int) -> list[dict]:
    drift_params = load_yaml("config/drift_params.yaml")
    gate_config = load_yaml("config/gate_config.yaml")
    config = {**drift_params, "gate": gate_config}

    summary = []
    for i in range(n_ticks):
        with get_connection() as conn:
            result = claim_and_run_tick(conn, batches, train_pool_df, config)

        if result is None:
            print(f"tick {i}: claim lost to another caller, skipping")
            continue
        if result.get("status") == "past_end_of_dataset":
            print("reached end of simulated dataset, stopping")
            break

        print(f"tick {i}: {result}")
        summary.append(result)
    return summary


def print_summary(summary: list[dict]) -> None:
    n_retrain_triggers = sum(1 for r in summary if r.get("retrain_triggered"))
    n_promotions = sum(1 for r in summary if r.get("promoted_champion_history_id"))
    n_rollbacks = sum(1 for r in summary if r.get("rollback", {}).get("rollback_triggered"))
    n_stale_flags = sum(1 for r in summary if r.get("rollback", {}).get("reference_stale"))

    print("\n=== demo loop summary ===")
    print(f"ticks run: {len(summary)}")
    print(f"retrain triggers: {n_retrain_triggers}")
    print(f"promotions: {n_promotions}")
    print(f"rollbacks: {n_rollbacks}")
    print(f"reference-stale flags (rollback checks suppressed): {n_stale_flags}")


def main() -> int:
    print("applying DB schema...")
    run_migrations()

    batches, train_pool_df, holdout_df = prepare_data()
    bootstrap_champion(train_pool_df, holdout_df)
    summary = run_ticks(batches, train_pool_df, n_ticks=N_TICKS)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
