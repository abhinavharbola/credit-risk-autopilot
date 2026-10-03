import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.ingest import (
    BATCHES_PATH,
    HOLDOUT_PATH,
    PROCESSED_DIR,
    TRAINING_POOL_PATH,
    confirm_positive_rate,
    load_processed_data,
    load_raw,
)
from src.data.split import (
    apply_imputation,
    build_pretrain_batches,
    carve_holdout,
    carve_stream,
    fit_imputation_medians,
)
from src.db.connection import get_connection, run_migrations
from src.db.repository import (
    get_champion_history,
    get_pipeline_state,
    insert_champion_history,
    write_audit_log,
)
from src.drift.detect import compute_fingerprint
from src.gate.evaluate import compute_metric
from src.model.features import TARGET
from src.model.train import score, train_challenger
from src.orchestration.clock import claim_and_run_tick
from src.orchestration.promote import set_production_alias
from src.utils.config import load_pipeline_config
from src.utils.logging import configure_logging

BATCH_SIZE = 200
HOLDOUT_FRACTION = 0.15
STREAM_FRACTION = 0.5
SPLIT_SEED = 42
N_TICKS = 25


def prepare_data() -> tuple[list, pd.DataFrame, pd.DataFrame]:
    paths = (BATCHES_PATH, TRAINING_POOL_PATH, HOLDOUT_PATH)
    if all(p.exists() for p in paths):
        print("processed data already present, reusing it")
        return load_processed_data()

    print("loading raw data from data/raw/...")
    raw = load_raw()
    rate = confirm_positive_rate(raw)
    print(f"loaded {len(raw)} rows, positive rate {rate:.4f}")

    remainder, holdout = carve_holdout(raw, holdout_frac=HOLDOUT_FRACTION, seed=SPLIT_SEED)
    base_pool, stream_pool = carve_stream(remainder, stream_frac=STREAM_FRACTION, seed=SPLIT_SEED)

    medians = fit_imputation_medians(base_pool)
    base_pool = apply_imputation(base_pool, medians)
    stream_pool = apply_imputation(stream_pool, medians)
    holdout = apply_imputation(holdout, medians)

    batches = build_pretrain_batches(stream_pool, batch_size=BATCH_SIZE, seed=SPLIT_SEED)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for path, obj in ((BATCHES_PATH, batches), (TRAINING_POOL_PATH, base_pool), (HOLDOUT_PATH, holdout)):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    print(
        f"prepared {len(batches)} stream batches of {BATCH_SIZE} rows, "
        f"{len(base_pool)} base pool rows, {len(holdout)} holdout rows"
    )
    return batches, base_pool, holdout


def ensure_fresh_database() -> None:
    with get_connection() as conn:
        state = get_pipeline_state(conn)
        history = get_champion_history(conn)
    if state["current_batch"] > 0 or history:
        raise SystemExit(
            "database already holds pipeline state (current_batch="
            f"{state['current_batch']}, champions={len(history)}). "
            "Truncate pipeline_state, predictions, champion_history and audit_log "
            "and reset pipeline_state to (1, 0, 0) before rerunning the demo loop."
        )


def bootstrap_champion(train_pool_df, holdout_df, gate_config: dict) -> None:
    print("training bootstrap champion...")
    _run_id, version, model = train_challenger(train_pool_df, run_name="bootstrap-champion")

    holdout_prob = score(model, holdout_df)
    holdout_metric = compute_metric(
        holdout_df[TARGET].to_numpy(),
        holdout_prob,
        gate_config["primary_metric"],
        gate_config["decision_threshold"],
    )
    bootstrap_fingerprint = compute_fingerprint(
        holdout_df,
        train_pool_df,
        ks_pvalue_threshold=gate_config["drift_ks_pvalue_threshold"],
    )

    metric_name = gate_config["primary_metric"]
    with get_connection() as conn:
        champion_history_id = insert_champion_history(
            conn,
            {
                "model_version": version,
                "holdout_metrics": {metric_name: holdout_metric},
                "window_metrics": {metric_name: holdout_metric},
                "drift_fingerprint": bootstrap_fingerprint,
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
    set_production_alias(version)
    print(f"bootstrap champion promoted: version {version}, holdout {metric_name} {holdout_metric:.4f}")


def run_ticks(batches, train_pool_df, holdout_df, config: dict, n_ticks: int) -> list[dict]:
    summary = []
    for i in range(n_ticks):
        with get_connection() as conn:
            result = claim_and_run_tick(conn, batches, train_pool_df, config, holdout_df)

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
    def rollback_flag(r: dict, key: str) -> bool:
        return bool(r.get("rollback", {}).get(key))

    print("\n=== demo loop summary ===")
    print(f"ticks run: {len(summary)}")
    print(f"drift detected: {sum(1 for r in summary if r.get('drift_detected'))}")
    print(f"retrains triggered: {sum(1 for r in summary if r.get('retrain_triggered'))}")
    print(f"promotions: {sum(1 for r in summary if r.get('promoted_champion_history_id'))}")
    print(f"rollbacks: {sum(1 for r in summary if rollback_flag(r, 'rollback_executed'))}")
    print(
        "degradation flagged without a rollback: "
        f"{sum(1 for r in summary if rollback_flag(r, 'degradation_flagged') and not rollback_flag(r, 'rollback_executed'))}"
    )
    print(
        "reference-stale flags (rollback checks suppressed): "
        f"{sum(1 for r in summary if rollback_flag(r, 'reference_stale'))}"
    )


def main() -> int:
    configure_logging()
    print("applying DB schema...")
    run_migrations()
    ensure_fresh_database()

    config = load_pipeline_config()
    batches, train_pool_df, holdout_df = prepare_data()
    bootstrap_champion(train_pool_df, holdout_df, config["gate"])
    summary = run_ticks(batches, train_pool_df, holdout_df, config, n_ticks=N_TICKS)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
