from typing import Any

import pandas as pd
from sqlalchemy.engine import Connection

from src.db.repository import advance_pipeline_state, get_pipeline_state, write_audit_log
from src.orchestration.pipeline import run_tick


def claim_and_run_tick(
    conn: Connection,
    raw_batches: list[pd.DataFrame],
    training_pool_df: pd.DataFrame,
    config: dict[str, Any],
    holdout_df: pd.DataFrame,
) -> dict[str, Any] | None:
    state = get_pipeline_state(conn)
    expected_version = state["version"]
    current_batch = state["current_batch"]

    if current_batch >= len(raw_batches):
        return {
            "batch": current_batch,
            "status": "past_end_of_dataset",
            "n_batches": len(raw_batches),
        }

    claimed = advance_pipeline_state(conn, expected_version)
    if not claimed:
        return None

    write_audit_log(
        conn,
        "clock_advance",
        {"batch": current_batch, "version": expected_version + 1},
    )
    return run_tick(conn, current_batch, raw_batches, training_pool_df, config, holdout_df)
