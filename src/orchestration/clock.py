"""Explicit clock advance logic. This is the single implementation cron and
manual invocation both go through (via scripts/advance_clock.py) - there is
no second, subtly-different copy of this logic anywhere else (4.6a).

Claims the current batch by advancing pipeline_state BEFORE doing any tick
work. If the claim fails (another caller already advanced), returns None
with no side effects - this is what prevents a losing racer from still
inserting duplicate predictions/audit_log rows even though its own clock
advance was a no-op.
"""

from typing import Any

import pandas as pd
from sqlalchemy.engine import Connection

from src.db.repository import advance_pipeline_state, get_pipeline_state
from src.orchestration.pipeline import run_tick


def claim_and_run_tick(
    conn: Connection,
    raw_batches: list[pd.DataFrame],
    training_pool_df: pd.DataFrame,
    config: dict[str, Any],
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
        return None  # another caller already claimed this tick

    return run_tick(conn, current_batch, raw_batches, training_pool_df, config)
