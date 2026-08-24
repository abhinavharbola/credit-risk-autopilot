"""Drift: per-batch drift share over time from drift_check audit_log events,
with retrain-triggered batches marked, so a reviewer can see the recession
scenario unfold and confirm the pipeline reacted to it.
"""

import pandas as pd
import streamlit as st

from src.db.repository import get_audit_log


def render(engine) -> None:
    st.header("Drift")

    with engine.connect() as conn:
        events = get_audit_log(conn, event_type="drift_check", limit=1000)

    if not events:
        st.info("No drift checks recorded yet.")
        return

    rows = []
    for e in events:
        payload = e["event_payload"]
        fingerprint = payload.get("fingerprint", {})
        rows.append(
            {
                "batch": payload.get("batch"),
                "drift_share": fingerprint.get("drift_share"),
                "retrain_triggered": payload.get("retrain_triggered"),
            }
        )

    df = pd.DataFrame(rows).sort_values("batch")

    st.line_chart(df.set_index("batch")["drift_share"])

    triggered_batches = df[df["retrain_triggered"]]["batch"].tolist()
    if triggered_batches:
        st.caption(f"Retrain triggered at batches: {', '.join(map(str, triggered_batches))}")

    st.subheader("Raw drift checks")
    st.dataframe(df, use_container_width=True)
