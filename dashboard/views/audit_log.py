"""Audit log: every governance decision, filterable by event type. This is
the view that proves the fix from the review - gate rejections and rollback
checks show up here just as clearly as promotions.
"""

import streamlit as st

from src.db.repository import get_audit_log

EVENT_TYPES = [
    "all",
    "gate_evaluation",
    "promotion",
    "rollback",
    "rollback_check",
    "drift_check",
    "label_release",
]


def render(engine) -> None:
    st.header("Audit log")

    selected = st.selectbox("Event type", EVENT_TYPES)
    event_type = None if selected == "all" else selected

    with engine.connect() as conn:
        events = get_audit_log(conn, event_type=event_type, limit=500)

    if not events:
        st.info("No matching audit_log entries.")
        return

    for e in events:
        with st.expander(f"{e['created_at']} · {e['event_type']}"):
            st.json(e["event_payload"])
