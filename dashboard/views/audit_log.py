"""Audit log: every governance decision, filterable by event type. Each row
shows a short, human-readable summary (color-coded by event type) so the
list is glanceable; the full JSON payload is one click away, not the
default view. This is the view that proves the audit trail is real -
gate rejections and rollback checks show up here just as clearly as
promotions, not just in a database table nobody looks at.
"""

import textwrap

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


def _summarize(event_type: str, payload: dict) -> str:
    if event_type == "gate_evaluation":
        verdict = "promoted" if payload.get("promote") else "rejected"
        delta = payload.get("delta")
        return f"{verdict} &middot; delta {delta:+.4f}" if delta is not None else verdict
    if event_type == "promotion":
        return f"model version {payload.get('model_version')} promoted to production"
    if event_type == "rollback":
        return (
            f"reverted from v{payload.get('rolled_back_from')} "
            f"to v{payload.get('rolled_back_to')}"
        )
    if event_type == "rollback_check":
        if payload.get("reference_stale"):
            return "reference stale, rollback check suppressed"
        return "rollback triggered" if payload.get("rollback_triggered") else "no rollback needed"
    if event_type == "drift_check":
        fingerprint = payload.get("fingerprint", {})
        share = fingerprint.get("drift_share")
        share_str = f"{share:.3f}" if isinstance(share, (int, float)) else "n/a"
        verdict = "retrain triggered" if payload.get("retrain_triggered") else "no drift"
        return f"{verdict} &middot; drift share {share_str}"
    if event_type == "label_release":
        n_released = payload.get("n_labels_released")
        batch_id = payload.get("batch_id")
        return f"{n_released} labels released for batch {batch_id}"
    return ""


def render(engine) -> None:
    st.markdown('<div class="crg-section-title">Audit log</div>', unsafe_allow_html=True)

    selected = st.selectbox("Event type", EVENT_TYPES, label_visibility="collapsed")
    event_type = None if selected == "all" else selected

    with engine.connect() as conn:
        events = get_audit_log(conn, event_type=event_type, limit=500)

    if not events:
        st.info("No matching audit_log entries.")
        return

    for e in events:
        summary = _summarize(e["event_type"], e["event_payload"])
        row_html = textwrap.dedent(
            f'<div class="crg-audit-row type-{e["event_type"]}">'
            f'<span class="crg-audit-time">{e["created_at"]}</span>'
            f'<span class="crg-badge crg-badge-neutral">{e["event_type"]}</span>'
            f'<span class="crg-audit-summary">{summary}</span>'
            "</div>"
        ).strip()
        st.markdown(row_html, unsafe_allow_html=True)
        with st.expander("Full payload"):
            st.json(e["event_payload"])
