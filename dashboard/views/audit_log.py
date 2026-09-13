"""Audit log: every governance decision, filterable by event type. Each row
shows a short, human-readable summary (color-coded by event type) so the
list is glanceable; the full JSON payload is one click away, not the
default view. This is the view that proves the audit trail is real -
gate rejections and rollback checks show up here just as clearly as
promotions, not just in a database table nobody looks at.

Rather than dumping every recorded event on load, the log opens on a small
"Recent" slice with quick-view chips (Promotions, Rollbacks, ...) and a
separate range chip (Last 10 / 25 / 50 / All) - the statement-view pattern
banking apps use, so the common case ("did anything get promoted lately")
doesn't require scrolling past hundreds of drift_check/label_release rows
first.
"""

import textwrap
from typing import Callable

import streamlit as st

from src.db.repository import get_audit_log
from src.llm.explain import explain_event

# label -> (event_type to query for, optional predicate over event_payload
# to narrow further). Rejections and promotions are both gate_evaluation
# rows distinguished only by payload["promote"], so a plain event_type
# filter can't separate them on its own.
QUICK_VIEWS: dict[str, tuple[str | None, Callable[[dict], bool] | None]] = {
    "Recent": (None, None),
    "Promotions": ("promotion", None),
    "Rollbacks": ("rollback", None),
    "Rejected challengers": ("gate_evaluation", lambda p: not p.get("promote")),
    "Drift checks": ("drift_check", None),
    "Label releases": ("label_release", None),
    "All": (None, None),
}

RANGE_OPTIONS = {"Last 10": 10, "Last 25": 25, "Last 50": 50, "Last 100": 100, "All": 5000}


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
        # rollback_triggered here only means "degradation flagged", not
        # "reverted": this row's own payload never records whether a
        # valid prior champion existed to revert to. When one did, a
        # separate "rollback" event (above) is the actual reversion
        # record - look for it right next to this one.
        if payload.get("rollback_triggered"):
            return "degradation flagged, see nearby rollback event for outcome"
        return "no rollback needed"
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
    col_view, col_range = st.columns([3, 2])
    with col_view:
        view = st.radio("Quick view", list(QUICK_VIEWS), horizontal=True, label_visibility="collapsed")
    with col_range:
        range_label = st.radio(
            "Range", list(RANGE_OPTIONS), horizontal=True, label_visibility="collapsed", index=1
        )

    event_type, predicate = QUICK_VIEWS[view]
    limit = RANGE_OPTIONS[range_label]

    with engine.connect() as conn:
        events = get_audit_log(conn, event_type=event_type, limit=limit)

    if predicate is not None:
        events = [e for e in events if predicate(e["event_payload"])]

    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

    if not events:
        st.info("No matching audit_log entries for this view.")
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
            if st.button("Explain in plain language", key=f"explain-{e['id']}"):
                with st.spinner("Asking the LLM..."):
                    st.write(explain_event(e["event_type"], e["event_payload"]))
