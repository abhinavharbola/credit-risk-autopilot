"""Overview: current clock position, current production model, latest
champion's headline metrics, and quick audit_log event counts.
"""

import streamlit as st

from src.db.repository import get_audit_log, get_champion_history, get_pipeline_state


def _metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="crg-card">
            <div class="crg-metric-label">{label}</div>
            <div class="crg-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(engine) -> None:
    st.header("Overview")

    with engine.connect() as conn:
        state = get_pipeline_state(conn)
        history = get_champion_history(conn)
        recent_events = get_audit_log(conn, limit=500)

    latest_champion = None
    for entry in reversed(history):
        if entry["rolled_back_at"] is None:
            latest_champion = entry
            break

    cols = st.columns(4)
    with cols[0]:
        _metric_card("Current batch", str(state["current_batch"]))
    with cols[1]:
        _metric_card(
            "Production model version",
            latest_champion["model_version"] if latest_champion else "none",
        )
    with cols[2]:
        n_promotions = sum(1 for e in recent_events if e["event_type"] == "promotion")
        _metric_card("Total promotions", str(n_promotions))
    with cols[3]:
        n_rollbacks = sum(1 for e in recent_events if e["event_type"] == "rollback")
        _metric_card("Total rollbacks", str(n_rollbacks))

    if latest_champion:
        st.subheader("Current champion")
        metric_cols = st.columns(len(latest_champion["window_metrics"]) or 1)
        for col, (metric, value) in zip(metric_cols, latest_champion["window_metrics"].items()):
            with col:
                _metric_card(metric, f"{value:.4f}")

        if latest_champion.get("reference_stale"):
            st.markdown(
                '<span class="crg-badge crg-badge-stale">reference stale</span> '
                "rollback checks are currently suppressed until the reference "
                "window is re-baselined against fresh labeled data.",
                unsafe_allow_html=True,
            )
    else:
        st.info("No champion has been promoted yet.")

    st.subheader("Recent activity")
    event_counts: dict[str, int] = {}
    for e in recent_events:
        event_counts[e["event_type"]] = event_counts.get(e["event_type"], 0) + 1
    st.bar_chart(event_counts)
