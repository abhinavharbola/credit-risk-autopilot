"""Lineage: the full N-hop champion history as a timeline, each entry showing
its metrics, promotion time, and whether/where it was rolled back to.
"""

import streamlit as st

from src.db.repository import get_champion_history


def render(engine) -> None:
    st.header("Champion lineage")

    with engine.connect() as conn:
        history = get_champion_history(conn)

    if not history:
        st.info("No champions promoted yet.")
        return

    st.markdown('<div class="crg-timeline">', unsafe_allow_html=True)
    for entry in reversed(history):
        rolled_back = entry["rolled_back_at"] is not None
        item_class = "crg-timeline-item rolled-back" if rolled_back else "crg-timeline-item"

        badge = (
            '<span class="crg-badge crg-badge-rollback">rolled back</span>'
            if rolled_back
            else '<span class="crg-badge crg-badge-promoted">active or superseded</span>'
        )
        stale_badge = (
            ' <span class="crg-badge crg-badge-stale">reference stale</span>'
            if entry.get("reference_stale")
            else ""
        )

        metrics_line = " · ".join(
            f"{k}: {v:.4f}" for k, v in entry["window_metrics"].items()
        )

        rollback_line = (
            f"<div>Rolled back to version {entry['rolled_back_to_version']} "
            f"at {entry['rolled_back_at']}</div>"
            if rolled_back
            else ""
        )

        st.markdown(
            f"""
            <div class="{item_class}">
                <div class="crg-card">
                    <strong>Version {entry['model_version']}</strong> {badge}{stale_badge}
                    <div>Promoted at {entry['promoted_at']}</div>
                    <div>Window metrics: {metrics_line}</div>
                    {rollback_line}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
