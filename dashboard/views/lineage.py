"""Lineage: the full N-hop champion history as a timeline, each entry showing
its metrics, promotion time, and whether/where it was rolled back to.
"""

import textwrap

import streamlit as st

from src.db.repository import get_champion_history


def render(engine) -> None:
    st.markdown('<div class="crg-section-title">Champion lineage</div>', unsafe_allow_html=True)

    with engine.connect() as conn:
        history = get_champion_history(conn)

    if not history:
        st.info("No champions promoted yet.")
        return

    items_html = []
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

        metrics_line = " &middot; ".join(
            f"{k}: {v:.4f}" for k, v in entry["window_metrics"].items()
        )

        rollback_line = (
            f'<div class="crg-timeline-meta">Rolled back to '
            f'<span class="crg-timeline-meta-mono">v{entry["rolled_back_to_version"]}</span> '
            f'at {entry["rolled_back_at"]}</div>'
            if rolled_back
            else ""
        )

        # All markup on single concatenated lines, not an indented multi-line
        # f-string: 4+ leading spaces inside a markdown block gets misread
        # as a code fence by Streamlit's markdown renderer, which is what
        # produced literal "</div>" text in an earlier version of this view.
        items_html.append(
            f'<div class="{item_class}"><div class="crg-timeline-card">'
            f'<span class="crg-timeline-version">v{entry["model_version"]}</span>'
            f"{badge}{stale_badge}"
            f'<div class="crg-timeline-meta">Promoted {entry["promoted_at"]}</div>'
            f'<div class="crg-timeline-meta">{metrics_line}</div>'
            f"{rollback_line}"
            "</div></div>"
        )

    # Built and rendered as ONE st.markdown call: Streamlit renders each
    # st.markdown call as its own isolated DOM element, so a wrapper div
    # opened in one call and closed in another never actually nests the
    # content between them - the wrapping styling would silently do nothing.
    full_html = textwrap.dedent(
        f'<div class="crg-timeline">{"".join(items_html)}</div>'
    ).strip()
    st.markdown(full_html, unsafe_allow_html=True)
