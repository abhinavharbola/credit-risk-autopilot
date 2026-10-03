import textwrap

import streamlit as st

from src.db.repository import get_champion_history


def _delta_badge(delta: float, unit: str) -> str:
    if abs(delta) < 1e-6:
        return '<span class="crg-timeline-delta flat">flat vs previous (holdout)</span>'
    direction = "up" if delta > 0 else "down"
    sign = "+" if delta > 0 else ""
    return f'<span class="crg-timeline-delta {direction}">{sign}{delta:.4f} {unit} vs previous (holdout)</span>'


def render(engine) -> None:
    with engine.connect() as conn:
        history = get_champion_history(conn)

    if not history:
        st.markdown(
            '<div class="crg-empty"><div class="crg-empty-title">No champions promoted yet</div>'
            '<div class="crg-empty-caption">The first champion appears here once the '
            "bootstrap step runs.</div></div>",
            unsafe_allow_html=True,
        )
        return

    items_html = []
    for i, entry in enumerate(reversed(history)):
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

        window_line = " &middot; ".join(
            f"{k}: {v:.4f}" for k, v in entry["window_metrics"].items()
        )
        holdout_line = " &middot; ".join(
            f"{k}: {v:.4f}" for k, v in entry["holdout_metrics"].items()
        )
        metrics_line = f"reference window {window_line} &middot; frozen holdout {holdout_line}"

        prev_entry = history[len(history) - 1 - i - 1] if (len(history) - 1 - i - 1) >= 0 else None
        delta_html = ""
        if prev_entry is not None:
            metric_name = next(iter(entry["holdout_metrics"]), None)
            if metric_name and metric_name in prev_entry["holdout_metrics"]:
                delta = entry["holdout_metrics"][metric_name] - prev_entry["holdout_metrics"][metric_name]
                delta_html = " " + _delta_badge(delta, metric_name)

        rollback_line = (
            f'<div class="crg-timeline-meta">Rolled back to '
            f'<span class="crg-timeline-meta-mono">v{entry["rolled_back_to_version"]}</span> '
            f'at {entry["rolled_back_at"]}</div>'
            if rolled_back
            else ""
        )

        items_html.append(
            f'<div class="{item_class}"><div class="crg-timeline-card">'
            f'<span class="crg-timeline-version">v{entry["model_version"]}</span>'
            f"{badge}{stale_badge}{delta_html}"
            f'<div class="crg-timeline-meta">Promoted {entry["promoted_at"]}</div>'
            f'<div class="crg-timeline-meta">{metrics_line}</div>'
            f"{rollback_line}"
            "</div></div>"
        )

    full_html = textwrap.dedent(
        f'<div class="crg-timeline">{"".join(items_html)}</div>'
    ).strip()
    st.markdown(full_html, unsafe_allow_html=True)
