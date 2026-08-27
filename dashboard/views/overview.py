"""Overview: current clock position, current production model, latest
champion's headline metrics, and quick audit_log event counts.
"""

import textwrap

import plotly.graph_objects as go
import streamlit as st

from src.db.repository import get_audit_log, get_champion_history, get_pipeline_state

EVENT_COLORS = {
    "drift_check": "#6172F3",
    "gate_evaluation": "#7C3AED",
    "promotion": "#027A48",
    "rollback": "#B42318",
    "rollback_check": "#B54708",
    "label_release": "#98A2B3",
}


def _metric_card(label: str, value: str, caption: str = "") -> None:
    caption_html = f'<div class="crg-metric-caption">{caption}</div>' if caption else ""
    # Single-line, dedented HTML in one st.markdown call: a multi-line
    # indented f-string risks Streamlit's markdown parser misreading 4+
    # leading spaces as a code fence.
    html = textwrap.dedent(
        '<div class="crg-metric-card">'
        f'<div class="crg-metric-label">{label}</div>'
        f'<div class="crg-metric-value">{value}</div>'
        f"{caption_html}"
        "</div>"
    ).strip()
    st.markdown(html, unsafe_allow_html=True)


def render(engine) -> None:
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
            "Production version",
            f'v{latest_champion["model_version"]}' if latest_champion else "none",
        )
    with cols[2]:
        n_promotions = sum(1 for e in recent_events if e["event_type"] == "promotion")
        _metric_card("Promotions", str(n_promotions))
    with cols[3]:
        n_rollbacks = sum(1 for e in recent_events if e["event_type"] == "rollback")
        _metric_card("Rollbacks executed", str(n_rollbacks))

    st.markdown('<div style="height: 26px"></div>', unsafe_allow_html=True)

    if latest_champion:
        st.markdown('<div class="crg-section-title">Current champion</div>', unsafe_allow_html=True)
        metrics_items = list(latest_champion["window_metrics"].items())
        metric_cols = st.columns(max(len(metrics_items), 1))
        for col, (metric, value) in zip(metric_cols, metrics_items):
            with col:
                _metric_card(metric.replace("_", " ").upper(), f"{value:.4f}")

        if latest_champion.get("reference_stale"):
            st.markdown(
                textwrap.dedent(
                    '<div style="margin-top: 12px;">'
                    '<span class="crg-badge crg-badge-stale">reference stale</span>'
                    '<span style="color: var(--text-secondary); font-size: 0.83rem; '
                    'margin-left: 8px;">rollback checks are suppressed until the '
                    "reference window is re-baselined against fresh labeled data."
                    "</span></div>"
                ).strip(),
                unsafe_allow_html=True,
            )
    else:
        st.info("No champion has been promoted yet.")

    st.markdown('<div style="height: 26px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="crg-section-title">Recent activity</div>', unsafe_allow_html=True)

    event_counts: dict[str, int] = {}
    for e in recent_events:
        event_counts[e["event_type"]] = event_counts.get(e["event_type"], 0) + 1

    if not event_counts:
        st.info("No governance events recorded yet.")
        return

    labels = list(event_counts.keys())
    values = [event_counts[k] for k in labels]
    bar_colors = [EVENT_COLORS.get(k, "#2E3F73") for k in labels]

    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=bar_colors, marker_line_width=0)])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#475467", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#EEF1F5", zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
