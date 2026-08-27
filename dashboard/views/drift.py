"""Drift: per-batch drift share over time from drift_check audit_log events,
with retrain-triggered batches marked and the configured threshold drawn in,
so a reviewer can see exactly why a given batch did or didn't trigger a
retrain, not just that one did.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.db.repository import get_audit_log
from src.utils.config import load_yaml

RETRAIN_THRESHOLD = load_yaml("config/gate_config.yaml")["reference_fingerprint_drift_threshold"]


def render(engine) -> None:
    st.markdown(
        '<div class="crg-section-title">Drift share vs training reference</div>',
        unsafe_allow_html=True,
    )

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
    triggered_df = df[df["retrain_triggered"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["batch"],
            y=df["drift_share"],
            mode="lines",
            name="drift share",
            line=dict(color="#0D7377", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(13, 115, 119, 0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=triggered_df["batch"],
            y=triggered_df["drift_share"],
            mode="markers",
            name="retrain triggered",
            marker=dict(color="#B9382A", size=9, line=dict(color="#FFFFFF", width=1.5)),
        )
    )
    fig.add_hline(
        y=RETRAIN_THRESHOLD,
        line_dash="dash",
        line_color="#7A8F88",
        annotation_text=f"retrain threshold ({RETRAIN_THRESHOLD})",
        annotation_font_size=11,
        annotation_font_color="#7A8F88",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#4A5D56", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=360,
        xaxis=dict(title="batch", showgrid=False),
        yaxis=dict(title="drift share", showgrid=True, gridcolor="#E8EDEB", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    triggered_batches = df[df["retrain_triggered"]]["batch"].tolist()
    if triggered_batches:
        st.caption(f"Retrain triggered at batches: {', '.join(map(str, triggered_batches))}")

    st.markdown(
        '<div class="crg-section-title" style="margin-top: 24px;">Raw drift checks</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
