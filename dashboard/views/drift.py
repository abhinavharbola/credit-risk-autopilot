"""Drift: per-batch drift share over time from drift_check audit_log events,
with retrain-triggered batches marked and the configured threshold drawn in,
so a reviewer can see exactly why a given batch did or didn't trigger a
retrain, not just that one did.
"""

import textwrap

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.db.repository import get_audit_log
from src.utils.config import load_yaml

RETRAIN_THRESHOLD = load_yaml("config/gate_config.yaml")["reference_fingerprint_drift_threshold"]


def _stat_strip(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        '<div class="crg-stat-strip-item">'
        f'<div class="crg-stat-strip-label">{label}</div>'
        f'<div class="crg-stat-strip-value">{value}</div>'
        "</div>"
        for label, value in items
    )
    st.markdown(f'<div class="crg-stat-strip">{cells}</div>', unsafe_allow_html=True)


def render(engine) -> None:
    with engine.connect() as conn:
        events = get_audit_log(conn, event_type="drift_check", limit=1000)

    if not events:
        st.markdown(
            '<div class="crg-empty"><div class="crg-empty-title">No drift checks recorded yet</div>'
            '<div class="crg-empty-caption">Drift is checked on every clock tick, before the '
            "gate decides whether a retrain is worth running.</div></div>",
            unsafe_allow_html=True,
        )
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

    latest_share = df.iloc[-1]["drift_share"]
    is_drifting = latest_share is not None and latest_share >= RETRAIN_THRESHOLD
    status_label = "Drifting" if is_drifting else "Stable"
    status_class = "crg-badge-stale" if is_drifting else "crg-badge-promoted"

    header_html = textwrap.dedent(
        '<div class="crg-section-header">'
        '<span class="crg-section-title">Drift share vs training reference</span>'
        f'<span class="{status_class} crg-badge">{status_label} &middot; batch {int(df.iloc[-1]["batch"])}</span>'
        "</div>"
    ).strip()
    st.markdown(header_html, unsafe_allow_html=True)

    _stat_strip(
        [
            ("Batches checked", str(len(df))),
            ("Current drift share", f"{latest_share:.3f}" if latest_share is not None else "—"),
            ("Retrain threshold", f"{RETRAIN_THRESHOLD:.2f}"),
            ("Retrains triggered", str(len(triggered_df))),
        ]
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["batch"],
            y=df["drift_share"],
            mode="lines",
            name="drift share",
            line=dict(color="#2E3F73", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(46, 63, 115, 0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=triggered_df["batch"],
            y=triggered_df["drift_share"],
            mode="markers",
            name="retrain triggered",
            marker=dict(color="#B42318", size=9, line=dict(color="#FFFFFF", width=1.5)),
        )
    )
    fig.add_hline(
        y=RETRAIN_THRESHOLD,
        line_dash="dash",
        line_color="#98A2B3",
        annotation_text=f"retrain threshold ({RETRAIN_THRESHOLD})",
        annotation_font_size=11,
        annotation_font_color="#98A2B3",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#475467", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
        xaxis=dict(title="batch", showgrid=False),
        yaxis=dict(title="drift share", showgrid=True, gridcolor="#EEF1F5", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    triggered_batches = df[df["retrain_triggered"]]["batch"].tolist()
    if triggered_batches:
        st.caption(f"Retrain triggered at batches: {', '.join(map(str, triggered_batches))}")

    st.markdown('<div class="crg-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="crg-section-title">Raw drift checks</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
