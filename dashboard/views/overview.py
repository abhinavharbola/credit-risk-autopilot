"""Overview: current clock position and production model, the metrics that
actually describe governance health (challenger win rate, rollback rate),
how champion quality has moved across promotions, a breakdown of how gate
evaluations resolve, and recent audit_log activity.
"""

import textwrap

import plotly.graph_objects as go
import streamlit as st

from src.db.repository import get_audit_log, get_champion_history, get_latest_champion, get_pipeline_state

EVENT_COLORS = {
    "drift_check": "#6172F3",
    "gate_evaluation": "#7C3AED",
    "promotion": "#027A48",
    "rollback": "#B42318",
    "rollback_check": "#B54708",
    "label_release": "#98A2B3",
}

KPI_DOT_COLORS = {
    "accent": "#2E3F73",
    "success": "#027A48",
    "warning": "#B54708",
    "danger": "#B42318",
}


def _kpi_card(label: str, value: str, caption: str, dot: str = "accent") -> None:
    color = KPI_DOT_COLORS.get(dot, KPI_DOT_COLORS["accent"])
    html = textwrap.dedent(
        '<div class="crg-kpi-card">'
        '<div class="crg-kpi-header">'
        f'<span class="crg-kpi-dot" style="background:{color};"></span>'
        f'<span class="crg-kpi-label">{label}</span>'
        "</div>"
        f'<div class="crg-kpi-value">{value}</div>'
        f'<div class="crg-kpi-caption">{caption}</div>'
        "</div>"
    ).strip()
    st.markdown(html, unsafe_allow_html=True)


def _empty(title: str, caption: str) -> None:
    html = textwrap.dedent(
        '<div class="crg-empty">'
        f'<div class="crg-empty-title">{title}</div>'
        f'<div class="crg-empty-caption">{caption}</div>'
        "</div>"
    ).strip()
    st.markdown(html, unsafe_allow_html=True)


def _section_header(title: str, meta: str = "") -> None:
    meta_html = f'<span class="crg-section-meta">{meta}</span>' if meta else ""
    html = (
        '<div class="crg-section-header">'
        f'<span class="crg-section-title">{title}</span>'
        f"{meta_html}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render(engine) -> None:
    with engine.connect() as conn:
        state = get_pipeline_state(conn)
        latest_champion = get_latest_champion(conn)
        champion_history = get_champion_history(conn)
        gate_evals = get_audit_log(conn, event_type="gate_evaluation", limit=2000)
        recent_events = get_audit_log(conn, limit=500)

    n_gate_evals = len(gate_evals)
    n_gate_promotions = sum(1 for e in gate_evals if e["event_payload"].get("promote"))
    n_gate_rejections = n_gate_evals - n_gate_promotions
    win_rate = n_gate_promotions / n_gate_evals if n_gate_evals else None

    n_promotions_total = len(champion_history)
    n_rollbacks = sum(1 for h in champion_history if h["rolled_back_at"] is not None)
    rollback_rate = n_rollbacks / n_promotions_total if n_promotions_total else None

    # --- KPI row -----------------------------------------------------
    cols = st.columns(4)
    with cols[0]:
        _kpi_card(
            "Current batch",
            str(state["current_batch"]),
            "pipeline clock position",
            dot="accent",
        )
    with cols[1]:
        if latest_champion:
            promoted_at = str(latest_champion["promoted_at"])[:19].replace("T", " ")
            caption = f"promoted {promoted_at}"
        else:
            caption = "no champion promoted yet"
        _kpi_card(
            "Production version",
            f'v{latest_champion["model_version"]}' if latest_champion else "none",
            caption,
            dot="accent",
        )
    with cols[2]:
        value = f"{win_rate:.0%}" if win_rate is not None else "—"
        caption = (
            f"{n_gate_promotions} of {n_gate_evals} challengers promoted"
            if n_gate_evals
            else "no challengers evaluated yet"
        )
        _kpi_card("Challenger win rate", value, caption, dot="success")
    with cols[3]:
        value = f"{rollback_rate:.0%}" if rollback_rate is not None else "—"
        caption = (
            f"{n_rollbacks} of {n_promotions_total} promotions reverted"
            if n_promotions_total
            else "no promotions yet"
        )
        dot = "danger" if (rollback_rate or 0) > 0 else "accent"
        _kpi_card("Rollback rate", value, caption, dot=dot)

    st.markdown('<div class="crg-divider"></div>', unsafe_allow_html=True)

    # --- current champion snapshot ------------------------------------
    _section_header("Current champion")
    if latest_champion:
        metrics_items = list(latest_champion["window_metrics"].items())
        metric_cols = st.columns(max(len(metrics_items), 1) + 1)
        for col, (metric, value) in zip(metric_cols, metrics_items):
            with col:
                _kpi_card(
                    metric.replace("_", " ").capitalize(),
                    f"{value:.4f}",
                    "on gated window",
                    dot="accent",
                )
        with metric_cols[-1]:
            status_label = "Reference stale" if latest_champion.get("reference_stale") else "Reference fresh"
            status_dot = "warning" if latest_champion.get("reference_stale") else "success"
            _kpi_card(
                "Rollback reference",
                status_label,
                "compared against live batches" if not latest_champion.get("reference_stale")
                else "checks suppressed until re-baselined",
                dot=status_dot,
            )
    else:
        _empty(
            "No champion has been promoted yet",
            "Run the bootstrap step, then advance the pipeline clock to see governance activity here.",
        )

    st.markdown('<div class="crg-divider"></div>', unsafe_allow_html=True)

    # --- champion performance trend -----------------------------------
    _section_header("Champion performance across promotions")
    if len(champion_history) >= 2:
        metric_name = next(iter(champion_history[0]["window_metrics"]))
        xs = list(range(1, len(champion_history) + 1))
        ys = [h["window_metrics"].get(metric_name) for h in champion_history]
        rolled_back_xs = [x for x, h in zip(xs, champion_history) if h["rolled_back_at"] is not None]
        rolled_back_ys = [y for y, h in zip(ys, champion_history) if h["rolled_back_at"] is not None]
        hover_versions = [f"v{h['model_version']}" for h in champion_history]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=metric_name,
                line=dict(color="#2E3F73", width=2.5),
                marker=dict(size=7, color="#2E3F73"),
                fill="tozeroy",
                fillcolor="rgba(46, 63, 115, 0.06)",
                text=hover_versions,
                hovertemplate="%{text}<br>" + metric_name + ": %{y:.4f}<extra></extra>",
            )
        )
        if rolled_back_xs:
            fig.add_trace(
                go.Scatter(
                    x=rolled_back_xs,
                    y=rolled_back_ys,
                    mode="markers",
                    name="rolled back",
                    marker=dict(color="#B42318", size=11, symbol="x", line=dict(width=2)),
                    hovertemplate="rolled back<br>" + metric_name + ": %{y:.4f}<extra></extra>",
                )
            )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#475467", size=12),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            xaxis=dict(title="promotion #", showgrid=False, dtick=1),
            yaxis=dict(title=metric_name, showgrid=True, gridcolor="#EEF1F5", zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        _empty(
            "Not enough promotions yet",
            "The trend chart needs at least two promoted champions to show how performance has moved.",
        )

    st.markdown('<div class="crg-divider"></div>', unsafe_allow_html=True)

    # --- gate evaluation breakdown + recent activity, side by side -----
    col_funnel, col_activity = st.columns([1, 1])

    with col_funnel:
        _section_header("Gate evaluations", meta=f"{n_gate_evals} total")
        if n_gate_evals:
            promoted_pct = 100 * n_gate_promotions / n_gate_evals
            rejected_pct = 100 * n_gate_rejections / n_gate_evals
            funnel_html = textwrap.dedent(
                '<div class="crg-funnel">'
                '<div class="crg-funnel-row">'
                '<div class="crg-funnel-row-label"><span>Promoted</span>'
                f'<span class="crg-funnel-row-count">{n_gate_promotions} &middot; {promoted_pct:.0f}%</span></div>'
                '<div class="crg-funnel-track">'
                f'<div class="crg-funnel-fill" style="width:{promoted_pct:.1f}%;background:var(--success);"></div>'
                "</div></div>"
                '<div class="crg-funnel-row">'
                '<div class="crg-funnel-row-label"><span>Rejected</span>'
                f'<span class="crg-funnel-row-count">{n_gate_rejections} &middot; {rejected_pct:.0f}%</span></div>'
                '<div class="crg-funnel-track">'
                f'<div class="crg-funnel-fill" style="width:{rejected_pct:.1f}%;background:var(--border-strong);"></div>'
                "</div></div>"
                "</div>"
            ).strip()
            st.markdown(funnel_html, unsafe_allow_html=True)
        else:
            _empty("No challengers evaluated yet", "Gate evaluations appear once a retrain is triggered.")

    with col_activity:
        _section_header("Recent activity", meta=f"last {len(recent_events)} events")
        event_counts: dict[str, int] = {}
        for e in recent_events:
            event_counts[e["event_type"]] = event_counts.get(e["event_type"], 0) + 1

        if not event_counts:
            _empty("No governance events recorded yet", "Advance the pipeline clock to generate activity.")
        else:
            labels = sorted(event_counts, key=lambda k: event_counts[k])
            values = [event_counts[k] for k in labels]
            bar_colors = [EVENT_COLORS.get(k, "#2E3F73") for k in labels]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=values,
                        y=labels,
                        orientation="h",
                        marker_color=bar_colors,
                        marker_line_width=0,
                        text=values,
                        textposition="outside",
                    )
                ]
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#475467", size=12),
                margin=dict(l=10, r=24, t=6, b=10),
                height=280,
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, family="JetBrains Mono")),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
