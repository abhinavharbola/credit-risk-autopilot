"""Streamlit entrypoint. Sidebar-driven navigation (brand mark, nav list,
persistent live-status footer) rather than top tabs - reads as an internal
ops tool with a fixed shell, not a set of report tabs. Each page gets its
own title/description in the main content area; the sidebar footer's status
(batch, production version, reference health) is fetched once per load and
stays visible regardless of which page is active.

Theme comes from dashboard/styles.css (typography, color tokens, sidebar
and component styling). .streamlit/config.toml pins the base Streamlit
theme to light so it can't fall back to a dark variant on a client machine.
"""

import sys
from pathlib import Path

import streamlit as st

# streamlit run dashboard/app.py puts dashboard/ itself on sys.path, not the
# repo root, so `from src...` / `from dashboard...` below would fail. Insert
# the repo root explicitly, before any of those imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.views import audit_log, drift, lineage, overview
from src.db.connection import get_engine
from src.db.repository import get_latest_champion, get_pipeline_state

st.set_page_config(
    page_title="Credit Risk Governance",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_css() -> str:
    return (Path(__file__).parent / "styles.css").read_text()


st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

engine = get_engine()

with engine.connect() as conn:
    state = get_pipeline_state(conn)
    champion = get_latest_champion(conn)

PAGES = {
    "Overview": {
        "render": overview.render,
        "desc": "Governance health at a glance: current champion, challenger win rate, and rollback rate.",
    },
    "Lineage": {
        "render": lineage.render,
        "desc": "Full N-hop champion history, in promotion order, with performance deltas between hops.",
    },
    "Drift": {
        "render": drift.render,
        "desc": "Per-batch drift share against the training reference, with retrain-triggering batches marked.",
    },
    "Audit Log": {
        "render": audit_log.render,
        "desc": "Every governance decision the pipeline has recorded, filterable by event type.",
    },
}

with st.sidebar:
    st.markdown(
        '<div class="crg-brand">'
        '<div class="crg-brand-mark">CRG</div>'
        "<div>"
        '<div class="crg-brand-name">Credit Risk<br/>Governance</div>'
        '<div class="crg-brand-sub">Autonomous MLOps pipeline</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="crg-nav-label">Navigate</div>', unsafe_allow_html=True)
    selected_page = st.radio(
        "Navigate", list(PAGES.keys()), label_visibility="collapsed", key="crg_nav"
    )

    version_label = f'v{champion["model_version"]}' if champion else "none"
    reference_stale = bool(champion and champion.get("reference_stale"))
    dot_class = "warn" if reference_stale else ""
    reference_label = "stale" if reference_stale else "fresh"

    st.markdown(
        '<div class="crg-sidebar-footer">'
        '<div class="crg-sidebar-status-row">'
        '<span class="crg-sidebar-status-label">Batch</span>'
        f'<span class="crg-sidebar-status-value">{state["current_batch"]}</span>'
        "</div>"
        '<div class="crg-sidebar-status-row">'
        '<span class="crg-sidebar-status-label">Production</span>'
        f'<span class="crg-sidebar-status-value">{version_label}</span>'
        "</div>"
        '<div class="crg-sidebar-status-row">'
        '<span class="crg-sidebar-status-label">Reference</span>'
        f'<span class="crg-sidebar-status-value"><span class="crg-dot {dot_class}"></span>{reference_label}</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

page = PAGES[selected_page]
st.markdown(
    '<div class="crg-page-header">'
    f'<div class="crg-page-title">{selected_page}</div>'
    f'<div class="crg-page-desc">{page["desc"]}</div>'
    "</div>",
    unsafe_allow_html=True,
)
page["render"](engine)
