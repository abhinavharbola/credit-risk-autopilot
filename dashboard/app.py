"""Streamlit entrypoint. A persistent header shows live status (current
batch, production version) pulled once per load, followed by top-level tabs
for the four views. No sidebar: tabs read as a more product-like nav than a
radio-button list, and the header gives constant context regardless of
which tab is active.

Theme comes from .streamlit/config.toml; typography, color tokens, and
component styling are layered on top from dashboard/styles.css.
"""

import sys
import textwrap
from pathlib import Path

import streamlit as st

# streamlit run dashboard/app.py puts dashboard/ itself on sys.path, not the
# repo root, so `from src...` / `from dashboard...` below would fail. Insert
# the repo root explicitly, before any of those imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.views import audit_log, drift, lineage, overview
from src.db.connection import get_engine
from src.db.repository import get_latest_champion, get_pipeline_state

st.set_page_config(page_title="Credit Risk Governance", layout="wide")


@st.cache_resource
def load_css() -> str:
    return (Path(__file__).parent / "styles.css").read_text()


st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

engine = get_engine()

with engine.connect() as conn:
    state = get_pipeline_state(conn)
    champion = get_latest_champion(conn)

version_label = f'v{champion["model_version"]}' if champion else "none promoted"

header_html = textwrap.dedent(
    '<div class="crg-appheader">'
    '<div>'
    '<div class="crg-appheader-title">Credit Risk Governance</div>'
    '<div class="crg-appheader-subtitle">Autonomous retrain, promote, and rollback '
    "pipeline &middot; recession scenario simulation</div>"
    "</div>"
    '<div class="crg-status-pill"><span class="crg-status-dot"></span>'
    f'batch {state["current_batch"]} &middot; production {version_label}</div>'
    "</div>"
).strip()
st.markdown(header_html, unsafe_allow_html=True)

tab_overview, tab_lineage, tab_drift, tab_audit = st.tabs(
    ["Overview", "Lineage", "Drift", "Audit log"]
)

with tab_overview:
    overview.render(engine)

with tab_lineage:
    lineage.render(engine)

with tab_drift:
    drift.render(engine)

with tab_audit:
    audit_log.render(engine)
