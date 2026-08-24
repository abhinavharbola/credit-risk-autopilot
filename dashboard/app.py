"""Streamlit entrypoint. Sidebar nav across four views (Overview, Lineage,
Drift, Audit log). Theme comes from .streamlit/config.toml; scoped extra CSS
(hairline cards, status badges, champion-lineage timeline) is injected from
dashboard/styles.css.
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

st.set_page_config(page_title="Credit Risk Governance", layout="wide")


@st.cache_resource
def load_css() -> str:
    return (Path(__file__).parent / "styles.css").read_text()


st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

VIEWS = {
    "Overview": overview.render,
    "Lineage": lineage.render,
    "Drift": drift.render,
    "Audit log": audit_log.render,
}

with st.sidebar:
    st.title("Credit Risk Governance")
    selection = st.radio("View", list(VIEWS.keys()), label_visibility="collapsed")

engine = get_engine()
VIEWS[selection](engine)
