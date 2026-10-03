import base64
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dashboard.views import audit_log, drift, lineage, overview
from src.db.connection import get_engine
from src.db.repository import get_latest_champion, get_pipeline_state

ASSETS_DIR = REPO_ROOT / "assets"
_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".ico": "image/x-icon"}


def _find_icon() -> Path | None:
    if not ASSETS_DIR.exists():
        return None
    for ext in ("*.svg", "*.png", "*.jpg", "*.jpeg", "*.ico"):
        matches = sorted(ASSETS_DIR.glob(ext))
        if matches:
            return matches[0]
    return None


ICON_PATH = _find_icon()

st.set_page_config(
    page_title="Credit Risk Governance",
    page_icon=str(ICON_PATH) if ICON_PATH else ":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_css() -> str:
    return (Path(__file__).parent / "styles.css").read_text()


@st.cache_resource
def load_brand_mark_html() -> str:
    if ICON_PATH is None:
        return '<div class="crg-brand-mark crg-brand-mark-fallback">CRG</div>'
    if ICON_PATH.suffix == ".svg":
        return f'<div class="crg-brand-mark">{ICON_PATH.read_text()}</div>'
    mime = _MIME_BY_SUFFIX.get(ICON_PATH.suffix, "image/png")
    encoded = base64.b64encode(ICON_PATH.read_bytes()).decode("ascii")
    return f'<div class="crg-brand-mark"><img src="data:{mime};base64,{encoded}" alt="logo"/></div>'


st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

engine = get_engine()

with engine.connect() as conn:
    state = get_pipeline_state(conn)
    champion = get_latest_champion(conn)

version_label = f'v{champion["model_version"]}' if champion else "none"
reference_stale = bool(champion and champion.get("reference_stale"))
dot_class = "warn" if reference_stale else ""
reference_label = "stale" if reference_stale else "fresh"

st.markdown(
    '<div class="crg-topbar">'
    f'<div class="crg-brand">{load_brand_mark_html()}'
    "<div>"
    '<div class="crg-brand-title">Credit Risk Governance</div>'
    '<div class="crg-brand-subtitle">Autonomous retrain, promote, and rollback pipeline</div>'
    "</div>"
    "</div>"
    '<div class="crg-status-strip">'
    '<div class="crg-status-item">'
    '<span class="crg-status-label">Batch</span>'
    f'<span class="crg-status-value">{state["current_batch"]}</span>'
    "</div>"
    '<div class="crg-status-item">'
    '<span class="crg-status-label">Production</span>'
    f'<span class="crg-status-value">{version_label}</span>'
    "</div>"
    '<div class="crg-status-item">'
    '<span class="crg-status-label">Reference</span>'
    f'<span class="crg-status-value"><span class="crg-dot {dot_class}"></span>&nbsp;{reference_label}</span>'
    "</div>"
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)

tab_overview, tab_lineage, tab_drift, tab_audit = st.tabs(
    ["Overview", "Lineage", "Drift", "Audit Log"]
)

with tab_overview:
    overview.render(engine)

with tab_lineage:
    lineage.render(engine)

with tab_drift:
    drift.render(engine)

with tab_audit:
    audit_log.render(engine)
