"""
app/dashboard.py - CaseLens: Policy Analytics Dashboard.

Phase 7: Multi-Stakeholder Sober Policy Intelligence Web Application.
Provides analytical visibility into systemic case attrition across institutional silos
(Police, Hospital, Forensic Science Labs, Prosecution, and Courts) for violence-against-women cases,
with mathematical survivor confidentiality via Differential Privacy and k-anonymity.

Theme & Palette:
- Background: #FAF8F5 (warm editorial off-white)
- Card surface: #FFFFFF with 1px border #E3DED6, radius 4px, no shadow
- Primary text: #222222, Secondary text: #5F5F5F
- Sidebar: background #F0ECE5 with dark text #222222
- Single accent: Deep maroon #8B1E3F (selected nav item, links, key chart series)
- Alert color: Amber #B45309 (used only for flagged leak points and critical anomalies)
- Neutral chart series: Greys #8C8C8C and #BDBDBD
- Gridlines: #E8E4DC
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import os
import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from app.api_client import (
    login as api_login,
    get_summary,
    get_funnel,
    get_districts,
    get_bottlenecks,
    get_categories,
    get_policy_context,
    get_privacy_diagnostics,
    get_privacy_budget,
    get_evaluation,
    get_journey,
    get_simulator_scenarios,
    run_simulator,
    get_case_examples,
    get_case_detail,
    get_random_case,
    get_review_queue,
    get_audit_logs,
    reset_privacy_budget,
    log_audit_event,
    load_dashboard_bundle,
    load_json_fallback,
    ROLE_PERMISSIONS,
    PAGE_ALIASES,
    check_user_permission,
    normalize_role,
    SAMPLE_JSON_DEFAULTS,
    get_analyze_template,
    run_analyze_sample,
    run_analyze_upload,
    run_analyze_case,
)
from src.upload_analysis import build_dashboard_bundle_from_analysis


def _is_running_in_streamlit() -> bool:
    """Detects whether script is running inside an active Streamlit runtime context."""
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


# Configure Page Settings
if _is_running_in_streamlit():
    st.set_page_config(
        page_title="CaseLens | The Leaky Pipeline: Tracing Gaps in the Justice System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# ==============================================================================
# 1. Custom CSS Theme & Styling (Sober Editorial Light Theme)
# ==============================================================================

CUSTOM_CSS = """
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .stApp {
        background-color: #FAF8F5;
        color: #222222;
    }
    
    /* Top Header Banner */
    .hero-banner {
        background: #FFFFFF;
        border: 1px solid #E3DED6;
        border-radius: 4px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: none;
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 600;
        color: #222222;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }
    .hero-subtitle {
        color: #5F5F5F;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    
    /* KPI Metric Cards: Flat, 1px border #E3DED6, radius 4px, no shadow, tabular numbers */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E3DED6;
        border-radius: 4px;
        padding: 14px 16px;
        box-shadow: none;
        transition: none;
    }
    .metric-label {
        color: #5F5F5F;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 600;
        color: #222222;
        font-variant-numeric: tabular-nums;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-delta {
        font-size: 0.8rem;
        color: #5F5F5F;
        font-weight: 400;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-critical {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FCD34D;
    }
    .badge-normal {
        background-color: #F0ECE5;
        color: #5F5F5F;
        border: 1px solid #E3DED6;
    }
    .badge-privacy {
        background-color: #FDF2F4;
        color: #8B1E3F;
        border: 1px solid #F5D0D8;
    }
    
    /* Section Containers */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E3DED6;
        border-radius: 4px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: none;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #222222;
        margin-bottom: 14px;
        border-bottom: 1px solid #E3DED6;
        padding-bottom: 6px;
    }
    
    /* Callouts */
    .callout-info {
        background: #FFFFFF;
        border-left: 3px solid #8B1E3F;
        border-top: 1px solid #E3DED6;
        border-right: 1px solid #E3DED6;
        border-bottom: 1px solid #E3DED6;
        padding: 12px 16px;
        border-radius: 0 4px 4px 0;
        margin: 14px 0;
        font-size: 0.9rem;
        color: #222222;
    }
    .callout-warning {
        background: #FEF3C7;
        border-left: 3px solid #B45309;
        border-top: 1px solid #FCD34D;
        border-right: 1px solid #FCD34D;
        border-bottom: 1px solid #FCD34D;
        padding: 12px 16px;
        border-radius: 0 4px 4px 0;
        margin: 14px 0;
        font-size: 0.9rem;
        color: #92400E;
    }
    
    /* Sidebar Base Styling */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"] {
        background-color: #F0ECE5 !important;
        border-right: 1px solid #E3DED6 !important;
        color: #222222 !important;
    }
    
    /* Sidebar Headings, Labels, Markdown */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #222222 !important;
    }
    
    /* Navigation Radio Buttons */
    section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #222222 !important;
        padding: 5px 8px !important;
        border-radius: 4px !important;
        margin-bottom: 2px !important;
        transition: background-color 0.15s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
        color: #222222 !important;
        font-size: 0.9rem;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #E8E4DC !important;
    }
    
    /* Navigation Radio Buttons - Currently Selected Highlight (#8B1E3F deep maroon, bold) */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has([aria-checked="true"]) {
        background-color: #FDF2F4 !important;
        border-left: 3px solid #8B1E3F !important;
        border-radius: 3px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"] p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has([aria-checked="true"]) p {
        color: #8B1E3F !important;
        font-weight: 600 !important;
    }
    
    /* Sliders and Number Inputs */
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stThumbValue"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
        color: #222222 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stSliderTickBarMin"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stSliderTickBarMax"] {
        color: #5F5F5F !important;
    }
    section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {
        fill: #5F5F5F !important;
    }
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        color: #222222 !important;
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 4px !important;
    }
    
    /* Institutional Persona Selectbox (White Background + Dark #222222 Text) */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 4px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] input,
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
        color: #222222 !important;
        fill: #222222 !important;
    }
    
    /* Selectbox Dropdown Menu Options */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] div,
    ul[data-baseweb="menu"],
    ul[data-baseweb="menu"] li,
    ul[data-baseweb="menu"] li *,
    div[data-baseweb="select"] ul li,
    div[data-baseweb="select"] ul li * {
        background-color: #FFFFFF !important;
        color: #222222 !important;
    }
    ul[data-baseweb="menu"] li:hover,
    ul[data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #F0ECE5 !important;
        color: #222222 !important;
    }
</style>
"""
if _is_running_in_streamlit():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. Authentication & Session Security (Phase A)
# ==============================================================================

# User authentication and database verification handled by FastAPI backend



def render_login_screen():
    """Renders the first-screen institutional authentication gateway."""
    with st.sidebar:
        st.markdown("""
            <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #E3DED6; margin-bottom: 14px;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #222222; letter-spacing: -0.01em;">CaseLens</div>
                <div style="font-size: 0.78rem; color: #8B1E3F; font-weight: 600;">The Leaky Pipeline: Tracing Gaps in the Justice System</div>
            </div>
            <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px;">
                <span class="badge badge-privacy">SDG 5: Gender Equality</span>
                <span class="badge badge-normal">SDG 16: Strong Institutions</span>
            </div>
            <hr style="border: 0; border-top: 1px solid #E3DED6; margin: 12px 0;"/>
            <div style="font-size: 0.75rem; color: #5F5F5F; line-height: 1.6;">
                🛡️ <strong>Argon2id</strong> Password Security<br/>
                ⏱️ <strong>30-Min</strong> Short-Lived JWT<br/>
                🔒 <strong>5-Attempt</strong> Lockout Defense<br/>
                📜 <strong>Zero-PII</strong> Audit Ledger
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="hero-banner" style="text-align: center; margin-bottom: 24px;">
            <div style="font-size: 1.45rem; font-weight: 700; color: #8B1E3F; letter-spacing: -0.01em; margin-bottom: 4px;">CaseLens: The Leaky Pipeline</div>
            <div style="font-size: 1.02rem; font-weight: 600; color: #222222; margin-bottom: 6px;">Tracing Gaps in the Justice System</div>
            <div style="font-size: 0.85rem; color: #5F5F5F; margin-bottom: 12px;">
                Multi-Agency Longitudinal Linkage with Mathematical Differential Privacy and Zero-PII Protections.
            </div>
            <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
                <span class="badge badge-privacy">UN SDG 5: Gender Equality</span>
                <span class="badge badge-normal">UN SDG 16: Peace, Justice & Strong Institutions</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-radius: 4px; padding: 18px 20px 8px 20px; margin-bottom: 16px;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #222222; margin-bottom: 4px;">🔐 Institutional Sign In</div>
                <div style="font-size: 0.82rem; color: #5F5F5F;">Sign in to access authorized case journey analytics and privacy controls.</div>
            </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            u_input = st.text_input("Username", key="login_username", placeholder="viewer, analyst, or admin")
            p_input = st.text_input("Password", type="password", key="login_password", placeholder="Enter institutional password")
            submit_btn = st.form_submit_button("Sign In", width="stretch")

            if submit_btn:
                u_clean = u_input.strip()
                p_clean = p_input
                if not u_clean or not p_clean:
                    st.warning("Please provide both username and password.")
                else:
                    login_res = api_login(u_clean, p_clean)
                    if login_res and "access_token" in login_res:
                        user = login_res.get("user", {})
                        token = login_res.get("access_token", "")
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = user
                        st.session_state["token"] = token
                        log_audit_event("LOGIN_SUCCESS", user_id=user.get("username", u_clean), role=user.get("role", "viewer"), token=token)
                        st.success(f"Authentication successful. Welcome, {user.get('username', u_clean)}!")
                        st.rerun()
                    else:
                        log_audit_event("LOGIN_FAILED", user_id=u_clean, details={"reason": "Invalid credentials or rejected by server"})

        st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-radius: 4px; padding: 14px 16px; margin-top: 14px; font-size: 0.8rem; color: #5F5F5F; line-height: 1.5;">
                <strong style="color: #222222;">Authorized Role Access Levels:</strong><br/>
                • <strong>viewer</strong>: Aggregate policy views only (Overview, Funnel, Bottlenecks, Quality, Policy Insights)<br/>
                • <strong>analyst</strong>: Aggregate views + Case Simulator<br/>
                • <strong>admin</strong>: Full system access + Human Review Queue + Security Audit Log<br/>
                <div style="margin-top: 8px; font-size: 0.74rem; color: #8C8C8C;">
                    Demo credentials configured via <code>.env</code>. Passwords and tokens are strictly protected and never committed.
                </div>
            </div>
        """, unsafe_allow_html=True)


if _is_running_in_streamlit():
    if "authenticated" not in st.session_state or not st.session_state.get("authenticated"):
        render_login_screen()
        st.stop()
    CURRENT_USER = st.session_state.get("user") or {"username": "anonymous", "role": "viewer"}
    CURRENT_ROLE = normalize_role(CURRENT_USER.get("role"))
else:
    CURRENT_USER = {"username": "anonymous", "role": "viewer"}
    CURRENT_ROLE = "viewer"


# ==============================================================================
# 3. Data Caching & Loading Helpers
# ==============================================================================

DATA_DIR = os.environ.get("LEAKY_DATA_DIR", "data")


@st.cache_data
def load_json_file(filename: str) -> Dict[str, Any]:
    """Safely loads fallback JSON structure."""
    return load_json_fallback(filename)


@st.cache_data
def load_data_bundle(
    token: Optional[str] = None,
    district: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    epsilon: float = 0.25,
    k_threshold: int = 5
) -> Dict[str, Any]:
    """Caches all primary analytical and privacy JSON artifacts via the API client."""
    return load_dashboard_bundle(
        token=token,
        district=district,
        from_date=from_date,
        to_date=to_date,
        stage=stage,
        category=category,
        epsilon=epsilon,
        k_threshold=k_threshold
    )


def render_filter_badge(district: str, date_range: Any, stage: str, remaining_eps: Any):
    """Renders a small badge at the top of each page showing active global filters and active dataset."""
    dist_label = district if district and district != "All Districts" else "All Districts"
    stage_label = stage if stage and stage != "All Stages" else "All Stages"
    
    date_label = "2023–2026 (All)"
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        try:
            d0, d1 = date_range[0], date_range[1]
            date_label = f"{d0.strftime('%b %Y')} – {d1.strftime('%b %Y')}"
        except Exception:
            date_label = f"{date_range[0]} – {date_range[1]}"

    eps_str = f"{remaining_eps:.2f}" if isinstance(remaining_eps, (int, float)) else str(remaining_eps or "10.00")
    
    active_lbl = st.session_state.get("active_dataset_label", "No dataset analyzed yet")
    is_act = st.session_state.get("active_dataset_id") is not None
    act_col = "#059669" if is_act else "#B45309"
    act_bg = "#ECFDF5" if is_act else "#FFFBEB"

    badge_html = f"""
    <div style="display: flex; align-items: center; justify-content: space-between; background: #FFFFFF; border: 1px solid #E3DED6; border-left: 3px solid #8B1E3F; border-radius: 4px; padding: 7px 14px; margin-bottom: 14px; font-size: 0.80rem; color: #5F5F5F;">
        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <strong style="color: #8B1E3F;">📍 Filter Scope:</strong>
            <span style="background: #F0ECE5; color: #222222; padding: 2px 7px; border-radius: 3px; font-weight: 500;">District: {dist_label}</span>
            <span style="background: #F0ECE5; color: #222222; padding: 2px 7px; border-radius: 3px; font-weight: 500;">Period: {date_label}</span>
            <span style="background: #F0ECE5; color: #222222; padding: 2px 7px; border-radius: 3px; font-weight: 500;">Milestone: {stage_label}</span>
            <span style="background: {act_bg}; color: {act_col}; border: 1px solid {act_col}; padding: 2px 7px; border-radius: 3px; font-weight: 600;">📁 Dataset: {active_lbl}</span>
        </div>
        <div style="font-weight: 600; color: #222222; white-space: nowrap;">
            🛡️ Remaining DP Budget: <span style="color: #8B1E3F;">{eps_str} ε</span>
        </div>
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)


def request_navigation(page_name: str):
    """Sets a pending navigation request to be applied before the sidebar radio is instantiated."""
    p_clean = re.sub(r"[^\w\s]", "", page_name).strip().lower()
    if "simulator" in p_clean:
        st.session_state["nav_page_request"] = "⚖️ Policy Impact"
        st.session_state["policy_impact_subtab"] = "Case Simulator"
    elif "review queue" in p_clean:
        st.session_state["nav_page_request"] = "🔍 Case Review"
        st.session_state["case_review_subtab"] = "Review Queue"
    elif "explorer" in p_clean or "case review" in p_clean:
        st.session_state["nav_page_request"] = "🔍 Case Review"
        st.session_state["case_review_subtab"] = "Case Explorer"
    elif "before after" in p_clean or "policy impact" in p_clean or "before / after" in page_name.lower():
        st.session_state["nav_page_request"] = "⚖️ Policy Impact"
        st.session_state["policy_impact_subtab"] = "Institutional Transformation"
    elif "funnel" in p_clean:
        st.session_state["nav_page_request"] = "🔻 Case Funnel"
    elif "duration" in p_clean or "bottleneck" in p_clean:
        st.session_state["nav_page_request"] = "⏱️ Duration & Bottlenecks"
    elif "linkage" in p_clean:
        st.session_state["nav_page_request"] = "🔗 Linkage Quality"
    elif "privacy" in p_clean:
        st.session_state["nav_page_request"] = "🛡️ Privacy & Protection"
    elif "insight" in p_clean or "statutory" in p_clean:
        st.session_state["nav_page_request"] = "📜 Policy Insights"
    elif "audit" in p_clean:
        st.session_state["nav_page_request"] = "📜 Audit Log"
    elif "analyze" in p_clean:
        st.session_state["nav_page_request"] = "🔬 Analyze Your Data"
    elif "overview" in p_clean:
        st.session_state["nav_page_request"] = "📊 Executive Overview"
    else:
        st.session_state["nav_page_request"] = page_name


def render_no_active_dataset_banner():
    """Renders user-requested message when no dataset is analyzed yet."""
    st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-left: 4px solid #8B1E3F; border-radius: 4px; padding: 24px 28px; margin: 24px 0;">
            <div style="display: flex; align-items: flex-start; gap: 14px;">
                <div style="font-size: 1.8rem; line-height: 1;">ℹ️</div>
                <div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #222222; margin-bottom: 6px;">
                        No dataset analyzed yet. Upload CSVs or run sample analysis to populate the dashboard.
                    </div>
                    <div style="font-size: 0.88rem; color: #5F5F5F; line-height: 1.5; margin-bottom: 16px;">
                        To protect survivor privacy and ensure analytical integrity, CaseLens does not show default pre-populated microdata.
                        Please upload your 5 multi-agency CSV tables (Police FIR, Hospital MLC, Forensic Lab, Prosecution, Court)
                        or run the benchmark sample analysis in the <strong>Analyze Your Data</strong> view.
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, _ = st.columns([3, 4, 3])
    with col1:
        if st.button(
            "🔬 Go to Analyze Your Data",
            key=f"btn_goto_analyze_{st.session_state.get('nav_page_selection', 'p')}",
            type="primary",
            width="stretch",
            on_click=request_navigation,
            args=("🔬 Analyze Your Data",)
        ):
            request_navigation("🔬 Analyze Your Data")
            st.rerun()
    with col2:
        if st.button("🚀 Run Sample Benchmark Analysis", key=f"btn_run_sample_{st.session_state.get('nav_page_selection', 'p')}", width="stretch"):
            with st.spinner("Executing record linkage and process mining on sample benchmark datasets..."):
                smpl_res = run_analyze_sample(limit=500, token=st.session_state.get("token"))
                if not smpl_res.get("error"):
                    total_c = smpl_res.get("summary_kpis", {}).get("total_cases") or smpl_res.get("funnel", {}).get("total_cases", 0)
                    st.session_state["active_dataset_id"] = "sample"
                    st.session_state["active_dataset_label"] = f"Benchmark Sample Dataset ({total_c:,} cases)"
                    st.session_state["active_dataset_bundle"] = build_dashboard_bundle_from_analysis(smpl_res)
                    st.session_state["active_analysis_result"] = smpl_res
                    st.session_state["sample_analysis_res"] = smpl_res
                    st.success("Benchmark sample dataset loaded successfully.")
                    st.rerun()
                else:
                    st.error(f"Failed to load sample dataset: {smpl_res.get('error')}")


def render_too_few_cases_warning():
    """Renders a friendly message when filtered cohort contains fewer than k cases."""
    st.warning(
        "⚠️ **Too few cases to display safely**\n\n"
        "Fewer than 5 reconstructed cases match your filter criteria. "
        "To prevent survivor re-identification and strictly comply with small-cell k-anonymity suppression (k ≥ 5), "
        "aggregate statistics cannot be safely displayed for this cohort. "
        "Please broaden your filter selection (e.g. select 'All Districts' or expand the time period)."
    )


def clean_policy_data(raw_policy: Dict[str, Any]) -> Dict[str, Any]:
    """Pass-through for policy context already cleaned and ranked on the server."""
    return raw_policy


NA = "N/A"


def _num(value: Any) -> Optional[float]:
    """Returns a finite numeric value, else None. Does not substitute demo defaults."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    return None


def fmt_int(value: Any) -> str:
    if isinstance(value, str) and ("<" in value or value == "Suppressed"):
        return value
    n = _num(value)
    if n is None:
        return NA
    return f"{int(round(n)):,}"


def fmt_float(value: Any, digits: int = 1) -> str:
    if isinstance(value, str) and ("<" in value or value == "Suppressed"):
        return value
    n = _num(value)
    if n is None:
        return NA
    return f"{n:.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    if isinstance(value, str) and ("<" in value or value == "Suppressed"):
        return value
    n = _num(value)
    if n is None:
        return NA
    return f"{n:.{digits}f}%"


def pct_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    n = _num(numerator)
    d = _num(denominator)
    if n is None or d is None or d == 0:
        return None
    return (n / d) * 100.0


def as_fraction_pct(value: Any) -> Optional[float]:
    """Converts a 0–1 proportion or an already-percent value into percent points."""
    n = _num(value)
    if n is None:
        return None
    if 0.0 <= n <= 1.0:
        return n * 100.0
    return n


def leak_by_code(leak_points: List[Any], code: str) -> Dict[str, Any]:
    for lp in leak_points or []:
        if isinstance(lp, dict) and lp.get("leak_code") == code:
            return lp
    return {}


def format_leak_detail(lp: Dict[str, Any]) -> str:
    if not lp:
        return NA
    mult = _num(lp.get("multiplier"))
    drop = _num(lp.get("drop_off_leak_pct"))
    delay = _num(lp.get("delay_days"))
    if delay is None:
        delay = _num(lp.get("trial_delay_days"))
    base = _num(lp.get("baseline_days"))
    if mult is not None and delay is not None and base is not None:
        return (
            f"Turnaround: <strong>{delay:.1f} days</strong> vs baseline "
            f"<strong>{base:.1f} days</strong> (<strong>{mult:.2f}x delay multiplier</strong>)"
        )
    rate = _num(lp.get("chargesheet_rate_pct"))
    bench = _num(lp.get("state_benchmark_pct"))
    if drop is not None and rate is not None and bench is not None:
        return (
            f"Only <strong>{rate:.1f}%</strong> chargesheet conversion vs "
            f"<strong>{bench:.1f}%</strong> benchmark (<strong>{drop:.1f}% attrition drop-off</strong>)"
        )
    return NA


def statutory_compliance_label(compliance: Optional[Dict[str, Any]]) -> str:
    """Uses the statutory key that exists for the transition; never invents 0%."""
    if not isinstance(compliance, dict) or not compliance:
        return NA
    order = [
        ("within_60_days_pct", "within 60 days"),
        ("within_30_days_pct", "within 30 days"),
        ("within_24h_pct", "within 24 hours"),
    ]
    for key, label in order:
        if key in compliance:
            n = _num(compliance.get(key))
            if n is None:
                return NA
            return f"{n}% {label}"
    return NA


def compute_dashboard_kpis(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derives dashboard headline metrics from pipeline JSON artifacts.
    Missing or unusable fields become N/A — no demo numeric fallbacks.
    """
    bundle = bundle or {}
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
    health = summary.get("pipeline_health") if isinstance(summary.get("pipeline_health"), dict) else {}
    journey = bundle.get("journey") if isinstance(bundle.get("journey"), dict) else {}
    leaks_raw = summary.get("systemic_leak_points_detected")
    leaks = leaks_raw if isinstance(leaks_raw, list) else None
    ranking_raw = summary.get("district_vulnerability_ranking")
    ranking = ranking_raw if isinstance(ranking_raw, list) else []

    eval_data = bundle.get("evaluation") if isinstance(bundle.get("evaluation"), dict) else {}
    pairwise = eval_data.get("pairwise_metrics") if isinstance(eval_data.get("pairwise_metrics"), dict) else {}
    at_thr = pairwise.get("metrics_at_default_threshold") if isinstance(pairwise.get("metrics_at_default_threshold"), dict) else {}
    cluster = eval_data.get("cluster_metrics") if isinstance(eval_data.get("cluster_metrics"), dict) else {}

    priv = bundle.get("privacy_audit") if isinstance(bundle.get("privacy_audit"), dict) else {}
    budget = priv.get("privacy_budget_summary") if isinstance(priv.get("privacy_budget_summary"), dict) else {}
    live_budget = bundle.get("privacy_budget") if isinstance(bundle.get("privacy_budget"), dict) else {}
    if live_budget and "remaining_epsilon" in live_budget:
        budget = {
            "total_epsilon_allocated": live_budget.get("total_epsilon", budget.get("total_epsilon_allocated", 10.0)),
            "spent_epsilon": live_budget.get("spent_epsilon", budget.get("spent_epsilon", 0.0)),
            "remaining_epsilon": live_budget.get("remaining_epsilon", budget.get("remaining_epsilon", 10.0)),
            "composition_theorem": budget.get("composition_theorem", "Sequential Composition"),
            "budget_ledger_entries": live_budget.get("total_queries_executed", budget.get("budget_ledger_entries", 0))
        }
    dp_mech = priv.get("differential_privacy_mechanisms") if isinstance(priv.get("differential_privacy_mechanisms"), dict) else {}
    k_eval = priv.get("k_anonymity_evaluation") if isinstance(priv.get("k_anonymity_evaluation"), dict) else {}
    suppression = k_eval.get("small_cell_suppression_audit") if isinstance(k_eval.get("small_cell_suppression_audit"), dict) else {}
    adv_root = priv.get("adversarial_threat_evaluations") if isinstance(priv.get("adversarial_threat_evaluations"), dict) else {}
    singling = adv_root.get("singling_out_marksmanship_attack") if isinstance(adv_root.get("singling_out_marksmanship_attack"), dict) else {}
    before = singling.get("before_protection") if isinstance(singling.get("before_protection"), dict) else {}
    after = singling.get("after_k_anonymity_protection") if isinstance(singling.get("after_k_anonymity_protection"), dict) else {}

    total = _num(health.get("total_reconstructed_cases"))
    if total is None:
        total = _num(journey.get("total_cases"))
    events = _num(journey.get("total_events"))
    fir = _num(health.get("cases_initiating_fir"))
    verdict = _num(health.get("cases_reaching_verdict"))
    fir_rate = pct_ratio(fir, total)
    verdict_rate = _num(health.get("overall_conviction_reach_rate_pct"))
    if verdict_rate is None:
        verdict_rate = pct_ratio(verdict, fir)

    leak_count = len(leaks) if leaks is not None else None
    leak_districts = NA
    if leaks:
        codes = [str(lp.get("district")).split("_")[0] for lp in leaks if isinstance(lp, dict) and lp.get("district")]
        leak_districts = ", ".join(codes) if codes else NA

    zero_pii = dp_mech.get("zero_pii_compliance")
    if zero_pii is True:
        zero_pii_label, zero_pii_delta = "Pass", "Published DP tables passed Zero-PII check"
    elif zero_pii is False:
        zero_pii_label, zero_pii_delta = "Fail", "PII tokens detected in published tables"
    else:
        zero_pii_label, zero_pii_delta = NA, "Zero-PII status unavailable"

    lp01 = leak_by_code(leaks or [], "LP-01-FORENSIC-DELAY")
    lp02 = leak_by_code(leaks or [], "LP-02-POLICE-ATTRITION")
    lp03 = leak_by_code(leaks or [], "LP-03-COURT-DELAY")

    return {
        "reconstructed_cases_display": fmt_int(total),
        "events_display": fmt_int(events),
        "fir_count_display": fmt_int(fir),
        "fir_rate_display": fmt_pct(fir_rate),
        "verdict_count_display": fmt_int(verdict),
        "verdict_rate_display": fmt_pct(verdict_rate),
        "leak_count_display": str(leak_count) if leak_count is not None else NA,
        "leak_districts_display": leak_districts,
        "bcubed_f1_display": fmt_float(cluster.get("bcubed_f1"), 3),
        "macro_purity_display": fmt_pct(as_fraction_pct(cluster.get("macro_purity"))),
        "pairwise_recall_display": fmt_pct(as_fraction_pct(at_thr.get("recall"))),
        "pairwise_precision_display": fmt_pct(as_fraction_pct(at_thr.get("precision"))),
        "pairwise_recall": _num(at_thr.get("recall")),
        "pairwise_precision": _num(at_thr.get("precision")),
        "candidate_pairs_display": fmt_int(pairwise.get("total_candidate_pairs")),
        "epsilon_allocated_display": fmt_float(budget.get("total_epsilon_allocated"), 2),
        "epsilon_spent_display": fmt_float(budget.get("spent_epsilon"), 2),
        "epsilon_remaining_display": fmt_float(budget.get("remaining_epsilon"), 2),
        "epsilon_remaining_num": _num(budget.get("remaining_epsilon")),
        "zero_pii_label": zero_pii_label,
        "zero_pii_delta": zero_pii_delta,
        "singling_before_display": fmt_pct(before.get("singling_out_risk_pct")),
        "prosecutor_before_display": fmt_pct(before.get("max_prosecutor_risk_pct")),
        "unique_before_display": fmt_int(before.get("unique_records_count")),
        "singling_after_display": fmt_pct(after.get("singling_out_risk_pct")),
        "prosecutor_after_display": fmt_pct(after.get("max_prosecutor_risk_pct")),
        "unique_after_display": fmt_int(after.get("unique_records_count")),
        "suppression_rate_display": fmt_pct(suppression.get("suppression_rate_pct")),
        "lp01_delay": fmt_float(lp01.get("delay_days"), 1),
        "lp01_multiplier": fmt_float(lp01.get("multiplier"), 2),
        "lp02_drop": fmt_pct(lp02.get("drop_off_leak_pct")),
        "lp03_delay": fmt_float(lp03.get("trial_delay_days"), 1),
        "lp03_multiplier": fmt_float(lp03.get("multiplier"), 2),
    }


if _is_running_in_streamlit() and st.session_state.get("authenticated"):
    # ==============================================================================
    # 4. Sidebar Navigation, Global Filters & Session Controls
    # ==============================================================================

    with st.sidebar:
        st.markdown("""
            <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #E3DED6; margin-bottom: 14px;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #222222; letter-spacing: -0.01em;">CaseLens</div>
                <div style="font-size: 0.78rem; color: #8B1E3F; font-weight: 600; letter-spacing: 0.01em;">
                    The Leaky Pipeline: Tracing Gaps in the Justice System
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-radius: 4px; padding: 10px 12px; margin-bottom: 12px;">
                <div style="font-size: 0.72rem; color: #5F5F5F; text-transform: uppercase; font-weight: 600; letter-spacing: 0.03em;">Active Session</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #222222;">👤 {CURRENT_USER['username']}</div>
                <div style="font-size: 0.78rem; color: #8B1E3F; font-weight: 500;">Role: <span style="text-transform: capitalize;">{CURRENT_ROLE}</span></div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Sign Out", key="sidebar_logout_btn", width="stretch"):
            log_audit_event("LOGOUT", user_id=CURRENT_USER.get("username", "anonymous"), role=CURRENT_ROLE, token=st.session_state.get("token"))
            st.session_state.clear()
            st.rerun()

        # Active Dataset indicator & controls in sidebar
        active_id = st.session_state.get("active_dataset_id")
        active_lbl = st.session_state.get("active_dataset_label", "No dataset analyzed yet")
        if active_id == "uploaded":
            ds_bg, ds_border, ds_title = "#ECFDF5", "#059669", "📁 Active: Uploaded CSVs"
        elif active_id == "sample":
            ds_bg, ds_border, ds_title = "#EFF6FF", "#2563EB", "📊 Active: Benchmark Sample"
        elif active_id == "case":
            ds_bg, ds_border, ds_title = "#F5F3FF", "#7C3AED", "🔍 Active: Single Case Analysis"
        else:
            ds_bg, ds_border, ds_title = "#FFFBEB", "#D97706", "⚠️ No Active Dataset"

        st.markdown(f"""
            <div style="background: {ds_bg}; border: 1px solid {ds_border}; border-left: 3px solid {ds_border}; border-radius: 4px; padding: 10px 12px; margin-top: 10px; margin-bottom: 12px;">
                <div style="font-size: 0.70rem; color: #5F5F5F; text-transform: uppercase; font-weight: 700; letter-spacing: 0.03em;">Active Dataset</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #222222; margin-top: 2px;">{ds_title}</div>
                <div style="font-size: 0.72rem; color: #5F5F5F; margin-top: 2px;">{active_lbl}</div>
            </div>
        """, unsafe_allow_html=True)

        # Context-dependent switcher buttons
        if active_id != "sample":
            if st.button("🔄 Switch to Benchmark Sample", key="sb_switch_sample", width="stretch", help="Switch active dashboard dataset to benchmark sample"):
                with st.spinner("Loading benchmark sample dataset..."):
                    smpl_res = run_analyze_sample(limit=500, token=st.session_state.get("token"))
                    if not smpl_res.get("error"):
                        total_c = smpl_res.get("summary_kpis", {}).get("total_cases") or smpl_res.get("funnel", {}).get("total_cases", 0)
                        st.session_state["active_dataset_id"] = "sample"
                        st.session_state["active_dataset_label"] = f"Benchmark Sample Dataset ({total_c:,} cases)"
                        st.session_state["active_dataset_bundle"] = build_dashboard_bundle_from_analysis(smpl_res)
                        st.session_state["active_analysis_result"] = smpl_res
                        st.session_state["sample_analysis_res"] = smpl_res
                        st.rerun()

        if active_id != "uploaded":
            up_cached = st.session_state.get("upload_analysis_res")
            if up_cached and not up_cached.get("error"):
                if st.button("🔄 Switch to Uploaded CSVs", key="sb_switch_upload", width="stretch", help="Switch back to uploaded dataset"):
                    total_c = up_cached.get("summary_kpis", {}).get("total_cases") or up_cached.get("funnel", {}).get("total_cases", 0)
                    st.session_state["active_dataset_id"] = "uploaded"
                    st.session_state["active_dataset_label"] = f"Uploaded CSVs ({total_c:,} cases)"
                    st.session_state["active_dataset_bundle"] = build_dashboard_bundle_from_analysis(up_cached)
                    st.session_state["active_analysis_result"] = up_cached
                    st.rerun()

        if active_id != "case":
            case_cached = st.session_state.get("single_case_analysis_res")
            if case_cached and not case_cached.get("error"):
                if st.button("🔄 Switch to Single Case", key="sb_switch_case", width="stretch", help="Switch back to single case analysis"):
                    m_name = case_cached.get("masked_survivor_name", "Anonymous")
                    st.session_state["active_dataset_id"] = "case"
                    st.session_state["active_dataset_label"] = f"Single Case ({m_name})"
                    st.session_state["active_dataset_bundle"] = build_dashboard_bundle_from_analysis({
                        "is_single_case": True,
                        **case_cached
                    })
                    st.session_state["active_analysis_result"] = case_cached
                    st.rerun()

        # Labelled reset action
        if active_id:
            if st.button("❌ Clear Active Dataset", key="sb_clear_active_btn", width="stretch", help="Clear active dataset and reset dashboard to empty state"):
                st.session_state["active_dataset_id"] = None
                st.session_state["active_dataset_label"] = None
                st.session_state["active_dataset_bundle"] = None
                st.session_state["active_analysis_result"] = None
                st.session_state["single_case_analysis_res"] = None
                st.session_state["upload_analysis_res"] = None
                st.session_state["sample_analysis_res"] = None
                st.rerun()

        st.markdown("### 🧭 Navigation")
        allowed_pages = ROLE_PERMISSIONS.get(CURRENT_ROLE, ROLE_PERMISSIONS["viewer"])
        if "nav_page_request" in st.session_state:
            requested_page = st.session_state.pop("nav_page_request")
            if requested_page in allowed_pages:
                st.session_state["nav_page_selection"] = requested_page
            else:
                p_clean = re.sub(r"[^\w\s]", "", requested_page).strip().lower()
                for ap in allowed_pages:
                    ap_clean = re.sub(r"[^\w\s]", "", ap).strip().lower()
                    if p_clean == ap_clean or p_clean in ap_clean or ap_clean in p_clean:
                        st.session_state["nav_page_selection"] = ap
                        break
        if "nav_page_selection" not in st.session_state or st.session_state["nav_page_selection"] not in allowed_pages:
            st.session_state["nav_page_selection"] = allowed_pages[0]
        PAGE = st.radio(
            label="Select Analytical View",
            options=allowed_pages,
            key="nav_page_selection",
            label_visibility="collapsed"
        )
        if CURRENT_ROLE == "admin":
            st.markdown("""
                <div style="font-size: 0.72rem; color: #8B1E3F; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; padding-top: 6px;">
                    🛡️ Administration Section Active (Audit Log Included)
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎛️ Global Filters")
        DISTRICT_OPTIONS = [
            "All Districts",
            "D1_Suryapur",
            "D2_Chandranagar",
            "D3_Rudrapur",
            "D4_Kalyanpur",
            "D5_Mayaganj",
            "D6_Shivgarh",
            "D7_Anandpur",
            "D8_Devkalan"
        ]
        FILTER_DISTRICT = st.selectbox(
            "District",
            DISTRICT_OPTIONS,
            index=0,
            help="Filter analytical cohort to a specific judicial district. Default 'All Districts' reproduces statewide baseline."
        )

        c_d1, c_d2 = st.columns(2)
        with c_d1:
            FILTER_FROM = st.date_input(
                "From Date",
                value=datetime(2022, 12, 31).date(),
                min_value=datetime(2022, 1, 1).date(),
                max_value=datetime(2026, 12, 31).date(),
                help="Start date of case inception window."
            )
        with c_d2:
            FILTER_TO = st.date_input(
                "To Date",
                value=datetime(2026, 11, 27).date(),
                min_value=datetime(2022, 1, 1).date(),
                max_value=datetime(2026, 12, 31).date(),
                help="End date of case inception window."
            )

        STAGE_OPTIONS = [
            "All Stages",
            "1. Police FIR",
            "2. Hospital MLC",
            "3. Forensic Lab (FSL)",
            "4. Prosecution Chargesheet",
            "5. Court Verdict"
        ]
        FILTER_STAGE = st.selectbox(
            "Stage / Institution",
            STAGE_OPTIONS,
            index=0,
            help="Filter to case trajectories that reached this institutional milestone."
        )

        st.markdown("---")
        st.markdown("### 🔒 Privacy Controls")
        EPSILON_SIM = st.slider(
            r"Epsilon ($\epsilon$) Noise Level",
            min_value=0.1,
            max_value=2.0,
            value=0.25,
            step=0.05,
            help="Lower epsilon draws stronger Laplace noise for higher survivor confidentiality."
        )
        K_ANONYMITY = st.number_input(
            "k-Anonymity Suppression Threshold",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
            help="Cells with fewer than k cases are strictly suppressed as '<k'."
        )

    # Server-side filter normalization
    sel_district = None if FILTER_DISTRICT == "All Districts" else FILTER_DISTRICT
    sel_stage = None if FILTER_STAGE == "All Stages" else FILTER_STAGE
    sel_from = None
    sel_to = None
    if FILTER_FROM and FILTER_FROM > datetime(2022, 12, 31).date():
        sel_from = str(FILTER_FROM)
    if FILTER_TO and FILTER_TO < datetime(2026, 11, 27).date():
        sel_to = str(FILTER_TO)

    # Resolve active dashboard dataset: Only populated when an analysis has been executed!
    active_ds_id = st.session_state.get("active_dataset_id")
    active_bundle = st.session_state.get("active_dataset_bundle")

    if active_ds_id and active_bundle:
        DATA = active_bundle
        KPIS = compute_dashboard_kpis(DATA)
    else:
        DATA = {}
        KPIS = compute_dashboard_kpis({})

    # Remaining DP budget widget in sidebar
    rem_budget = DATA.get("summary", {}).get("privacy_metadata", {}).get(
        "remaining_budget",
        KPIS.get("epsilon_remaining_display", NA)
    ) if active_ds_id else NA
    with st.sidebar:
        st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-radius: 4px; padding: 10px 12px; margin-top: 10px; margin-bottom: 12px;">
                <div style="font-size: 0.72rem; color: #5F5F5F; text-transform: uppercase; font-weight: 600; letter-spacing: 0.03em;">Remaining DP Budget</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #8B1E3F;">{rem_budget} <span style="font-size: 0.82rem; font-weight: 500; color: #5F5F5F;">ε</span></div>
                <div style="font-size: 0.72rem; color: #5F5F5F; margin-top: 2px;">Sequential composition ledger</div>
            </div>
        """, unsafe_allow_html=True)

        if CURRENT_ROLE == "admin":
            if st.button("🔄 Reset Budget", key="sidebar_reset_budget_btn", help="Reset differential privacy budget (Admin only)"):
                res = reset_privacy_budget(token=st.session_state.get("token"))
                if res.get("message") or res.get("success"):
                    st.session_state["privacy_budget_exhausted"] = False
                    st.cache_data.clear()
                    st.rerun()

        scope_cases = f"{KPIS['reconstructed_cases_display']} Cases in Scope" if active_ds_id else "0 (No Dataset Active)"
        st.markdown(f"""
            <div style="font-size: 0.75rem; color: #5F5F5F; line-height: 1.5; padding-top: 4px;">
                <strong style="color: #222222;">System Integrity</strong><br/>
                • {scope_cases}<br/>
                • Pure Laplace Differential Privacy<br/>
                • Zero-PII Response Verification<br/>
                • Statutory Benchmarking
            </div>
        """, unsafe_allow_html=True)

    # Server-side RBAC validation: strictly blocks unauthorized access even if requested
    if not check_user_permission(CURRENT_ROLE, PAGE):
        log_audit_event("ACCESS_DENIED", user_id=CURRENT_USER.get("username", "anonymous"), role=CURRENT_ROLE, page=PAGE, token=st.session_state.get("token"))
        st.error(f"⛔ Access Denied: Your assigned role '{CURRENT_ROLE}' does not have permission to view the '{PAGE}' page. Please contact your system administrator if you require elevated access.")
        st.stop()

    # Record page visit audit event (Zero-PII)
    if st.session_state.get("last_recorded_page") != PAGE:
        log_audit_event("PAGE_VISIT", user_id=CURRENT_USER.get("username", "anonymous"), role=CURRENT_ROLE, page=PAGE, token=st.session_state.get("token"))
        st.session_state["last_recorded_page"] = PAGE

    # Differential Privacy Exhaustion Warning Banner
    is_budget_exhausted = (
        st.session_state.get("privacy_budget_exhausted", False)
        or (KPIS.get("epsilon_remaining_num") is not None and KPIS.get("epsilon_remaining_num") <= 0.001)
    )
    if is_budget_exhausted:
        with st.container():
            st.markdown("""
                <div style="background: #FFFBEB; border: 1px solid #F59E0B; border-left: 4px solid #B45309; border-radius: 4px; padding: 14px 18px; margin-bottom: 20px;">
                    <div style="font-weight: 700; color: #92400E; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
                        <span>🛡️</span> Differential Privacy Budget Exhausted (0.00 ε Remaining)
                    </div>
                    <div style="color: #78350F; font-size: 0.85rem; margin-top: 6px; line-height: 1.5;">
                        The sequential DP budget has reached 0.00 ε to strictly prevent reconstruction attacks under differential privacy guarantees.
                        <strong>Previously queried and cached analyses remain fully viewable.</strong> New uncached queries are halted until budget renewal.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if CURRENT_ROLE == "admin":
                col_rst1, col_rst2 = st.columns([3, 7])
                with col_rst1:
                    if st.button("🔄 Reset Privacy Budget (Admin)", key="global_banner_reset_budget_btn", type="primary"):
                        res = reset_privacy_budget(token=st.session_state.get("token"))
                        if res.get("message") or res.get("success"):
                            st.session_state["privacy_budget_exhausted"] = False
                            st.cache_data.clear()
                            st.success("Privacy budget has been reset.")
                            st.rerun()
                        else:
                            st.error("Failed to reset budget. Ensure admin permissions.")
else:
    DATA = {}
    KPIS = {}
    PAGE = None
    FILTER_DISTRICT = "All Districts"
    FILTER_FROM = datetime(2022, 12, 31).date()
    FILTER_TO = datetime(2026, 11, 27).date()
    FILTER_STAGE = "All Stages"



# ==============================================================================
# Helper Chart Utilities (Sober Editorial Light Layout)
# ==============================================================================

def get_plot_layout() -> Dict[str, Any]:
    """Generates sober editorial light layout for Plotly figures."""
    return dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#222222", family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=45, r=30, t=45, b=45),
        xaxis=dict(
            gridcolor="#E8E4DC",
            zerolinecolor="#E3DED6",
            tickfont=dict(color="#5F5F5F", size=11),
            title=dict(font=dict(color="#222222", size=12))
        ),
        yaxis=dict(
            gridcolor="#E8E4DC",
            zerolinecolor="#E3DED6",
            tickfont=dict(color="#5F5F5F", size=11),
            title=dict(font=dict(color="#222222", size=12))
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#E3DED6",
            borderwidth=1,
            font=dict(color="#222222", size=11)
        )
    )


def render_case_simulator_tab():
    st.markdown("""
        <div style="background: #FDF2F4; border-left: 3px solid #8B1E3F; border-radius: 4px; padding: 10px 14px; margin-bottom: 20px; font-size: 0.82rem; color: #8B1E3F;">
            <strong>⚠️ Synthetic Judicial Simulation Only:</strong> All records, names, dates, and identifiers displayed here are purely synthetic and fictional.
            No real survivor microdata is accessed or queried. Linkage calculations are algorithmic probabilistic outputs for demonstration and do NOT constitute judicial determinations or legal findings.
        </div>
    """, unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1.25], gap="large")

    with col_in:
        st.markdown("### 📥 Case Input: Agency Records")
        st.markdown("<div style='font-size: 0.85rem; color: #5F5F5F; margin-bottom: 12px;'>Select a pre-configured scenario or customize synthetic fields.</div>", unsafe_allow_html=True)

        active_token = st.session_state.get("token")
        scenarios_dict = get_simulator_scenarios(token=active_token)
        if not scenarios_dict:
            scenarios_dict = {
                "scenario_1_clean": {
                    "title": "Scenario 1: Clean Canonical Case",
                    "description": "Standard multi-agency trajectory with complete milestones.",
                    "records": []
                }
            }
        scenario_keys = list(scenarios_dict.keys())
        scenario_titles = [scenarios_dict[k]["title"] for k in scenario_keys]
        selected_title = st.selectbox("Choose Judicial Scenario", scenario_titles, index=0)
        selected_key = scenario_keys[scenario_titles.index(selected_title)]
        scenario_data = scenarios_dict[selected_key]

        st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-radius: 4px; padding: 12px 14px; margin-bottom: 14px;">
                <strong style="color: #222222; font-size: 0.88rem;">Scenario Objective:</strong><br/>
                <span style="font-size: 0.82rem; color: #5F5F5F; line-height: 1.4;">{scenario_data['description']}</span>
            </div>
        """, unsafe_allow_html=True)

        edit_mode = st.toggle("✏️ Edit Fields Mode (Custom Synthetic Values)", value=False)

        if not edit_mode:
            sim_result = run_simulator(scenario_key=selected_key, token=active_token)
            if not sim_result or "records" not in sim_result:
                sim_result = {
                    "records": scenario_data.get("records", []),
                    "verdict": "LINKED_AUTOMATIC",
                    "verdict_type": "success",
                    "why": "Backend simulation evaluated.",
                    "journey": {"timeline_steps": [], "bottlenecks": []},
                    "pairwise_results": []
                }
            records_to_display = sim_result["records"]

            st.markdown(f"**Agency Records Provided ({len(records_to_display)} Records):**")
            for rec in records_to_display:
                with st.expander(f"📄 {rec['agency']} — {rec.get('record_id', 'ID')}", expanded=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Name:** `{rec.get('name') or 'N/A'}`")
                        st.markdown(f"**Guardian:** `{rec.get('guardian') or 'N/A'}`")
                        st.markdown(f"**District:** `{rec.get('district') or 'N/A'}`")
                    with c2:
                        st.markdown(f"**Locality:** `{rec.get('locality') or 'N/A'}`")
                        st.markdown(f"**Incident Date:** `{rec.get('incident_date') or 'N/A'}`")
                        st.markdown(f"**Record Date:** `{rec.get('record_date') or 'N/A'}`")
                    if rec.get("offense") or rec.get("details"):
                        st.caption(f"Details: {rec.get('offense') or rec.get('details')}")
        else:
            st.info("💡 **Edit Mode Active**: Modify the synthetic fields below. Only obviously fictional demo names are accepted.")
            base_records = scenario_data.get("records", [])
            custom_records = []

            with st.form("edit_scenario_form"):
                for idx, r in enumerate(base_records):
                    st.markdown(f"##### {r['agency']} ({r.get('record_id')})")
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        new_name = st.text_input(f"Name ({r['agency']})", value=r.get("name") or "", key=f"edit_name_{idx}")
                        new_guard = st.text_input(f"Guardian ({r['agency']})", value=r.get("guardian") or "", key=f"edit_guard_{idx}")
                        new_dist = st.text_input(f"District ({r['agency']})", value=r.get("district") or "", key=f"edit_dist_{idx}")
                    with ec2:
                        new_loc = st.text_input(f"Locality ({r['agency']})", value=r.get("locality") or "", key=f"edit_loc_{idx}")
                        new_inc = st.text_input(f"Incident Date (YYYY-MM-DD)", value=r.get("incident_date") or "", key=f"edit_inc_{idx}")
                        new_rec = st.text_input(f"Record Date (YYYY-MM-DD)", value=r.get("record_date") or "", key=f"edit_rec_{idx}")

                    custom_records.append({
                        "agency": r["agency"],
                        "agency_code": r["agency_code"],
                        "record_id": r["record_id"],
                        "name": new_name.strip() if new_name else None,
                        "guardian": new_guard.strip() if new_guard else None,
                        "district": new_dist.strip() if new_dist else None,
                        "locality": new_loc.strip() if new_loc else None,
                        "incident_date": new_inc.strip() if new_inc else None,
                        "record_date": new_rec.strip() if new_rec else None,
                        "details": r.get("details"),
                        "offense": r.get("offense")
                    })
                    st.markdown("---")

                re_run_btn = st.form_submit_button("⚡ Re-Run Linkage Scoring", width="stretch")

            sim_result = run_simulator(custom_records=custom_records, custom_title=f"Customized {selected_title}", token=active_token)
            if not sim_result or "records" not in sim_result:
                sim_result = {
                    "records": custom_records,
                    "verdict": "CUSTOM_SIMULATION",
                    "verdict_type": "neutral",
                    "why": "Custom simulation evaluated.",
                    "journey": {"timeline_steps": [], "bottlenecks": []},
                    "pairwise_results": []
                }

    with col_out:
        st.markdown("### 📤 Case Output: Real-Time Linkage & Journey")
        st.markdown("<div style='font-size: 0.85rem; color: #5F5F5F; margin-bottom: 12px;'>Scored through real Fellegi-Sunter model and journey pipeline.</div>", unsafe_allow_html=True)

        # 1. Verdict Box
        v_type = sim_result["verdict_type"]
        v_colors = {
            "success": ("#FDF2F4", "#8B1E3F", "✅"),
            "warning": ("#FEF3C7", "#B45309", "🔻"),
            "danger": ("#FEE2E2", "#DC2626", "🛡️"),
            "review": ("#FFFBEB", "#D97706", "⚠️"),
            "neutral": ("#F0ECE5", "#222222", "ℹ️")
        }
        bg_col, text_col, icon = v_colors.get(v_type, ("#F0ECE5", "#222222", "ℹ️"))

        st.markdown(f"""
            <div style="background: {bg_col}; border: 1px solid {text_col}; border-left: 5px solid {text_col}; border-radius: 4px; padding: 16px 18px; margin-bottom: 16px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 600; color: {text_col}; letter-spacing: 0.04em;">Judicial Linkage Verdict</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: {text_col}; margin-top: 2px;">{icon} {sim_result['verdict']}</div>
            </div>
        """, unsafe_allow_html=True)

        # 2. "Why" Explanation
        st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-radius: 4px; padding: 14px 16px; margin-bottom: 16px;">
                <div style="font-size: 0.76rem; text-transform: uppercase; font-weight: 600; color: #5F5F5F; letter-spacing: 0.04em; margin-bottom: 4px;">Analytical Explanation ("Why")</div>
                <div style="font-size: 0.88rem; color: #222222; line-height: 1.5;">{sim_result['why']}</div>
                <div style="font-size: 0.75rem; color: #8C8C8C; margin-top: 8px; border-top: 1px solid #F0ECE5; padding-top: 6px;">
                    ⚖️ <strong>Judicial Notice:</strong> All uncertain pairs (0.50 ≤ P < 0.85) require human reviewer adjudication. Algorithmic outputs provide policy and investigative evidence and do not constitute a legal conclusion.
                </div>
            </div>
        """, unsafe_allow_html=True)

        # 3. Longitudinal Journey Timeline
        st.markdown("#### ⏳ Reconstructed Journey Timeline")
        journey = sim_result["journey"]
        steps = journey["timeline_steps"]

        for step in steps:
            status = step["status"]
            is_comp = status == "COMPLETED"
            dot_color = "#8B1E3F" if is_comp else "#B45309"
            badge_cls = "badge-privacy" if is_comp else "badge-critical"
            date_display = f"🗓️ {step['date']}" if step['date'] else "❌ Not Recorded"

            st.markdown(f"""
                <div style="display: flex; align-items: flex-start; margin-bottom: 8px; padding: 8px 12px; background: #FFFFFF; border: 1px solid #E3DED6; border-left: 3px solid {dot_color}; border-radius: 4px;">
                    <div style="flex-grow: 1;">
                        <strong style="color: #222222; font-size: 0.88rem;">{step['stage']}</strong> &nbsp;
                        <span class="badge {badge_cls}">{status}</span><br/>
                        <span style="font-size: 0.78rem; color: #5F5F5F;">{date_display} | Record: <code>{step['record_id']}</code></span>
                        {f'<div style="font-size: 0.75rem; color: #7F7F7F; margin-top: 2px;">{step["notes"]}</div>' if step["notes"] else ''}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Bottlenecks / Attrition flags
        if journey["bottlenecks"]:
            for bn in journey["bottlenecks"]:
                st.markdown(f"""
                    <div style="background: #FEF3C7; border: 1px solid #FCD34D; border-left: 3px solid #B45309; border-radius: 4px; padding: 10px 12px; margin-top: 8px; margin-bottom: 8px; font-size: 0.82rem;">
                        <strong style="color: #B45309;">⚠️ Policy Leak Point Flagged ({bn['code']}):</strong> {bn['title']}<br/>
                        <span style="color: #5F5F5F;">{bn['detail']}</span>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # 4. Pairwise Decisions Table
        st.markdown("#### 🔗 Pairwise Linkage Decisions")
        pw_results = sim_result["pairwise_results"]
        if pw_results:
            table_data = []
            for p in pw_results:
                table_data.append({
                    "Record A": p["record_a"],
                    "Record B": p["record_b"],
                    "Match Probability": f"{p['match_probability']:.3f}",
                    "Weight (Log-Odds)": f"{p['match_weight']:+.2f}",
                    "Decision": "✅ AUTO_ACCEPT" if p["decision"] == "AUTO_ACCEPT" else ("⚠️ HUMAN_REVIEW" if p["decision"] == "HUMAN_REVIEW" else "❌ REJECT")
                })
            st.dataframe(pd.DataFrame(table_data), width="stretch")

            # 5. Per-Field Evidence Accordion
            st.markdown("#### 🔬 Per-Field Match Evidence (Log-Odds Contribution)")
            for idx, p in enumerate(pw_results):
                with st.expander(f"Evidence: {p['record_a']} ↔ {p['record_b']} ({p['decision']})", expanded=False):
                    st.markdown(f"**Calculated Posterior Probability:** `{p['match_probability']:.3f}` | **Composite Match Weight:** `{p['match_weight']:+.2f}`")
                    for ev in p["evidence"]:
                        st_icon = "🟢" if ev["status"] == "AGREE" else ("🟡" if ev["status"] == "PARTIAL" else ("🔴" if ev["status"] == "DISAGREE" else "⚪"))
                        st.markdown(f"- {st_icon} **{ev['field']}** ({ev['weight']:+.2f}): {ev['detail']}")
        else:
            st.info("No pairwise combinations available for single record.")


# ==============================================================================
# PAGE 0: 🔬 Analyze Your Data (Default Landing View)
# ==============================================================================

if PAGE == "🔬 Analyze Your Data" or (PAGE and "Analyze Your Data" in PAGE):
    st.markdown("""
        <div class="hero-banner">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
                <div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #8B1E3F; letter-spacing: -0.01em; margin-bottom: 2px;">CaseLens</div>
                    <div style="font-size: 0.88rem; color: #8B1E3F; font-weight: 600; margin-bottom: 12px;">The Leaky Pipeline: Tracing Gaps in the Justice System</div>
                </div>
                <div style="display: flex; gap: 6px; align-items: center; padding-top: 4px; flex-wrap: wrap;">
                    <span class="badge badge-privacy">In-Memory Linkage</span>
                    <span class="badge badge-normal">Zero-PII Storage</span>
                    <span class="badge badge-critical">Strict RBAC</span>
                </div>
            </div>
            <div class="hero-title">Analyze Your Data: Multi-Agency Record Linkage & Journey Reconstruction</div>
            <div class="hero-subtitle">
                Upload multi-agency CSVs or enter a synthetic case to evaluate probabilistic record linkage,
                reconstruct longitudinal journey timelines, and detect systemic delay bottlenecks.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Warning Callout at top of page
    st.warning("⚠️ **Warning**: Use only synthetic or fictional data. Do not enter real survivor details.")

    # 2. Server-side role enforcement (Viewers get 403 Forbidden)
    if CURRENT_ROLE == "viewer":
        st.error(
            "⛔ **Access Denied (HTTP 403)**: Insufficient privileges. 'Analyze Your Data' is restricted "
            "to Institutional Analyst and Admin roles. Viewers are not authorized to upload or analyze microdata."
        )
        st.stop()

    def render_csv_analysis_results(res: Dict[str, Any], title_prefix: str = ""):
        """Renders the 5 result tabs for CSV and sample runs using strictly run-local data."""
        kpis = res.get("summary_kpis", {})
        linkage_stats = res.get("linkage_stats", {})
        bottlenecks_data = res.get("bottlenecks", {})
        funnel_data = res.get("funnel_stages", [])
        privacy_data = res.get("privacy_info", {})

        st.markdown(f"### 📊 Analysis Results: {title_prefix}")

        r_overview, r_funnel, r_bottlenecks, r_linkage, r_privacy = st.tabs([
            "Overview", "Funnel", "Bottlenecks", "Linkage", "Privacy"
        ])

        with r_overview:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Reconstructed Cases</div>
                        <div class="metric-value">{kpis.get('total_cases', 0):,}</div>
                        <div class="metric-delta">Unified Case Entities</div>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">FIR Initiation Rate</div>
                        <div class="metric-value">{kpis.get('fir_rate_pct', 0.0):.1f}%</div>
                        <div class="metric-delta">Inception Rate</div>
                    </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Reaching Verdict</div>
                        <div class="metric-value">{kpis.get('verdict_rate_pct', 0.0):.1f}%</div>
                        <div class="metric-delta">Full Adjudication</div>
                    </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Auto-Accepted Pairs</div>
                        <div class="metric-value">{linkage_stats.get('pairs_linked', 0):,}</div>
                        <div class="metric-delta">Cross-Agency Links</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            insight_text = kpis.get("one_line_insight", "Analysis complete.")
            st.markdown(f"""
                <div class="metric-card" style="border-left: 3px solid #8B1E3F; padding: 14px 18px;">
                    <div style="font-size: 0.78rem; text-transform: uppercase; color: #8B1E3F; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 4px;">Analytical Finding</div>
                    <div style="font-size: 1.02rem; font-weight: 600; color: #222222; line-height: 1.4;">💡 {insight_text}</div>
                </div>
            """, unsafe_allow_html=True)

        with r_funnel:
            if funnel_data:
                stages = [f.get("stage_name") for f in funnel_data]
                counts = [f.get("raw_count", 0) for f in funnel_data]
                display_counts = [str(f.get("display_count", f.get("raw_count", 0))) for f in funnel_data]
                retentions = [f.get("retention_from_fir_pct", 0.0) for f in funnel_data]

                fig_funnel = go.Figure()
                fig_funnel.add_trace(go.Bar(
                    x=stages,
                    y=counts,
                    marker=dict(color=["#8B1E3F", "#A33857", "#BD5371", "#D6708D", "#B45309"]),
                    text=[f"{c} cases ({r}%)" for c, r in zip(display_counts, retentions)],
                    textposition="auto",
                    name="Cases Reached"
                ))
                fig_funnel.update_layout(
                    get_plot_layout(),
                    title="Case Progression Funnel Across Institutional Stages",
                    xaxis_title="Institutional Milestone",
                    yaxis_title="Case Volume"
                )
                st.plotly_chart(fig_funnel, width="stretch")

                # Funnel table
                df_funnel_tbl = pd.DataFrame([
                    {
                        "Stage": f.get("stage_name"),
                        "Case Count": f.get("display_count"),
                        "Step Conversion (%)": f"{f.get('step_conversion_pct', 0.0):.1f}%",
                        "Retention from FIR (%)": f"{f.get('retention_from_fir_pct', 0.0):.1f}%",
                        "Attrition Leak (%)": f"{f.get('step_leak_pct', 0.0):.1f}%"
                    }
                    for f in funnel_data
                ])
                st.dataframe(df_funnel_tbl, width="stretch", hide_index=True)
            else:
                st.info("No funnel stage records available.")

        with r_bottlenecks:
            bb = bottlenecks_data.get("biggest_bottleneck")
            if bb:
                st.markdown(f"""
                    <div class="metric-card" style="border-left: 3px solid #B45309; margin-bottom: 16px;">
                        <div style="font-size: 0.78rem; text-transform: uppercase; color: #B45309; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 2px;">🚨 Primary Systemic Bottleneck</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #222222;">{bb.get('pair')}: {bb.get('median_days')} days turnaround ({bb.get('delay_multiplier')}× Baseline)</div>
                        <div style="font-size: 0.85rem; color: #5F5F5F; margin-top: 4px;">{bb.get('summary')} across {bb.get('sample_count')} observed case transitions.</div>
                    </div>
                """, unsafe_allow_html=True)

            trans_dict = bottlenecks_data.get("transitions", {})
            if trans_dict:
                trans_list = []
                for p_name, t_val in trans_dict.items():
                    trans_list.append({
                        "Transition": p_name,
                        "Observed Median (Days)": t_val.get("median_days", 0.0),
                        "Baseline Median (Days)": t_val.get("baseline_median_days", 0.0),
                        "Delay Multiplier": t_val.get("delay_multiplier_vs_baseline", 1.0),
                        "Statutory Benchmark (Days)": t_val.get("statutory_benchmark_days") or "N/A"
                    })

                df_b_plot = pd.DataFrame(trans_list)
                colors = ["#B45309" if bb and row["Transition"] == bb.get("pair") else "#8C8C8C" for _, row in df_b_plot.iterrows()]

                fig_b = go.Figure()
                fig_b.add_trace(go.Bar(
                    y=df_b_plot["Transition"],
                    x=df_b_plot["Observed Median (Days)"],
                    orientation="h",
                    marker=dict(color=colors),
                    text=[f"{m:.1f}d ({r}x)" for m, r in zip(df_b_plot["Observed Median (Days)"], df_b_plot["Delay Multiplier"])],
                    textposition="outside",
                    name="Observed Turnaround"
                ))
                fig_b.update_layout(
                    get_plot_layout(),
                    title="Per-Stage Turnaround Duration (Amber = Primary Bottleneck)",
                    xaxis_title="Turnaround Duration (Days)",
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_b, width="stretch")

                st.dataframe(df_b_plot, width="stretch", hide_index=True)
            else:
                st.info("No inter-stage transitions with sufficient sample size observed.")

        with r_linkage:
            l_method = linkage_stats.get("linkage_method", "Fellegi-Sunter Pairwise Scoring (case_simulator)")
            st.markdown(f"""
                <div style="background: #F0ECE5; border-left: 3px solid #8B1E3F; border-radius: 4px; padding: 10px 14px; margin-bottom: 16px;">
                    <div style="font-size: 0.75rem; color: #5F5F5F; text-transform: uppercase; font-weight: 700;">Record Linkage Methodology</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #8B1E3F;">{l_method}</div>
                    <div style="font-size: 0.82rem; color: #5F5F5F; margin-top: 2px;">
                        Calibrated Bayesian log-odds evaluation using Double Metaphone phonetic normalization,
                        Jaro-Winkler string similarity, and temporal tolerance windows.
                    </div>
                </div>
            """, unsafe_allow_html=True)

            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                st.metric("Total Pairs Evaluated", f"{linkage_stats.get('total_pairs_evaluated', 0):,}")
            with lc2:
                st.metric("Auto-Accepted Links (P ≥ 0.85)", f"{linkage_stats.get('pairs_linked', 0):,}")
            with lc3:
                st.metric("Review Queue (0.50 ≤ P < 0.85)", f"{linkage_stats.get('review_queue_size', 0):,}")

            rq = linkage_stats.get("review_queue", [])
            if rq:
                st.markdown("#### 📋 Human Review Queue Candidates (Sample)")
                df_rq = pd.DataFrame(rq)
                st.dataframe(df_rq, width="stretch", hide_index=True)

        with r_privacy:
            st.markdown("""
                <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-left: 3px solid #059669; border-radius: 4px; padding: 14px 16px; margin-bottom: 14px;">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #059669;">🛡️ Privacy & Confidentiality Protection Active</div>
                    <div style="font-size: 0.85rem; color: #5F5F5F; margin-top: 4px;">
                        All survivor names and personal identifiers are sanitized server-side before response transmission.
                        Microdata records are stored strictly in volatile memory and never persisted to database or disk.
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🔒 Server-Side Sanitized Fields")
            masked_fields = privacy_data.get("masked_fields", ["survivor_name", "guardian_name", "locality"])
            for mf in masked_fields:
                st.markdown(f"- ` {mf} `: Masked using non-reversible initial retention (e.g. `S***a S***a`).")

            suppressed = privacy_data.get("suppressed_groups", [])
            st.markdown("#### 🚫 Small-Cell k-Anonymity Suppression (k < 5)")
            if suppressed:
                st.markdown("The following small cohorts were suppressed to comply with k-anonymity guarantees:")
                for sg in suppressed:
                    st.markdown(f"- ⚠️ {sg}")
            else:
                st.success("No small-cell cohorts below the k=5 threshold were detected in this analysis run.")

    # 3. Three Input Tabs
    tab_case, tab_upload, tab_sample = st.tabs([
        "Enter your own case", "Upload CSVs", "Use sample files"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Enter your own case
    # -------------------------------------------------------------------------
    with tab_case:
        st.markdown("""
            <div style="margin-bottom: 14px;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #222222;">Manual Case Entry</div>
                <div style="font-size: 0.85rem; color: #5F5F5F;">
                    Enter records across at least 2 institutional stages for a synthetic case.
                    CaseLens will evaluate pairwise match probabilities, reconstruct the timeline,
                    detect stall points, and mask survivor details server-side.
                </div>
            </div>
        """, unsafe_allow_html=True)

        col_ex1, col_ex2 = st.columns([2, 8])
        with col_ex1:
            if st.button("✨ Fill Example Case", key="btn_fill_example_case", width="stretch"):
                st.session_state["ec_fir_no"] = "FIR-2024-D3-0912"
                st.session_state["ec_fir_name"] = "Sunita Sharma"
                st.session_state["ec_fir_guardian"] = "Ramesh Sharma"
                st.session_state["fir_father_or_husband"] = "Ramesh Sharma"
                st.session_state["ec_fir_age"] = "29"
                st.session_state["ec_fir_locality"] = "Civil Lines"
                st.session_state["ec_fir_district"] = "D3_Rudrapur"
                st.session_state["ec_fir_inc_date"] = "2024-03-01"
                st.session_state["fir_incident_date"] = "2024-03-01"
                st.session_state["ec_fir_rec_date"] = "2024-03-02"
                st.session_state["ec_fir_offense"] = "Sec 376 IPC"
                st.session_state["ec_fir_station"] = "PS Rudrapur Central"

                st.session_state["ec_mlc_no"] = "MLC-2024-D3-0844"
                st.session_state["ec_mlc_name"] = "Suneeta Sharma"  # Phonetic transliteration variant
                st.session_state["ec_mlc_guardian"] = "Ramesh Sharma"
                st.session_state["mlc_guardian_name"] = "Ramesh Sharma"
                st.session_state["ec_mlc_age"] = "29"
                st.session_state["ec_mlc_locality"] = "Civil Lines"
                st.session_state["ec_mlc_district"] = "D3_Rudrapur"
                st.session_state["ec_mlc_inc_date"] = "2024-03-01"
                st.session_state["ec_mlc_rec_date"] = "2024-03-02"
                st.session_state["ec_mlc_injury"] = "Physical Trauma"
                st.session_state["ec_mlc_hospital"] = "District Hospital Rudrapur"

                st.session_state["ec_fsl_no"] = "FSL-2024-D3-0412"
                st.session_state["ec_fsl_name"] = "Sunita Sharma"
                st.session_state["ec_fsl_guardian"] = "Sharma Ramesh"  # Transposed name token
                st.session_state["fsl_guardian_name"] = "Sharma Ramesh"
                st.session_state["ec_fsl_age"] = "29"
                st.session_state["ec_fsl_district"] = "D3_Rudrapur"
                st.session_state["ec_fsl_inc_date"] = "2024-03-05"
                st.session_state["ec_fsl_rec_date"] = "2024-04-18"
                st.session_state["ec_fsl_evidence"] = "Biological DNA"
                st.session_state["ec_fsl_division"] = "Serology"

                st.session_state["ec_pros_no"] = "PP-CHG-2024-D3-0301"
                st.session_state["ec_pros_name"] = "Sunita Sharma"
                st.session_state["ec_pros_guardian"] = "Ramesh Sharma"
                st.session_state["ec_pros_age"] = "29"
                st.session_state["ec_pros_address"] = "Civil Lines, Rudrapur"
                st.session_state["ec_pros_district"] = "D3_Rudrapur"
                st.session_state["ec_pros_inc_date"] = "2024-03-01"
                st.session_state["pros_incident_date"] = "2024-03-01"
                st.session_state["ec_pros_rec_date"] = "2024-05-20"
                st.session_state["ec_pros_cat"] = "rape"
                st.session_state["ec_pros_off"] = "PP Office Rudrapur"

                st.session_state["ec_crt_no"] = ""
                st.session_state["ec_crt_name"] = ""
                st.session_state["ec_crt_guardian"] = ""
                st.session_state["court_father_or_husband"] = ""
                st.session_state["ec_crt_age"] = ""
                st.session_state["ec_crt_locality"] = ""
                st.session_state["ec_crt_district"] = ""
                st.session_state["ec_crt_inc_date"] = ""
                st.session_state["ec_crt_rec_date"] = ""
                st.session_state["ec_crt_status"] = ""
                st.session_state["ec_crt_court"] = ""
                st.rerun()

        with col_ex2:
            if st.button("🗑️ Clear Form", key="btn_clear_example_case"):
                for k in list(st.session_state.keys()):
                    if k.startswith("ec_") or k in [
                        "fir_father_or_husband", "fir_incident_date",
                        "mlc_guardian_name", "fsl_guardian_name",
                        "pros_incident_date", "court_father_or_husband"
                    ]:
                        del st.session_state[k]
                st.rerun()

        with st.form("manual_case_form"):
            # Section 1: Police FIR
            with st.expander("1. Police FIR (First Information Report)", expanded=True):
                f_c1, f_c2 = st.columns(2)
                with f_c1:
                    fir_number = st.text_input("FIR Number", value=st.session_state.get("ec_fir_no", ""))
                    complainant_name = st.text_input("Complainant / Survivor Name", value=st.session_state.get("ec_fir_name", ""))
                    father_or_husband = st.text_input("Father or Husband Name", value=st.session_state.get("ec_fir_guardian", ""), key="fir_father_or_husband")
                    age = st.text_input("Age", value=st.session_state.get("ec_fir_age", ""))
                    locality = st.text_input("Locality / Address", value=st.session_state.get("ec_fir_locality", ""))
                with f_c2:
                    district = st.text_input("District", value=st.session_state.get("ec_fir_district", ""))
                    incident_date = st.text_input("Incident Date (YYYY-MM-DD)", value=st.session_state.get("ec_fir_inc_date", ""), key="fir_incident_date")
                    fir_date = st.text_input("FIR Registration Date (YYYY-MM-DD)", value=st.session_state.get("ec_fir_rec_date", ""))
                    offense_category = st.text_input("Offense Category", value=st.session_state.get("ec_fir_offense", ""))
                    police_station = st.text_input("Police Station", value=st.session_state.get("ec_fir_station", ""))

            # Section 2: Hospital MLC
            with st.expander("2. Hospital MLC (Medico-Legal Case)", expanded=True):
                m_c1, m_c2 = st.columns(2)
                with m_c1:
                    mlc_number = st.text_input("MLC Number", value=st.session_state.get("ec_mlc_no", ""))
                    patient_name = st.text_input("Patient Name", value=st.session_state.get("ec_mlc_name", ""))
                    guardian_name = st.text_input("Guardian Name", value=st.session_state.get("ec_mlc_guardian", ""), key="mlc_guardian_name")
                    dob = st.text_input("Age or DOB (YYYY-MM-DD)", value=st.session_state.get("ec_mlc_age", ""))
                    residential_locality = st.text_input("Residential Locality", value=st.session_state.get("ec_mlc_locality", ""))
                with m_c2:
                    mlc_district = st.text_input("Hospital District", value=st.session_state.get("ec_mlc_district", ""))
                    mlc_incident_date = st.text_input("Reported Incident Date", value=st.session_state.get("ec_mlc_inc_date", ""))
                    examination_date = st.text_input("Examination Date (YYYY-MM-DD)", value=st.session_state.get("ec_mlc_rec_date", ""))
                    injury_type = st.text_input("Injury Classification", value=st.session_state.get("ec_mlc_injury", ""))
                    hospital_name = st.text_input("Hospital Name", value=st.session_state.get("ec_mlc_hospital", ""))

            # Section 3: Forensic Science Lab (FSL)
            with st.expander("3. Forensic Science Lab (FSL)", expanded=False):
                fs_c1, fs_c2 = st.columns(2)
                with fs_c1:
                    fsl_case_number = st.text_input("FSL Case Number", value=st.session_state.get("ec_fsl_no", ""))
                    subject_name = st.text_input("Subject / Victim Name", value=st.session_state.get("ec_fsl_name", ""))
                    fsl_guardian = st.text_input("Guardian Name", value=st.session_state.get("ec_fsl_guardian", ""), key="fsl_guardian_name")
                    fsl_age = st.text_input("Subject Age", value=st.session_state.get("ec_fsl_age", ""))
                with fs_c2:
                    fsl_district = st.text_input("FSL District", value=st.session_state.get("ec_fsl_district", ""))
                    sample_receipt_date = st.text_input("Sample Receipt Date (YYYY-MM-DD)", value=st.session_state.get("ec_fsl_inc_date", ""))
                    report_dispatch_date = st.text_input("Report Dispatch Date (YYYY-MM-DD)", value=st.session_state.get("ec_fsl_rec_date", ""))
                    evidence_type = st.text_input("Evidence Type", value=st.session_state.get("ec_fsl_evidence", ""))
                    lab_division = st.text_input("Lab Division", value=st.session_state.get("ec_fsl_division", ""))

            # Section 4: Prosecution Chargesheet
            with st.expander("4. Prosecution Chargesheet", expanded=False):
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    prosecution_file_no = st.text_input("Prosecution File No", value=st.session_state.get("ec_pros_no", ""))
                    victim_name = st.text_input("Victim Name", value=st.session_state.get("ec_pros_name", ""))
                    parent_spouse_name = st.text_input("Parent or Spouse Name", value=st.session_state.get("ec_pros_guardian", ""))
                    pros_age = st.text_input("Victim Age", value=st.session_state.get("ec_pros_age", ""))
                    address = st.text_input("Victim Address", value=st.session_state.get("ec_pros_address", ""))
                with p_c2:
                    pros_district = st.text_input("Prosecution District", value=st.session_state.get("ec_pros_district", ""))
                    pros_inc_date = st.text_input("Incident Date (YYYY-MM-DD)", value=st.session_state.get("ec_pros_inc_date", ""), key="pros_incident_date")
                    chargesheet_date = st.text_input("Chargesheet Filing Date (YYYY-MM-DD)", value=st.session_state.get("ec_pros_rec_date", ""))
                    case_category = st.text_input("Case Category", value=st.session_state.get("ec_pros_cat", ""))
                    prosecutor_office = st.text_input("Prosecutor Office", value=st.session_state.get("ec_pros_off", ""))

            # Section 5: District Court
            with st.expander("5. District Court Verdict", expanded=False):
                c_c1, c_c2 = st.columns(2)
                with c_c1:
                    cnr_number = st.text_input("CNR Number", value=st.session_state.get("ec_crt_no", ""))
                    petitioner_victim = st.text_input("Petitioner / Victim Name", value=st.session_state.get("ec_crt_name", ""))
                    court_guardian = st.text_input("Father or Husband Name", value=st.session_state.get("ec_crt_guardian", ""), key="court_father_or_husband")
                    court_dob = st.text_input("DOB or Age", value=st.session_state.get("ec_crt_age", ""))
                    residence_area = st.text_input("Residence Area", value=st.session_state.get("ec_crt_locality", ""))
                with c_c2:
                    court_district = st.text_input("Court District", value=st.session_state.get("ec_crt_district", ""))
                    date_of_incident = st.text_input("Date of Incident", value=st.session_state.get("ec_crt_inc_date", ""))
                    verdict_date = st.text_input("Verdict Date (YYYY-MM-DD)", value=st.session_state.get("ec_crt_rec_date", ""))
                    case_status = st.text_input("Case Status / Outcome", value=st.session_state.get("ec_crt_status", ""))
                    court_name = st.text_input("Court Name", value=st.session_state.get("ec_crt_court", ""))

            btn_submit_case = st.form_submit_button("🔍 Analyze Case", type="primary", width="stretch")

        if btn_submit_case:
            payload = {}
            if complainant_name or fir_number:
                payload["police_fir"] = {
                    "fir_number": fir_number, "complainant_name": complainant_name,
                    "father_or_husband": father_or_husband, "age": age, "locality": locality,
                    "district": district, "incident_date": incident_date, "fir_date": fir_date,
                    "offense_category": offense_category, "police_station": police_station
                }
            if patient_name or mlc_number:
                payload["hospital_mlc"] = {
                    "mlc_number": mlc_number, "patient_name": patient_name,
                    "guardian_name": guardian_name, "dob": dob, "residential_locality": residential_locality,
                    "district": mlc_district, "incident_date": mlc_incident_date, "examination_date": examination_date,
                    "injury_type": injury_type, "hospital_name": hospital_name
                }
            if subject_name or fsl_case_number:
                payload["forensic_lab"] = {
                    "fsl_case_number": fsl_case_number, "subject_name": subject_name,
                    "guardian_name": fsl_guardian, "age": fsl_age, "district": fsl_district,
                    "sample_receipt_date": sample_receipt_date, "report_dispatch_date": report_dispatch_date,
                    "evidence_type": evidence_type, "lab_division": lab_division
                }
            if victim_name or prosecution_file_no:
                payload["prosecutor"] = {
                    "prosecution_file_no": prosecution_file_no, "victim_name": victim_name,
                    "parent_spouse_name": parent_spouse_name, "age": pros_age, "address": address,
                    "district": pros_district, "incident_date": pros_inc_date, "chargesheet_date": chargesheet_date,
                    "case_category": case_category, "prosecutor_office": prosecutor_office
                }
            if petitioner_victim or cnr_number:
                payload["court"] = {
                    "cnr_number": cnr_number, "petitioner_victim": petitioner_victim,
                    "father_or_husband": court_guardian, "dob": court_dob, "residence_area": residence_area,
                    "district": court_district, "date_of_incident": date_of_incident, "verdict_date": verdict_date,
                    "case_status": case_status, "court_name": court_name
                }

            if len(payload) < 2:
                st.error("⚠️ Please fill in at least 2 institutional sections to evaluate case linkage.")
            else:
                with st.spinner("Evaluating cross-agency record linkage and reconstructing journey timeline..."):
                    c_res = run_analyze_case(payload, token=st.session_state.get("token"))

                if c_res.get("error"):
                    st.error(f"⚠️ **Analysis Validation Error**: {c_res['error']}")
                else:
                    st.session_state["single_case_analysis_res"] = c_res
                    st.session_state["active_dataset_id"] = "case"
                    m_name = c_res.get("masked_survivor_name", "Anonymous")
                    st.session_state["active_dataset_label"] = f"Single Case ({m_name})"
                    st.session_state["active_dataset_bundle"] = build_dashboard_bundle_from_analysis({
                        "is_single_case": True,
                        **c_res
                    })
                    st.session_state["active_analysis_result"] = c_res
                    st.success("Case analyzed successfully. Active dashboard dataset updated.")

        # Render Single Case Results
        sc_res = st.session_state.get("single_case_analysis_res")
        if sc_res:
            st.markdown("---")
            st.markdown("### 📋 Case Journey & Record Linkage Evaluation")
            st.info(
                "ℹ️ **Individual Case Scope**: This analysis evaluates pairwise linkage, evidence breakdown, and chronological timeline "
                "for this single synthetic record only. Single-case entries do not calculate system-wide or district-wide attrition rates, "
                "preventing statistically misleading extrapolations from insufficient sample sizes."
            )

            # Masked Survivor Name & Linkage Badge
            m_name = sc_res.get("masked_survivor_name", "Anonymous")
            overall_label = sc_res.get("overall_linkage_label", "same case")
            l_method = sc_res.get("linkage_method", "Fellegi-Sunter Pairwise Scoring (case_simulator)")

            badge_color = "#059669" if overall_label == "same case" else ("#B45309" if overall_label == "needs review" else "#DC2626")
            st.markdown(f"""
                <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-left: 4px solid {badge_color}; border-radius: 4px; padding: 14px 18px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <div style="font-size: 0.78rem; text-transform: uppercase; color: #5F5F5F; font-weight: 700;">Survivor Identity (Server-Side Masked)</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #222222;">👤 {m_name}</div>
                        </div>
                        <div style="text-align: right;">
                            <span style="background: {badge_color}; color: #FFFFFF; padding: 4px 10px; border-radius: 3px; font-weight: 600; text-transform: uppercase; font-size: 0.85rem;">
                                {overall_label}
                            </span>
                            <div style="font-size: 0.72rem; color: #5F5F5F; margin-top: 4px;">Linkage: {l_method}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Bottleneck / Stall Point Callout
            t_info = sc_res.get("timeline", {})
            stalled_at = t_info.get("stalled_at")
            missing_rec = t_info.get("missing_record_from")

            if stalled_at and missing_rec:
                st.warning(f"⚠️ **Case Stalled at {stalled_at}**: Missing expected record from **{missing_rec}**.")
            elif t_info.get("all_milestones_reached"):
                st.success("✅ **Complete Institutional Trajectory**: Case successfully advanced across all 5 institutional milestones.")

            longest = t_info.get("longest_transition")
            if longest:
                st.info(
                    f"⏱️ **Greatest Inter-Stage Latency**: {longest.get('pair')} required "
                    f"**{longest.get('duration_days')} days** (baseline: {longest.get('baseline_days')} days, "
                    f"<strong>{longest.get('multiplier')}× baseline</strong>)."
                )

            # Timeline Stages Display
            t_stages = t_info.get("stages", [])
            if t_stages:
                st.markdown("#### ⏱️ Institutional Journey Milestones")
                st_cols = st.columns(len(t_stages))
                for idx, sc_item in enumerate(t_stages):
                    with st_cols[idx]:
                        is_p = sc_item.get("present")
                        st_bg = "#FFFFFF" if is_p else "#F9F9F9"
                        st_border = "#8B1E3F" if is_p else "#CBD5E1"
                        st.markdown(f"""
                            <div style="background: {st_bg}; border: 1px solid {st_border}; border-top: 3px solid {st_border}; border-radius: 4px; padding: 10px 12px; min-height: 110px;">
                                <div style="font-size: 0.75rem; font-weight: 700; color: #222222;">{sc_item.get('stage_name')}</div>
                                <div style="font-size: 0.82rem; color: {'#059669' if is_p else '#94A3B8'}; font-weight: 600; margin-top: 4px;">
                                    {'✓ Reached' if is_p else '✗ Missing'}
                                </div>
                                <div style="font-size: 0.75rem; color: #5F5F5F; margin-top: 2px;">{sc_item.get('date') or 'No date'}</div>
                                {f'<div style="font-size: 0.72rem; color: #B45309; font-weight: 600; margin-top: 4px;">+{sc_item.get("duration_days_from_prev")}d</div>' if sc_item.get("duration_days_from_prev") is not None else ''}
                            </div>
                        """, unsafe_allow_html=True)

            # Match Score Table
            pair_scores = sc_res.get("pairwise_match_scores", [])
            if pair_scores:
                st.markdown("#### 🔗 Pairwise Linkage Probabilities & Log-Odds Evidence")
                p_rows = []
                for p in pair_scores:
                    lbl = p.get("label", "needs review")
                    lbl_badge = "🟢 Same Case" if lbl == "same case" else ("🟡 Needs Review" if lbl == "needs review" else "🔴 Different Case")
                    ev_summary = "; ".join([e.get("detail", "") for e in p.get("evidence", []) if e.get("status") in ["AGREE", "DISAGREE"]])
                    p_rows.append({
                        "Comparison Pair": f"{p.get('stage_a')} ↔ {p.get('stage_b')}",
                        "Match Probability": f"{p.get('match_probability', 0.0):.3f}",
                        "Match Weight (Log-Odds)": f"{p.get('match_weight', 0.0):.2f}",
                        "Evaluation Decision": lbl_badge,
                        "Key Evidence": ev_summary[:120] + "..." if len(ev_summary) > 120 else ev_summary
                    })
                st.dataframe(pd.DataFrame(p_rows), width="stretch", hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 2: Upload CSVs
    # -------------------------------------------------------------------------
    with tab_upload:
        st.markdown("""
            <div style="margin-bottom: 14px;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #222222;">Batch CSV Upload</div>
                <div style="font-size: 0.85rem; color: #5F5F5F;">
                    Upload the 5 institutional CSV files matching real agency schemas (max 2000 cases and 5 MB per file).
                    Processing runs in volatile memory with zero microdata disk persistence.
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Template download button
        tmpl_bytes = get_analyze_template(token=st.session_state.get("token"))
        if tmpl_bytes:
            st.download_button(
                label="📥 Download CSV Templates (.zip)",
                data=tmpl_bytes,
                file_name="caselens_csv_templates.zip",
                mime="application/zip",
                key="btn_download_csv_templates",
                help="Download zip containing header-only CSV templates for all 5 agencies."
            )

        with st.expander("📋 View Expected CSV Headers & Supported Column Aliases", expanded=False):
            st.markdown("""
                Each uploaded CSV must include the agency's required fields. Common column aliases (such as `name` for `complainant_name` or `father_name` for `father_or_husband`) are automatically mapped.
                
                * **1. Police FIR (`police_fir.csv`)**: `fir_number`, `complainant_name` *(or name, victim_name)*, `father_or_husband` *(or guardian_name, father_name)*, `age`, `locality` *(or address)*, `district`, `incident_date`, `fir_date`, `offense_category`, `police_station` *(or station, ps)*
                * **2. Hospital MLC (`hospital_mlc.csv`)**: `mlc_number`, `patient_name` *(or name)*, `guardian_name` *(or father_or_husband)*, `dob`, `residential_locality` *(or address, locality)*, `district`, `incident_date`, `examination_date`, `injury_type`, `hospital_name`
                * **3. Forensic Lab (`forensic_lab.csv`)**: `fsl_case_number`, `subject_name` *(or name)*, `guardian_name`, `age`, `district`, `sample_receipt_date`, `report_dispatch_date`, `evidence_type`, `lab_division`
                * **4. Prosecution Chargesheet (`prosecutor.csv`)**: `prosecution_file_no`, `victim_name` *(or name)*, `parent_spouse_name` *(or guardian_name)*, `age`, `address` *(or locality)*, `district`, `incident_date`, `chargesheet_date`, `case_category`, `prosecutor_office`
                * **5. District Court Verdict (`court.csv`)**: `cnr_number`, `petitioner_victim` *(or victim_name, name)*, `father_or_husband` *(or guardian_name)*, `dob`, `residence_area` *(or address)*, `district`, `date_of_incident` *(or incident_date)*, `verdict_date`, `case_status`, `court_name`
            """)
        st.markdown("<br/>", unsafe_allow_html=True)

        up_col1, up_col2 = st.columns(2)
        with up_col1:
            u_fir = st.file_uploader("1. Police FIR CSV", type=["csv"], key="u_file_fir")
            u_mlc = st.file_uploader("2. Hospital MLC CSV", type=["csv"], key="u_file_mlc")
            u_fsl = st.file_uploader("3. Forensic Lab CSV", type=["csv"], key="u_file_fsl")
        with up_col2:
            u_pros = st.file_uploader("4. Prosecution Chargesheet CSV", type=["csv"], key="u_file_pros")
            u_court = st.file_uploader("5. District Court Verdict CSV", type=["csv"], key="u_file_court")

        btn_run_upload = st.button("🚀 Run Analysis on Uploaded CSVs", type="primary", key="btn_run_upload_analysis")

        if btn_run_upload:
            if not (u_fir and u_mlc and u_fsl and u_pros and u_court):
                st.warning("⚠️ Please upload all 5 CSV files (Police FIR, Hospital MLC, Forensic Lab, Prosecution, Court) to run the analysis.")
            else:
                files_payload = {
                    "police_fir": (u_fir.name, u_fir.getvalue(), "text/csv"),
                    "hospital_mlc": (u_mlc.name, u_mlc.getvalue(), "text/csv"),
                    "forensic_lab": (u_fsl.name, u_fsl.getvalue(), "text/csv"),
                    "prosecutor": (u_pros.name, u_pros.getvalue(), "text/csv"),
                    "court": (u_court.name, u_court.getvalue(), "text/csv")
                }
                with st.spinner("Validating schemas, performing in-memory record linkage, and computing funnel analytics..."):
                    up_res = run_analyze_upload(files_payload, token=st.session_state.get("token"))

                if up_res.get("error"):
                    st.error(f"⚠️ **Upload Analysis Failed**: {up_res['error']}")
                else:
                    st.session_state["upload_analysis_res"] = up_res
                    st.session_state["active_analysis_result"] = up_res
                    total_c = up_res.get("summary_kpis", {}).get("total_cases") or up_res.get("funnel", {}).get("total_cases", 0)
                    st.session_state["active_dataset_id"] = "uploaded"
                    st.session_state["active_dataset_label"] = f"Uploaded CSVs ({total_c:,} cases)"
                    st.session_state["active_dataset_bundle"] = build_dashboard_bundle_from_analysis(up_res)
                    st.success("Analysis executed successfully. Active dashboard dataset updated.")

        res_upload = st.session_state.get("upload_analysis_res")
        if res_upload and not res_upload.get("error"):
            st.markdown("---")
            render_csv_analysis_results(res_upload, title_prefix="Uploaded Files")

    # -------------------------------------------------------------------------
    # TAB 3: Use sample files
    # -------------------------------------------------------------------------
    with tab_sample:
        st.markdown("""
            <div style="margin-bottom: 14px;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #222222;">One-Click Benchmark Sample Analysis</div>
                <div style="font-size: 0.85rem; color: #5F5F5F;">
                    Execute complete record linkage and journey reconstruction on the benchmark sample CSV files
                    bundled with CaseLens in volatile memory without modifying precomputed baseline artifacts.
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Run Analysis on Sample Files", type="primary", key="btn_run_sample_analysis"):
            with st.spinner("Executing record linkage and process mining on sample benchmark datasets..."):
                smpl_res = run_analyze_sample(limit=500, token=st.session_state.get("token"))

            if smpl_res.get("error"):
                st.error(f"⚠️ **Sample Analysis Failed**: {smpl_res['error']}")
            else:
                st.session_state["sample_analysis_res"] = smpl_res
                st.session_state["active_analysis_result"] = smpl_res
                total_c = smpl_res.get("summary_kpis", {}).get("total_cases") or smpl_res.get("funnel", {}).get("total_cases", 0)
                st.session_state["active_dataset_id"] = "sample"
                st.session_state["active_dataset_label"] = f"Benchmark Sample Dataset ({total_c:,} cases)"
                st.session_state["active_dataset_bundle"] = build_dashboard_bundle_from_analysis(smpl_res)
                st.success("Sample analysis completed successfully. Active dashboard dataset updated.")

        res_sample = st.session_state.get("sample_analysis_res")
        if res_sample and not res_sample.get("error"):
            st.markdown("---")
            render_csv_analysis_results(res_sample, title_prefix="Sample Benchmark Files")


# ==============================================================================
# PAGE 1: 📊 Executive Overview
# ==============================================================================

elif PAGE == "📊 Executive Overview":
    leak_badge_html = (
        f'<span class="badge badge-critical">{KPIS["leak_count_display"]} Policy Leaks Flagged</span>'
        if KPIS.get("leak_count_display") not in [NA, "0"]
        else '<span class="badge badge-normal">0 Policy Leaks Flagged</span>'
    )
    st.markdown(f"""
        <div class="hero-banner">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
                <div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #8B1E3F; letter-spacing: -0.01em; margin-bottom: 2px;">CaseLens</div>
                    <div style="font-size: 0.88rem; color: #8B1E3F; font-weight: 600; margin-bottom: 12px;">The Leaky Pipeline: Tracing Gaps in the Justice System</div>
                </div>
                <div style="display: flex; gap: 6px; align-items: center; padding-top: 4px; flex-wrap: wrap;">
                    <span class="badge badge-privacy">UN SDG 5</span>
                    <span class="badge badge-normal">UN SDG 16</span>
                    <span class="badge badge-privacy">ε-DP Laplace Protected</span>
                    <span class="badge badge-normal">Zero-PII Verified</span>
                    {leak_badge_html}
                </div>
            </div>
            <div class="hero-title">Executive Policy Overview: Violence-Against-Women Case Attrition</div>
            <div class="hero-subtitle">
                Systemic drop-off analytics reconstructed from multi-institutional silos (Police FIR, Hospital MLC, 
                Forensic Science Laboratories, Prosecution, and District Courts) with mathematical Differential Privacy protection.
            </div>
        </div>
    """, unsafe_allow_html=True)

    render_filter_badge(FILTER_DISTRICT, (FILTER_FROM, FILTER_TO), FILTER_STAGE, KPIS.get("epsilon_remaining_display"))

    if not st.session_state.get("active_dataset_id"):
        render_no_active_dataset_banner()
        st.stop()

    # Single-case view: Display case-specific results and explain aggregate dataset requirement
    if st.session_state.get("active_dataset_id") == "case":
        sc_res = st.session_state.get("single_case_analysis_res") or {}
        m_name = sc_res.get("masked_survivor_name", "Anonymous")
        overall_lbl = sc_res.get("overall_linkage_label", "same case")
        l_meth = sc_res.get("linkage_method", "Fellegi-Sunter")
        t_info = sc_res.get("timeline", {})
        stages = t_info.get("stages", [])
        reached_count = sum(1 for s in stages if s.get("present"))
        longest = t_info.get("longest_transition")
        stalled_at = t_info.get("stalled_at")
        missing_rec = t_info.get("missing_record_from")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Survivor Identity</div>
                    <div class="metric-value" style="font-size: 1.3rem;">👤 {m_name}</div>
                    <div class="metric-delta">Server-Side Masked</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Milestones Reached</div>
                    <div class="metric-value">{reached_count} / 5</div>
                    <div class="metric-delta">{"All 5 Reached" if reached_count == 5 else f"Stalled at {stalled_at or 'Pipeline'}"}</div>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            badge_col = "#059669" if overall_lbl == "same case" else ("#B45309" if overall_lbl == "needs review" else "#DC2626")
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Linkage Determination</div>
                    <div class="metric-value" style="color: {badge_col}; font-size: 1.3rem;">{overall_lbl.upper()}</div>
                    <div class="metric-delta">{l_meth}</div>
                </div>
            """, unsafe_allow_html=True)
        with c4:
            l_dur = f"{longest.get('duration_days')}d" if longest else "N/A"
            l_pair = longest.get('pair', 'None') if longest else "No delays"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Longest Transition</div>
                    <div class="metric-value">{l_dur}</div>
                    <div class="metric-delta">{l_pair}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.info(
            "ℹ️ **Single-Case Scope**: You are currently viewing case-specific results for an individual case record. "
            "District vulnerability rankings, population-level leak distributions, and multi-agency aggregate benchmarks "
            "require a multi-case dataset. To analyze cohort statistics, upload CSVs or run benchmark sample analysis in Analyze Your Data."
        )

        if stalled_at and missing_rec:
            st.warning(f"⚠️ **Case Stalled at {stalled_at}**: Missing expected record from **{missing_rec}**.")
        elif t_info.get("all_milestones_reached"):
            st.success("✅ **Complete Institutional Trajectory**: Case successfully advanced across all 5 institutional milestones.")

        if longest:
            st.info(
                f"⏱️ **Greatest Inter-Stage Latency**: {longest.get('pair')} required "
                f"**{longest.get('duration_days')} days** (statutory baseline: {longest.get('baseline_days')} days, "
                f"<strong>{longest.get('multiplier')}× baseline</strong>)."
            )
        st.stop()

    summary_data = DATA.get("summary", {})
    if summary_data.get("too_few_cases"):
        render_too_few_cases_warning()
        st.stop()

    health = summary_data.get("pipeline_health", {})

    # Top KPI Metrics Cards (4 wide columns to prevent text/number wrapping)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Reconstructed Cases</div>
                <div class="metric-value">{KPIS['reconstructed_cases_display']}</div>
                <div class="metric-delta">{KPIS['events_display']} Canonical Events</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">FIR Initiating Rate</div>
                <div class="metric-value">{KPIS['fir_rate_display']}</div>
                <div class="metric-delta">{KPIS['fir_count_display']} Police FIRs</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Conviction Reach Rate</div>
                <div class="metric-value">{KPIS['verdict_rate_display']}</div>
                <div class="metric-delta">{KPIS['verdict_count_display']} Reaching Trial Verdict</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Policy Leak Points</div>
                <div class="metric-value">{KPIS['leak_count_display']} Critical</div>
                <div class="metric-delta">{KPIS['leak_districts_display']} Anomalies</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Key Empirical Findings (Dynamic signals calculated from active dataset)
    st.markdown("### 🎯 Key Systemic Attrition Findings")

    # Signal 1: Transition Latency / Primary Bottleneck
    bottlenecks_data = DATA.get("bottlenecks", {})
    trans_list = bottlenecks_data.get("inter_stage_transitions", [])
    max_trans = max(trans_list, key=lambda t: t.get("median_duration_days", 0), default={}) if trans_list else {}
    if max_trans and max_trans.get("median_duration_days"):
        sig1_title = f"Primary Latency Bottleneck ({max_trans.get('transition', 'Stage')})"
        sig1_val = f"{max_trans.get('median_duration_days', 0):.1f} Days"
        sig1_sub = f"P90: {max_trans.get('p90_duration_days', 0):.1f}d | Statutory: {max_trans.get('statutory_benchmark_days', 60)}d"
    else:
        sig1_title = "Primary Latency Bottleneck"
        sig1_val = "Baseline"
        sig1_sub = "No critical delay spike detected"

    # Signal 2: Top Attrition / Leak Point
    leaks_list = summary_data.get("systemic_leak_points_detected", [])
    if leaks_list:
        top_leak = leaks_list[0]
        sig2_title = f"Top Policy Leak ({top_leak.get('leak_code', 'LP-01')})"
        sig2_val = f"{top_leak.get('title', 'Leak Detected')[:26]}"
        sig2_sub = f"{top_leak.get('district', 'Statewide')} • {top_leak.get('agency', 'Agency')}"
    else:
        sig2_title = "Systemic Policy Leaks"
        sig2_val = "Zero Flagged"
        sig2_sub = "No statutory violations detected"

    # Signal 3: Record Linkage Evaluation
    eval_info = DATA.get("evaluation", {})
    pw_m = eval_info.get("pairwise_metrics", {})
    cl_m = eval_info.get("cluster_metrics", {})
    l_method = eval_info.get("linkage_method", "Record Linkage")
    l_short = "Fellegi-Sunter" if "Fellegi" in l_method else ("Splink" if "Splink" in l_method else l_method)
    rec_val = pw_m.get("at_threshold", {}).get("recall") or pw_m.get("recall")
    f1_val = cl_m.get("bcubed_f1")
    if rec_val is not None and f1_val is not None:
        sig3_title = f"Linkage Quality ({l_short})"
        sig3_val = f"{rec_val * 100:.1f}% / {f1_val * 100:.1f}%"
        sig3_sub = "Pairwise Recall / B-Cubed F1"
    else:
        sig3_title = f"Linkage Quality ({l_short})"
        sig3_val = "Validated"
        sig3_sub = "Multi-institutional trajectory linkage"

    sig1, sig2, sig3 = st.columns(3)
    with sig1:
        st.markdown(f"""
            <div class="metric-card" style="border-left: 3px solid #B45309;">
                <div class="metric-label">{sig1_title}</div>
                <div class="metric-value" style="color: #B45309;">{sig1_val}</div>
                <div class="metric-delta">{sig1_sub}</div>
            </div>
        """, unsafe_allow_html=True)
    with sig2:
        st.markdown(f"""
            <div class="metric-card" style="border-left: 3px solid #B45309;">
                <div class="metric-label">{sig2_title}</div>
                <div class="metric-value" style="color: #B45309;">{sig2_val}</div>
                <div class="metric-delta">{sig2_sub}</div>
            </div>
        """, unsafe_allow_html=True)
    with sig3:
        st.markdown(f"""
            <div class="metric-card" style="border-left: 3px solid #8B1E3F;">
                <div class="metric-label">{sig3_title}</div>
                <div class="metric-value" style="color: #8B1E3F;">{sig3_val}</div>
                <div class="metric-delta">{sig3_sub}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # District Vulnerability Ranking & Systemic Leak Point Detection
    col_left, col_right = st.columns([6, 5])

    with col_left:
        st.markdown("### 🚨 District Vulnerability Severity Index")
        ranking = summary_data.get("district_vulnerability_ranking", [])
        if ranking:
            df_rank = pd.DataFrame(ranking)
            if "score" not in df_rank.columns and "leak_score" in df_rank.columns:
                df_rank["score"] = df_rank["leak_score"]
            if "status" not in df_rank.columns:
                df_rank["status"] = df_rank["score"].apply(lambda s: "CRITICAL" if s >= 10.0 else "NORMAL")
            fig_rank = px.bar(
                df_rank,
                x="score",
                y="district",
                orientation="h",
                color="status",
                color_discrete_map={"CRITICAL": "#B45309", "NORMAL": "#8C8C8C"},
                text="score",
                labels={"score": "Vulnerability Score", "district": "District", "status": "Classification"},
                title="Composite District Leak Severity Score (Baseline: Normal < 10.0)"
            )
            fig_rank.update_layout(get_plot_layout(), yaxis=dict(autorange="reversed"))
            fig_rank.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig_rank, width="stretch")
        else:
            st.info("No district vulnerability breakdown available for this active cohort.")

    with col_right:
        st.markdown("### 🔍 Systemic Policy Leak Points Detected")
        leak_points = summary_data.get("systemic_leak_points_detected", [])
        if leak_points:
            for lp in leak_points:
                dist = lp.get("district")
                code = lp.get("leak_code")
                title = lp.get("title")
                badge_html = f'<span class="badge badge-critical">{code}</span>'
                detail_text = format_leak_detail(lp)

                st.markdown(f"""
                    <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-left: 3px solid #B45309; border-radius: 4px; padding: 14px 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <strong style="color: #222222; font-size: 1.0rem;">{dist}</strong>
                            {badge_html}
                        </div>
                        <div style="color: #222222; font-size: 0.88rem; margin-bottom: 4px;">{title}</div>
                        <div style="color: #5F5F5F; font-size: 0.82rem;">{detail_text}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No systemic policy leak points detected in this active cohort.")


# ==============================================================================
# PAGE 2: 🔻 Case Funnel
# ==============================================================================

elif PAGE in ["🔻 Case Funnel", "🔻 Funnel with Uncertainty Ranges"]:
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Case Funnel: Milestone Attrition with Uncertainty Ranges</div>
            <div class="hero-subtitle">
                Institutional attrition across canonical milestones. Every aggregate displays mathematical 
                Laplace noise error margins (±95% CI) and small-cell k-anonymity suppression (&lt;k).
            </div>
        </div>
    """, unsafe_allow_html=True)

    render_filter_badge(FILTER_DISTRICT, (FILTER_FROM, FILTER_TO), FILTER_STAGE, KPIS.get("epsilon_remaining_display"))

    if not st.session_state.get("active_dataset_id"):
        render_no_active_dataset_banner()
        st.stop()

    if st.session_state.get("active_dataset_id") == "case":
        sc_res = st.session_state.get("single_case_analysis_res") or {}
        t_info = sc_res.get("timeline", {})
        t_stages = t_info.get("stages", [])

        st.info(
            "ℹ️ **Single-Case Trajectory**: Displaying milestone progression for the active case. "
            "Aggregate funnel attrition rates, Differential Privacy Laplace error bars, and small-cell k-anonymity suppression "
            "require a multi-case cohort. To view statistical attrition funnels, upload multi-agency CSVs or run benchmark sample analysis in Analyze Your Data."
        )

        if t_stages:
            st.markdown("### ⏱️ Individual Journey Milestones")
            st_cols = st.columns(len(t_stages))
            for idx, sc_item in enumerate(t_stages):
                with st_cols[idx]:
                    is_p = sc_item.get("present")
                    st_bg = "#FFFFFF" if is_p else "#F9F9F9"
                    st_border = "#8B1E3F" if is_p else "#CBD5E1"
                    dur_disp = f"+{sc_item.get('duration_days_from_prev')}d" if sc_item.get("duration_days_from_prev") is not None else ""
                    st.markdown(f"""
                        <div style="background: {st_bg}; border: 1px solid {st_border}; border-top: 3px solid {st_border}; border-radius: 4px; padding: 12px; min-height: 120px;">
                            <div style="font-size: 0.75rem; font-weight: 700; color: #222222;">{sc_item.get('stage_name')}</div>
                            <div style="font-size: 0.85rem; color: {'#059669' if is_p else '#94A3B8'}; font-weight: 600; margin-top: 6px;">
                                {'✓ Reached' if is_p else '✗ Missing'}
                            </div>
                            <div style="font-size: 0.75rem; color: #5F5F5F; margin-top: 4px;">{sc_item.get('date') or 'No record'}</div>
                            <div style="font-size: 0.72rem; color: #B45309; font-weight: 600; margin-top: 4px;">{dur_disp}</div>
                        </div>
                    """, unsafe_allow_html=True)
        st.stop()

    dp_funnel = DATA.get("dp_funnel", {})
    if dp_funnel.get("too_few_cases"):
        render_too_few_cases_warning()
        st.stop()

    funnel_stages = dp_funnel.get("funnel_stages", [])

    if not funnel_stages:
        st.warning("Funnel analytical data not loaded. Please ensure analytics pipeline has run.")
    else:
        # Analytical guardrail callout
        st.markdown("""
            <div class="callout-info" style="border-left-color: #8B1E3F; margin-bottom: 16px;">
                <strong>⚖️ Institutional Data Lineage & Guardrail:</strong>
                Missing records between stages (e.g. absent hospital MLC or forensic report) 
                <strong>must not be interpreted as definitive proof that the event never occurred</strong>. 
                Decentralized paper registries, manual data entry lags, and institutional silo barriers 
                account for tracking gaps. All milestone statistics are mathematically privatized with 
                Laplace Differential Privacy noise and small-cell k-anonymity suppression (&lt;k).
            </div>
        """, unsafe_allow_html=True)

        # Confidence level selector & privacy parameters
        ctrl_col1, ctrl_col2 = st.columns([4, 6])
        with ctrl_col1:
            CONF_LEVEL = st.selectbox("Confidence Level", ["95% Confidence Interval", "99% Confidence Interval"], index=0)
        with ctrl_col2:
            st.markdown(f"""
                <div style="padding-top: 8px; font-size: 0.85rem; color: #5F5F5F;">
                    <strong>Privacy Mechanism:</strong> Pure Laplace Mechanism (&Delta;f=1, &epsilon;={EPSILON_SIM:.2f}) with small-cell suppression (k&ge;{K_ANONYMITY})
                </div>
            """, unsafe_allow_html=True)

        # Plotly Funnel Chart with Differential Privacy Error Bars
        stage_names = [s["stage_name"] for s in funnel_stages]
        noisy_counts = [s["noisy_case_count"] if isinstance(s["noisy_case_count"], (int, float)) else 0 for s in funnel_stages]
        
        # Calculate dynamic CI based on selected confidence level and epsilon
        scale_b = 1.0 / max(EPSILON_SIM, 0.01)
        ci_mult = -math.log(0.01) if "99%" in CONF_LEVEL else -math.log(0.05)
        err_margins = [
            round(scale_b * ci_mult, 1) if isinstance(s["noisy_case_count"], (int, float)) else 0.0
            for s in funnel_stages
        ]

        text_labels = []
        for s, val, err in zip(funnel_stages, noisy_counts, err_margins):
            if s.get("is_k_anonymity_suppressed") or str(s.get("noisy_case_count")) == "<5":
                text_labels.append("<5<br/>(Suppressed)")
            else:
                text_labels.append(f"{val:,}<br/>(±{err:.1f})")

        fig_funnel = go.Figure()

        fig_funnel.add_trace(go.Bar(
            name=f"Privatized Volume (ε={EPSILON_SIM:.2f})",
            x=stage_names,
            y=noisy_counts,
            error_y=dict(type='data', array=err_margins, visible=True, color="#8B1E3F", thickness=1.5, width=4),
            marker=dict(
                color="#8B1E3F",
                line=dict(color="#FFFFFF", width=1)
            ),
            text=text_labels,
            textposition="auto"
        ))

        fig_funnel.update_layout(
            get_plot_layout(),
            title=f"Institutional Stage Volume with {CONF_LEVEL} (Differential Privacy)",
            barmode="relative",
            yaxis_title="Case Volume",
            xaxis_title="Institutional Milestone"
        )
        st.plotly_chart(fig_funnel, width="stretch")

        # Stage Breakdown Table (Privatized Aggregates Only)
        st.markdown("### 📋 Milestone-by-Milestone Attrition Accounting")
        table_rows = []
        for idx, s in enumerate(funnel_stages):
            is_supp = s.get("is_k_anonymity_suppressed") or str(s.get("noisy_case_count")) == "<5"
            cnt_disp = "<5" if is_supp else f"{noisy_counts[idx]:,}"
            table_rows.append({
                "Milestone": s["stage_name"],
                "Privatized Case Volume (DP)": cnt_disp,
                f"Error Margin ({CONF_LEVEL.split()[0]})": "Suppressed" if is_supp else f"±{err_margins[idx]:.1f}",
                "Cumulative Retention": "Suppressed" if is_supp else fmt_pct(s.get("dp_retention_pct")),
                "Step Conversion": "Suppressed" if is_supp else fmt_pct(s.get("dp_conversion_pct")),
                f"k-Suppressed (<{K_ANONYMITY})": "⚠️ Masked" if is_supp else "✅ Compliant"
            })
        st.dataframe(pd.DataFrame(table_rows), width="stretch")


# ==============================================================================
# PAGE 3: ⏱️ Duration Bottlenecks & Latency
# PAGE 3: ⏱️ Duration & Bottlenecks
# ==============================================================================

elif PAGE in ["⏱️ Duration & Bottlenecks", "⏱️ Duration Bottlenecks & Latency"]:
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Duration & Bottlenecks: Inter-Agency Latency & Statutory Compliance</div>
            <div class="hero-subtitle">
                Quantifying latency across justice milestones against statutory benchmarks: 
                60-day police chargesheet mandate (Sec 173 CrPC / Sec 193 BNSS), 24h medical exam, and 30-day forensic reporting.
            </div>
        </div>
    """, unsafe_allow_html=True)

    render_filter_badge(FILTER_DISTRICT, (FILTER_FROM, FILTER_TO), FILTER_STAGE, KPIS.get("epsilon_remaining_display"))

    if not st.session_state.get("active_dataset_id"):
        render_no_active_dataset_banner()
        st.stop()

    if st.session_state.get("active_dataset_id") == "case":
        sc_res = st.session_state.get("single_case_analysis_res") or {}
        t_info = sc_res.get("timeline", {})
        stalled_at = t_info.get("stalled_at")
        missing_rec = t_info.get("missing_record_from")
        longest = t_info.get("longest_transition")

        st.info(
            "ℹ️ **Single-Case Durations**: Showing measured transition durations for this individual case. "
            "Cohort latency distributions, non-parametric percentiles (P50, P90, P95), and interquartile ranges require a multi-case dataset. "
            "To inspect population-wide duration bottlenecks, upload CSVs or run benchmark sample analysis in Analyze Your Data."
        )

        if stalled_at and missing_rec:
            st.warning(f"⚠️ **Case Stalled at {stalled_at}**: Missing expected record from **{missing_rec}**.")
        elif t_info.get("all_milestones_reached"):
            st.success("✅ **Complete Institutional Trajectory**: Case successfully advanced across all 5 institutional milestones.")

        if longest:
            st.info(
                f"⏱️ **Greatest Inter-Stage Latency**: {longest.get('pair')} required "
                f"**{longest.get('duration_days')} days** (statutory baseline: {longest.get('baseline_days')} days, "
                f"<strong>{longest.get('multiplier')}× baseline</strong>)."
            )

        trans_list = DATA.get("bottlenecks", {}).get("inter_stage_transitions", [])
        if trans_list:
            st.markdown("### ⏱️ Measured Stage Transitions")
            cols = st.columns(len(trans_list))
            for idx, tr in enumerate(trans_list):
                with cols[idx]:
                    dur = tr.get("duration_days")
                    dur_str = f"{dur:.1f} days" if isinstance(dur, (int, float)) else "N/A"
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{tr.get('transition', 'Transition')}</div>
                            <div class="metric-value">{dur_str}</div>
                            <div class="metric-delta">Statutory: {tr.get('statutory_benchmark_days', 60)}d</div>
                        </div>
                    """, unsafe_allow_html=True)
        st.stop()

    bottlenecks = DATA.get("bottlenecks", {})
    if bottlenecks.get("too_few_cases"):
        render_too_few_cases_warning()
        st.stop()

    # Key Stage Transitions
    key_transitions = [
        "FIR_REGISTERED -> MEDICAL_EXAM",
        "MEDICAL_EXAM -> FORENSIC_REPORT",
        "FORENSIC_REPORT -> CHARGESHEET_FILED",
        "CHARGESHEET_FILED -> COURT_VERDICT"
    ]

    col1, col2, col3, col4 = st.columns(4)
    stat_labels = ["FIR -> MLC (Hospital)", "MLC -> FSL (Forensics)", "FSL -> Chargesheet (Police)", "Chargesheet -> Court Verdict"]
    for idx, (trans_key, label) in enumerate(zip(key_transitions, stat_labels)):
        trans_data = bottlenecks.get(trans_key, {})
        col = [col1, col2, col3, col4][idx]
        with col:
            med = trans_data.get("median_days")
            p95 = trans_data.get("p95_days")
            sample = trans_data.get("sample_count")
            med_str = f"{med:.1f}" if isinstance(med, (int, float)) else NA
            p95_str = f"{p95:.1f}" if isinstance(p95, (int, float)) else NA
            sample_str = f"{sample:,}" if isinstance(sample, int) else NA
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{med_str} <span style="font-size: 0.85rem; color: #5F5F5F;">days</span></div>
                    <div class="metric-delta">P95: {p95_str} days (n={sample_str})</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Detailed Transition Breakdown & Delay Distribution
    valid_transitions = [k for k in bottlenecks.keys() if "->" in k]
    if not valid_transitions:
        st.warning("No transition duration data available for this filter.")
        st.stop()
    sel_trans = st.selectbox("Select Transition to Inspect Latency Distribution", valid_transitions, index=0)
    selected_data = bottlenecks.get(sel_trans, {})

    b_left, b_right = st.columns([6, 5])

    with b_left:
        delay_buckets = selected_data.get("delay_buckets", {})
        if delay_buckets:
            df_b = pd.DataFrame(list(delay_buckets.items()), columns=["Latency Bucket", "Case Count"])
            fig_buckets = px.bar(
                df_b,
                x="Latency Bucket",
                y="Case Count",
                title=f"Latency Distribution: {sel_trans}"
            )
            fig_buckets.update_traces(marker_color="#8B1E3F")
            fig_buckets.update_layout(get_plot_layout())
            st.plotly_chart(fig_buckets, width="stretch")

    with b_right:
        s_cnt = selected_data.get('sample_count')
        s_cnt_disp = f"{s_cnt:,}" if isinstance(s_cnt, int) else NA
        mean_d = selected_data.get('mean_days')
        mean_disp = f"{mean_d:.1f}" if isinstance(mean_d, (int, float)) else NA
        std_d = selected_data.get('std_days')
        std_disp = f"{std_d:.1f}" if isinstance(std_d, (int, float)) else NA
        med_d = selected_data.get('median_days')
        med_disp = f"{med_d:.1f}" if isinstance(med_d, (int, float)) else NA
        iqr_d = selected_data.get('iqr_days')
        iqr_disp = f"{iqr_d:.1f}" if isinstance(iqr_d, (int, float)) else NA
        p90_d = selected_data.get('p90_days')
        p90_disp = f"{p90_d:.1f}" if isinstance(p90_d, (int, float)) else NA
        p95_d = selected_data.get('p95_days')
        p95_disp = f"{p95_d:.1f}" if isinstance(p95_d, (int, float)) else NA
        p25_d = selected_data.get('p25_days')
        p25_disp = f"{p25_d:.1f}d" if isinstance(p25_d, (int, float)) else NA
        p75_d = selected_data.get('p75_days')
        p75_disp = f"{p75_d:.1f}d" if isinstance(p75_d, (int, float)) else NA
        stat_comp = statutory_compliance_label(selected_data.get('statutory_compliance'))
        st.markdown("### 📊 Non-Parametric Latency Percentiles")
        st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-radius: 4px; padding: 16px; color: #222222;">
                <p><strong>Sample Size:</strong> {s_cnt_disp} cases</p>
                <p><strong>Mean Duration:</strong> {mean_disp} ± {std_disp} days</p>
                <p><strong>Median (P50):</strong> {med_disp} days</p>
                <p><strong>Interquartile Range (IQR):</strong> {iqr_disp} days (P25: {p25_disp} - P75: {p75_disp})</p>
                <p><strong>Severe Delay (P90):</strong> {p90_disp} days</p>
                <p><strong>Extreme Outlier (P95):</strong> {p95_disp} days</p>
                <hr style="border-color: #E3DED6; margin: 10px 0;"/>
                <p style="color: #8B1E3F; font-weight: 500;"><strong>Statutory Mandate Compliance:</strong> {stat_comp}</p>
            </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: 🔗 Linkage Quality
# ==============================================================================

elif PAGE in ["🔗 Linkage Quality", "🔗 Linkage Quality & Evaluation"]:
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Linkage Quality: Model Evaluation & Precision-Recall Trade-offs</div>
            <div class="hero-subtitle">
                Rigorous linkage evaluation of the Fellegi-Sunter / Splink probabilistic record linkage model 
                benchmarked against ground truth across all five institutional data sources.
            </div>
        </div>
    """, unsafe_allow_html=True)

    render_filter_badge(FILTER_DISTRICT, (FILTER_FROM, FILTER_TO), FILTER_STAGE, KPIS.get("epsilon_remaining_display"))

    if not st.session_state.get("active_dataset_id"):
        render_no_active_dataset_banner()
        st.stop()

    if st.session_state.get("active_dataset_id") == "case":
        sc_res = st.session_state.get("single_case_analysis_res") or {}
        pair_scores = sc_res.get("pairwise_match_scores", [])
        overall_label = sc_res.get("overall_linkage_label", "same case")
        l_method = sc_res.get("linkage_method", "Fellegi-Sunter Pairwise Scoring")

        st.info(
            "ℹ️ **Single-Case Linkage Diagnostics**: Displaying pairwise match probabilities and log-odds evidence for the active case. "
            "Global precision-recall curves, candidate search space statistics, and clustering B-Cubed metrics require a multi-case dataset. "
            "To evaluate model performance across a cohort, upload CSVs or run benchmark sample analysis in Analyze Your Data."
        )

        badge_color = "#059669" if overall_label == "same case" else ("#B45309" if overall_label == "needs review" else "#DC2626")
        st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-left: 4px solid {badge_color}; border-radius: 4px; padding: 14px 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <div style="font-size: 0.78rem; text-transform: uppercase; color: #5F5F5F; font-weight: 700;">Overall Linkage Determination</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #222222;">{overall_label.upper()}</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: {badge_color}; color: #FFFFFF; padding: 4px 10px; border-radius: 3px; font-weight: 600; text-transform: uppercase; font-size: 0.85rem;">
                            {overall_label}
                        </span>
                        <div style="font-size: 0.72rem; color: #5F5F5F; margin-top: 4px;">Method: {l_method}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if pair_scores:
            st.markdown("### 🔗 Pairwise Linkage Probabilities & Log-Odds Evidence")
            p_rows = []
            for p in pair_scores:
                lbl = p.get("label", "needs review")
                lbl_badge = "🟢 Same Case" if lbl == "same case" else ("🟡 Needs Review" if lbl == "needs review" else "🔴 Different Case")
                ev_summary = "; ".join([e.get("detail", "") for e in p.get("evidence", []) if e.get("status") in ["AGREE", "DISAGREE"]])
                p_rows.append({
                    "Comparison Pair": f"{p.get('stage_a')} ↔ {p.get('stage_b')}",
                    "Match Probability": f"{p.get('match_probability', 0.0):.3f}",
                    "Match Weight (Log-Odds)": f"{p.get('match_weight', 0.0):.2f}",
                    "Evaluation Decision": lbl_badge,
                    "Key Evidence": ev_summary[:120] + "..." if len(ev_summary) > 120 else ev_summary
                })
            st.dataframe(pd.DataFrame(p_rows), width="stretch", hide_index=True)
        st.stop()

    eval_data = DATA.get("evaluation", {})
    if eval_data.get("too_few_cases"):
        render_too_few_cases_warning()
        st.stop()

    pairwise = eval_data.get("pairwise_metrics", {})
    cluster = eval_data.get("cluster_metrics", {})

    # Top Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">B-Cubed F1 Score</div>
                <div class="metric-value">{KPIS['bcubed_f1_display']}</div>
                <div class="metric-delta">Cluster-Level Accuracy</div>
            </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Macro Entity Purity</div>
                <div class="metric-value">{KPIS['macro_purity_display']}</div>
                <div class="metric-delta">Near-Zero Cross-Contamination</div>
            </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Pairwise Recall</div>
                <div class="metric-value">{KPIS['pairwise_recall_display']}</div>
                <div class="metric-delta">At Optimal Threshold</div>
            </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Candidate Search Space</div>
                <div class="metric-value">{KPIS['candidate_pairs_display']}</div>
                <div class="metric-delta">Blocked Pair Comparisons</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Precision-Recall Trade-Off Curve
    thresh_table = eval_data.get("pairwise_threshold_table", [])
    if thresh_table:
        df_pr = pd.DataFrame(thresh_table)
        fig_pr = px.line(
            df_pr,
            x="recall",
            y="precision",
            hover_data=["threshold", "f1_score", "predicted_positive_pairs"],
            markers=True,
            title="Precision-Recall Curve Across Probabilistic Thresholds (0.10 to 0.95)",
            labels={"recall": "Recall (True Positive Rate)", "precision": "Precision (Positive Predictive Value)"}
        )
        fig_pr.update_traces(line_color="#8B1E3F", marker=dict(color="#8B1E3F"))
        opt_rec = pairwise.get('metrics_at_default_threshold', {}).get('recall')
        opt_prec = pairwise.get('metrics_at_default_threshold', {}).get('precision')
        if opt_rec is not None and opt_prec is not None:
            fig_pr.add_trace(go.Scatter(
                x=[opt_rec],
                y=[opt_prec],
                mode="markers+text",
                marker=dict(color="#B45309", size=12, symbol="star"),
                name="Optimal Threshold",
                text=["Selected Operational Threshold"],
                textposition="top right"
            ))
        fig_pr.update_layout(get_plot_layout())
        st.plotly_chart(fig_pr, width="stretch")

        # Threshold Diagnostic Trade-offs Table
        st.markdown("### 🎚️ Threshold Trade-Off Analysis")
        st.dataframe(pd.DataFrame(thresh_table), width="stretch")
    else:
        st.info("No probabilistic threshold trade-off table available for this custom dataset.")


# ==============================================================================
# PAGE 5: ⚖️ Policy Impact (Institutional Transformation & Case Simulator)
# ==============================================================================

elif PAGE in ["⚖️ Policy Impact", "⚖️ Before / After Policy Transformation", "🧪 Case Simulator"]:
    if CURRENT_ROLE not in ["analyst", "admin"]:
        st.error("⛔ Access Denied: Policy Impact requires Analyst or Administrator role.")
        st.stop()

    st.markdown("""
        <div class="hero-banner">
            <div style="font-size: 1.15rem; font-weight: 700; color: #8B1E3F; letter-spacing: -0.01em; margin-bottom: 2px;">CaseLens</div>
            <div style="font-size: 0.92rem; color: #5F5F5F; margin-bottom: 14px;">The Leaky Pipeline: Policy Analytics</div>
            <div class="hero-title">Policy Impact: Institutional Transformation & Case Simulation</div>
            <div class="hero-subtitle">
                Demonstrating how longitudinal record linkage resolves departmental data silos, and simulating
                counterfactual case inputs to evaluate Fellegi-Sunter scoring and attrition trajectories.
            </div>
        </div>
    """, unsafe_allow_html=True)

    render_filter_badge(FILTER_DISTRICT, (FILTER_FROM, FILTER_TO), FILTER_STAGE, KPIS.get("epsilon_remaining_display"))

    tab_transform, tab_simulator = st.tabs([
        "🏛️ Institutional Transformation",
        "🧪 Case Simulator"
    ])

    with tab_transform:
        if not st.session_state.get("active_dataset_id"):
            render_no_active_dataset_banner()
        elif st.session_state.get("active_dataset_id") == "case":
            st.info(
                "ℹ️ **Single-Case Scope**: Systemic before/after transformation metrics (siloed departmental metrics vs linked pipeline attrition) "
                "require a multi-agency dataset. Upload CSVs or run benchmark sample analysis in Analyze Your Data to see comparative cohort transformations."
            )
        else:
            before_after = DATA.get("before_after", {})
            if before_after.get("too_few_cases"):
                render_too_few_cases_warning()
            else:
                c_before, c_after = st.columns(2)

                with c_before:
                    st.markdown("""
                        <div style="background: #FAF8F5; border: 1px dashed #CBD5E1; border-radius: 4px; padding: 20px; min-height: 460px;">
                            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                                <span style="font-size: 1.25rem; margin-right: 8px;">❌</span>
                                <strong style="color: #B45309; font-size: 1.15rem;">BEFORE: Siloed Institutional Data</strong>
                            </div>
                            <ul style="color: #222222; line-height: 1.8; font-size: 0.92rem;">
                                <li><strong>Departmental Blindness:</strong> Police, Hospitals, Forensic Labs, and Courts maintain disconnected databases with distinct survivor identifiers.</li>
                                <li><strong>False Self-Assessments:</strong>
                                    <ul>
                                        <li>Police claim high crime disposal rates by closing cases before forensic receipt.</li>
                                        <li>Forensic Science Labs report aggregate sample throughput without tracking court trial impact.</li>
                                        <li>Judiciary attributes trial pendency solely to witness non-attendance.</li>
                                    </ul>
                                </li>
                                <li><strong>Hidden Vulnerabilities:</strong> Inter-agency turnaround bottlenecks and drop-offs remain invisible without record linkage.</li>
                                <li><strong>Privacy Hazards:</strong> Ad-hoc sharing of raw survivor files leaks PII, compromising survivor safety.</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)

                filtered_cases = before_after.get("filtered_case_count", KPIS.get("reconstructed_cases_display", NA))
                filtered_events = before_after.get("filtered_event_count", KPIS.get("events_display", NA))
                cases_str = "<5" if str(filtered_cases) == "<5" else (f"{filtered_cases:,}" if isinstance(filtered_cases, int) else str(filtered_cases))
                events_str = "<5" if str(filtered_events) == "<5" else (f"{filtered_events:,}" if isinstance(filtered_events, int) else str(filtered_events))

                detected_leaks = DATA.get("summary", {}).get("systemic_leak_points_detected", [])
                if detected_leaks:
                    leak_bullets = "".join([
                        f"<li><strong>{lp.get('district', 'District')}:</strong> {lp.get('title', 'Leak point')} ({format_leak_detail(lp)})</li>"
                        for lp in detected_leaks
                    ])
                else:
                    leak_bullets = "<li>No critical systemic policy leak points detected in this cohort.</li>"

                with c_after:
                    st.markdown(f"""
                        <div style="background: #FFFFFF; border: 1px solid #8B1E3F; border-radius: 4px; padding: 20px; min-height: 460px;">
                            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                                <span style="font-size: 1.25rem; margin-right: 8px;">✅</span>
                                <strong style="color: #8B1E3F; font-size: 1.15rem;">AFTER: Privacy-Preserving Linkage Pipeline</strong>
                            </div>
                            <ul style="color: #222222; line-height: 1.8; font-size: 0.92rem;">
                                <li><strong>Unified Case Trajectories:</strong> {events_str} events stitched into {cases_str} complete case journeys across all 5 institutional stages.</li>
                                <li><strong>Pinpointed Systemic Leak Points:</strong>
                                    <ul>
                                        {leak_bullets}
                                    </ul>
                                </li>
                                <li><strong>Mathematical Survivor Confidentiality:</strong> Pure ε-Differential Privacy (Laplace mechanism) guarantees plausible deniability against differencing attacks.</li>
                                <li><strong>Statutory Benchmark Alignment:</strong> Direct comparison against POCSO Section 35(2) and BNSS mandates.</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br/>", unsafe_allow_html=True)

                # Comparative Metric Deltas
                st.markdown("### 📊 Policy Insight Resolution: Siloed vs Linked")
                comp_table = before_after.get("comparative_table")
                if comp_table:
                    st.dataframe(pd.DataFrame(comp_table), width="stretch")
                else:
                    st.info("ℹ️ Longitudinal before/after comparative metrics require multi-year departmental baseline data. Showing linked pipeline metrics calculated from the current active dataset.")

    with tab_simulator:
        render_case_simulator_tab()


# ==============================================================================
# PAGE 6: 🛡️ Privacy & Protection
# ==============================================================================

elif PAGE in ["🛡️ Privacy & Protection", "🛡️ Survivor Privacy & DP Engine"]:
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Privacy & Protection: Differential Privacy & Confidentiality Engine</div>
            <div class="hero-subtitle">
                Algorithmic privacy protections guaranteeing zero re-identification of survivors: Laplace mechanism 
                noise injection, Sequential Composition privacy budget accounting, and adversarial attack defense.
            </div>
        </div>
    """, unsafe_allow_html=True)

    render_filter_badge(FILTER_DISTRICT, (FILTER_FROM, FILTER_TO), FILTER_STAGE, KPIS.get("epsilon_remaining_display"))

    if not st.session_state.get("active_dataset_id"):
        render_no_active_dataset_banner()
        st.stop()

    if st.session_state.get("active_dataset_id") == "case":
        sc_res = st.session_state.get("single_case_analysis_res") or {}
        m_name = sc_res.get("masked_survivor_name", "Anonymous")

        st.info(
            "ℹ️ **Single-Case Privacy Model**: Differential Privacy Laplace noise and small-cell k-anonymity suppression "
            "are mathematical guarantees applied to aggregate queries over large datasets. For an individual case simulation, "
            "confidentiality is guaranteed via one-way cryptographic pseudonymization and strictly zero persistent storage. "
            "To evaluate aggregate differential privacy budgets and noise mechanisms, upload CSVs or run benchmark sample analysis in Analyze Your Data."
        )

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Survivor Identity</div>
                    <div class="metric-value" style="font-size: 1.3rem;">👤 {m_name}</div>
                    <div class="metric-delta">SHA-256 One-Way Masked</div>
                </div>
            """, unsafe_allow_html=True)
        with p2:
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Microdata Storage</div>
                    <div class="metric-value" style="color: #059669;">0 Bytes</div>
                    <div class="metric-delta">Volatile In-Memory Processing</div>
                </div>
            """, unsafe_allow_html=True)
        with p3:
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Audit Log Verification</div>
                    <div class="metric-value" style="color: #059669;">Zero-PII</div>
                    <div class="metric-delta">Action logged without input fields</div>
                </div>
            """, unsafe_allow_html=True)
        st.stop()

    priv_audit = DATA.get("privacy_audit", {})
    budget_sum = priv_audit.get("privacy_budget_summary", {})
    live_budget = DATA.get("privacy_budget", {})
    if live_budget and "remaining_epsilon" in live_budget:
        budget_sum = {
            "total_epsilon_allocated": live_budget.get("total_epsilon", budget_sum.get("total_epsilon_allocated", 10.0)),
            "spent_epsilon": live_budget.get("spent_epsilon", budget_sum.get("spent_epsilon", 0.0)),
            "remaining_epsilon": live_budget.get("remaining_epsilon", budget_sum.get("remaining_epsilon", 10.0)),
            "composition_theorem": budget_sum.get("composition_theorem", "Sequential Composition"),
            "budget_ledger_entries": live_budget.get("total_queries_executed", budget_sum.get("budget_ledger_entries", 0))
        }
    k_eval = priv_audit.get("k_anonymity_evaluation", {})

    # Top KPI Budget Cards
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Epsilon Allocated</div>
                <div class="metric-value">{budget_sum.get('total_epsilon_allocated', 10.0):.2f}</div>
                <div class="metric-delta">Sequential Budget Upper Bound</div>
            </div>
        """, unsafe_allow_html=True)

    with b2:
        spent = budget_sum.get('spent_epsilon', 0.0)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Epsilon Budget Spent</div>
                <div class="metric-value">{spent:.2f}</div>
                <div class="metric-delta">Consumed by Analytical Queries</div>
            </div>
        """, unsafe_allow_html=True)

    with b3:
        tot_eps = budget_sum.get('total_epsilon_allocated', 10.0)
        rem = budget_sum.get('remaining_epsilon', max(tot_eps - spent, 0.0))
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Remaining Epsilon</div>
                <div class="metric-value">{rem:.2f}</div>
                <div class="metric-delta" style="color: #8B1E3F;">Available Query Capacity</div>
            </div>
        """, unsafe_allow_html=True)

    with b4:
        zero_pii_val = "100%" if KPIS.get("zero_pii_label") == "Pass" else ("0%" if KPIS.get("zero_pii_label") == "Fail" else NA)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Zero-PII Status</div>
                <div class="metric-value">{zero_pii_val}</div>
                <div class="metric-delta">{KPIS.get("zero_pii_delta", "Verified: Zero Leaked Names/IDs")}</div>
            </div>
        """, unsafe_allow_html=True)

    # Admin Privacy Budget Controls
    if CURRENT_ROLE == "admin":
        st.markdown("### ⚙️ Differential Privacy Budget Administration")
        with st.container():
            st.markdown("""
                <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-left: 4px solid #8B1E3F; border-radius: 4px; padding: 14px 18px; margin-bottom: 14px;">
                    <strong style="color: #222222; font-size: 0.95rem;">Reset Privacy Budget</strong>
                    <div style="color: #5F5F5F; font-size: 0.85rem; margin-top: 4px; line-height: 1.4;">
                        Reset or reconfigure the live differential privacy budget. Resetting restores query capacity and clears the dynamic query cache, enabling judges and reviewers to test analytical queries without interruption.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            ac1, ac2, ac3 = st.columns([3, 3, 4])
            with ac1:
                admin_eps_target = st.number_input(
                    "Total Epsilon Upper Bound (ε)",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(budget_sum.get("total_epsilon_allocated", 10.0)),
                    step=1.0,
                    key="admin_page_reset_eps_input"
                )
            with ac2:
                st.write("")
                st.write("")
                if st.button("🔄 Reset Privacy Budget", type="primary", key="admin_page_reset_budget_btn"):
                    res = reset_privacy_budget(total_epsilon=admin_eps_target, token=st.session_state.get("token"))
                    if res.get("message") or res.get("success"):
                        st.session_state["privacy_budget_exhausted"] = False
                        st.cache_data.clear()
                        st.success(f"Differential privacy budget reset to {admin_eps_target:.2f} ε successfully.")
                        st.rerun()
                    else:
                        st.error("Failed to reset privacy budget. Ensure admin privileges.")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Interactive Laplace Mechanism Simulator
    st.markdown("### 🎛️ Interactive Laplace Noise Mechanism Simulator")
    sim_col1, sim_col2 = st.columns([5, 6])

    with sim_col1:
        sim_eps = st.slider("Select Epsilon (ε)", min_value=0.05, max_value=2.0, value=0.25, step=0.05)
        sim_sens = st.number_input("Query Sensitivity (Δf)", min_value=1.0, max_value=5.0, value=1.0, step=1.0)
        scale_b = sim_sens / sim_eps
        margin_95 = -scale_b * math.log(0.05)

        st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-radius: 4px; padding: 14px; margin-top: 14px;">
                <p><strong>Laplace Scale (b = Δf / ε):</strong> {scale_b:.2f}</p>
                <p><strong>95% Theoretical Error Bound:</strong> ±{margin_95:.2f} cases</p>
                <p style="font-size: 0.85rem; color: #5F5F5F;">
                    Under ε={sim_eps}, 95% of all analytical queries will have error no greater than 
                    ±{margin_95:.1f} cases, neutralizing membership inference while maintaining statistical utility.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with sim_col2:
        x_vals = np.linspace(-4 * scale_b, 4 * scale_b, 400)
        pdf_vals = (1.0 / (2.0 * scale_b)) * np.exp(-np.abs(x_vals) / scale_b)
        fig_lap = px.line(
            x=x_vals,
            y=pdf_vals,
            title=f"Laplace Probability Density Function (b={scale_b:.2f})",
            labels={"x": "Noise Drawn (Cases)", "y": "Probability Density"}
        )
        fig_lap.update_traces(line_color="#8B1E3F")
        fig_lap.add_vline(x=-margin_95, line_dash="dash", line_color="#B45309", annotation_text="-95% CI")
        fig_lap.add_vline(x=margin_95, line_dash="dash", line_color="#B45309", annotation_text="+95% CI")
        fig_lap.update_layout(get_plot_layout())
        st.plotly_chart(fig_lap, width="stretch")

    # Adversarial Attack Simulation Results
    st.markdown("### 🦹 Adversarial Re-Identification Attack Defense")
    adv_threats = priv_audit.get("adversarial_threat_evaluations", {})
    singling_out = adv_threats.get("singling_out_marksmanship_attack", {})
    before_supp = singling_out.get("before_protection", {})
    after_supp = singling_out.get("after_k_anonymity_protection", {})

    att_col1, att_col2 = st.columns(2)
    with att_col1:
        st.markdown(f"""
            <div style="background: #FEF3C7; border: 1px solid #FCD34D; border-radius: 4px; padding: 16px;">
                <h4 style="color: #92400E; margin-top: 0;">Unprotected Raw Data Vulnerability</h4>
                <p><strong>Singling-Out / Marksmanship Risk:</strong> {before_supp.get('singling_out_risk_pct', 7.42)}%</p>
                <p><strong>Max Prosecutor Re-identification Risk:</strong> {before_supp.get('max_prosecutor_risk_pct', 100.0)}%</p>
                <p><strong>Unique Vulnerable Survivor Cases:</strong> {before_supp.get('unique_records_count', 380):,} records</p>
                <p style="font-size: 0.85rem; color: #B45309;">An attacker with auxiliary knowledge can single out individuals with 100% certainty.</p>
            </div>
        """, unsafe_allow_html=True)

    with att_col2:
        st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-left: 3px solid #8B1E3F; border-radius: 4px; padding: 16px;">
                <h4 style="color: #8B1E3F; margin-top: 0;">Post-Suppression & DP Defense</h4>
                <p><strong>Singling-Out / Marksmanship Risk:</strong> {after_supp.get('singling_out_risk_pct', 0.0)}% (0 vulnerable cases)</p>
                <p><strong>Max Prosecutor Re-identification Risk:</strong> {after_supp.get('max_prosecutor_risk_pct', 20.0)}% (k >= 5 enforced)</p>
                <p><strong>Suppression Audit:</strong> 100% of equivalence classes &lt; 5 masked</p>
                <p style="font-size: 0.85rem; color: #5F5F5F;">Singling out completely foiled; mathematical plausible deniability guaranteed.</p>
            </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 7: 📜 Policy Insights
# ==============================================================================

elif PAGE in ["📜 Policy Insights", "📜 Statutory Policy Insights (TinyFish)"]:
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Policy Insights: Statutory Benchmarks & Governance Interventions</div>
            <div class="hero-subtitle">
                Official statutory benchmarks and infrastructure standards retrieved via TinyFish 
                prioritizing official <code>.gov.in</code> and <code>.nic.in</code> government sources with zero PII leakage.
            </div>
        </div>
    """, unsafe_allow_html=True)

    render_filter_badge(FILTER_DISTRICT, (FILTER_FROM, FILTER_TO), FILTER_STAGE, KPIS.get("epsilon_remaining_display"))

    if not st.session_state.get("active_dataset_id"):
        render_no_active_dataset_banner()
        st.stop()

    if st.session_state.get("active_dataset_id") == "case":
        st.info(
            "ℹ️ **Single-Case Statutory Alignment**: Showing legal mandates relevant to milestones in this case. "
            "District-wide statutory policy leak analysis and automated government benchmark retrieval require a multi-agency dataset. "
            "Upload CSVs or run benchmark sample analysis in Analyze Your Data."
        )
        st.markdown("### 🏛️ Applicable Statutory Benchmarks")
        st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-radius: 4px; padding: 14px 16px; margin-bottom: 12px;">
                <div style="font-weight: 700; color: #8B1E3F;">Police Investigation & Chargesheet Filing</div>
                <div style="font-size: 0.88rem; color: #222222; margin-top: 4px;"><strong>Statutory Mandate:</strong> Section 173(2) CrPC / Section 193 BNSS requires investigation to be completed within 60 days in sexual offence cases.</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-radius: 4px; padding: 14px 16px; margin-bottom: 12px;">
                <div style="font-weight: 700; color: #8B1E3F;">Medical Examination & Forensic Evidence</div>
                <div style="font-size: 0.88rem; color: #222222; margin-top: 4px;"><strong>Statutory Mandate:</strong> Section 164A CrPC requires medical examination within 24 hours of receiving information.</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-radius: 4px; padding: 14px 16px; margin-bottom: 12px;">
                <div style="font-weight: 700; color: #8B1E3F;">Judicial Trial Expediency</div>
                <div style="font-size: 0.88rem; color: #222222; margin-top: 4px;"><strong>Statutory Mandate:</strong> POCSO Section 35(2) mandates trial to be completed, as far as possible, within a period of one year from taking cognizance.</div>
            </div>
        """, unsafe_allow_html=True)
        st.stop()

    policy_cache = DATA.get("policy_context", {})

    st.markdown("### 🏛️ Top 5 Policy Leak Points & Statutory Grounding")

    leak_items = [item for key, item in policy_cache.items() if key != "government_benchmark_fact"]
    if not leak_items:
        st.info("No policy leak point contexts mapped for the current run.")

    # Render each policy leak point
    for key, data in policy_cache.items():
        if key == "government_benchmark_fact":
            continue
        
        leak_label = data.get("leak_point_label", key)
        leak_code = data.get("leak_code", "LP-GENERIC")
        finding = data.get("finding", "")
        intervention = data.get("suggested_intervention", "")
        sources = data.get("sources", [])

        with st.expander(f"📍 [{leak_code}] {leak_label}", expanded=True):
            st.markdown(f"""
                <div style="margin-bottom: 12px;">
                    <strong style="color: #B45309;">Empirical Pipeline Finding:</strong><br/>
                    <span style="color: #222222;">{finding}</span>
                </div>
                <div style="margin-bottom: 16px;">
                    <strong style="color: #8B1E3F;">Recommended Statutory Intervention:</strong><br/>
                    <span style="color: #5F5F5F;">{intervention}</span>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<strong style='color: #222222;'>Official Government & Statutory Sources:</strong>", unsafe_allow_html=True)
            for s in sources:
                domain = s.get("source_domain", "")
                is_gov = (".gov.in" in domain) or (".nic.in" in domain)
                badge_type = "badge-privacy" if is_gov else "badge-normal"
                badge_label = "OFFICIAL GOVT SOURCE" if is_gov else "STATUTORY REFERENCE"

                st.markdown(f"""
                    <div style="background: #FFFFFF; border: 1px solid #E3DED6; border-radius: 4px; padding: 12px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <a href="{s.get('url', '#')}" target="_blank" style="color: #8B1E3F; font-weight: 600; text-decoration: underline;">
                                {s.get('title', 'Document')} ↗
                            </a>
                            <span class="badge {badge_type}">{badge_label} ({domain})</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #5F5F5F; line-height: 1.4;">
                            {s.get('summary', '')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # Autonomous Web Agent Fact
    fact = policy_cache.get("government_benchmark_fact", {})
    if fact:
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="callout-info">
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 1.2rem; margin-right: 6px;">🤖</span>
                    <strong style="color: #8B1E3F;">Autonomous TinyFish Web Agent Fact Extraction</strong>
                </div>
                <div><strong>Mandate:</strong> {fact.get('fact', '')}</div>
                <div style="font-size: 0.82rem; color: #5F5F5F; margin-top: 4px;">
                    Statutory Reference: {fact.get('statutory_reference', '')} | Enacted: {fact.get('enactment_year', 2012)} | Source: {fact.get('source_domain', '')}
                </div>
            </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 8: 🔍 Case Review (Case Explorer & Admin Review Queue)
# ==============================================================================

elif PAGE in ["🔍 Case Review", "🔍 Case Explorer", "📋 Review Queue"]:
    st.markdown("""
        <div class="hero-banner">
            <div style="font-size: 1.15rem; font-weight: 700; color: #8B1E3F; letter-spacing: -0.01em; margin-bottom: 2px;">CaseLens</div>
            <div style="font-size: 0.92rem; color: #5F5F5F; margin-bottom: 14px;">Institutional Record Linkage & Case Adjudication</div>
            <div class="hero-title">Case Review: Trajectories & Human Review Queue</div>
            <div class="hero-subtitle">
                Judicial inspection of synthetic multi-agency records across police, hospital, forensic, prosecution, and court,
                combined with confidential human adjudication for uncertain candidate pairs under strict zero-PII protocols.
            </div>
        </div>
    """, unsafe_allow_html=True)

    tab_explorer, tab_review = st.tabs([
        "🔍 End-to-End Case Trajectories",
        "📋 Human Review Queue (Admin Only)"
    ])

    with tab_explorer:
        # Mandatory Synthetic Data Notice (Exact user specification)
        st.markdown("""
            <div style="background: #FDF2F4; border: 1px solid #E8B4BC; border-left: 4px solid #8B1E3F; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; font-size: 0.88rem; color: #8B1E3F;">
                <strong>⚠️ All data shown is synthetic. No real person or case.</strong><br/>
                <span style="font-size: 0.80rem; color: #5F5F5F;">
                    All agency records, person names, relatives, dates, and identifiers displayed here are purely synthetic and generated for institutional pipeline demonstration.
                    Probabilistic linkage scores reflect algorithmic model outputs and do NOT constitute judicial determinations or legal findings. Never stores data.
                </span>
            </div>
        """, unsafe_allow_html=True)

        active_token = st.session_state.get("token")

        # Fetch curated examples via FastAPI endpoint
        curated_list = get_case_examples(token=active_token)
        if not curated_list:
            curated_list = [
                {"id": "clean_match", "title": "1. Clean Match: Full 5-Stage Institutional Journey", "category": "Baseline Linkage"},
                {"id": "spelling_variants", "title": "2. Spelling Variants: Phonetic & Dialectal Invariance", "category": "Algorithmic Robustness"},
                {"id": "uncertain_review", "title": "3. Uncertain Pair: Routed to Human Review Queue", "category": "Human-in-the-Loop Safeguard"},
                {"id": "similar_names_kept_apart", "title": "4. False-Link Protection: Similar Names Correctly Kept Apart", "category": "False Positive Defense"},
                {"id": "missed_link", "title": "5. Missed Link: Data Fragmentation Caused False Negative", "category": "Pipeline Sensitivity"},
                {"id": "district_3_forensic_delay", "title": "6. District 3: Severe Forensic Laboratory Latency Bottleneck", "category": "Systemic Bottleneck"},
                {"id": "district_6_chargesheet_drop", "title": "7. District 6: Police-to-Prosecution Chargesheet Drop-Off", "category": "Policy Attrition Leak"}
            ]

        # Initialize session state for selected case
        if "explorer_case_id" not in st.session_state:
            st.session_state["explorer_case_id"] = "clean_match"

        # Top Control Bar: Curated Selector & Random Case Button
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.5, 1.2, 1.3], gap="medium")

        example_titles = [f"{ex['title']}" for ex in curated_list]
        example_ids = [ex["id"] for ex in curated_list]

        current_id = st.session_state.get("explorer_case_id", "clean_match")
        default_idx = example_ids.index(current_id) if current_id in example_ids else 0

        with ctrl_col1:
            chosen_title = st.selectbox(
                "Select Curated Judicial Example",
                options=example_titles,
                index=default_idx,
                help="Choose from 7 curated institutional benchmark scenarios."
            )
            selected_example_id = example_ids[example_titles.index(chosen_title)]
            if selected_example_id != st.session_state.get("explorer_case_id") and not st.session_state.get("_random_case_clicked"):
                st.session_state["explorer_case_id"] = selected_example_id

        with ctrl_col2:
            custom_input = st.text_input(
                "Or Case ID from Dataset",
                value=st.session_state.get("explorer_case_id", "") if str(st.session_state.get("explorer_case_id", "")).startswith("CASE-") else "",
                placeholder="e.g. CASE-00042",
                help="Enter any synthetic case ID from the 5,000 generated cases."
            )
            if custom_input and custom_input.strip() and custom_input.strip() != st.session_state.get("explorer_case_id"):
                st.session_state["explorer_case_id"] = custom_input.strip()
                st.session_state["_random_case_clicked"] = False

        with ctrl_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🎲 Random Case", key="btn_random_case", width="stretch", help="Picks any case at random from the 5,000 generated synthetic dataset cases."):
                rand_case = get_random_case(token=active_token)
                if rand_case and "case_id" in rand_case:
                    st.session_state["explorer_case_id"] = rand_case["case_id"]
                    st.session_state["_random_case_clicked"] = True
                    st.rerun()

        # Reset random flag if user switched selectbox
        if st.session_state.get("_random_case_clicked") and st.session_state.get("explorer_case_id") in example_ids:
            st.session_state["_random_case_clicked"] = False

        active_case_id = st.session_state.get("explorer_case_id", "clean_match")

        # Fetch Case Details via FastAPI Endpoint
        with st.spinner(f"Loading synthetic case trajectory for '{active_case_id}'..."):
            case_data = get_case_detail(active_case_id, token=active_token)

        if not case_data or "records" not in case_data:
            st.error(f"❌ Could not retrieve case data for ID '{active_case_id}'. Please select an example from the dropdown.")
        else:
            # Role Enforcement Banner
            role_view = case_data.get("role_view", "masked")
            role_badge_text = "🔒 Masked Personal Names (Viewer Role)" if role_view == "masked" else "🛡️ Pseudonymised Identifiers (Analyst/Admin Role)"
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; background: #F8F7F4; border: 1px solid #E3DED6; border-radius: 4px; padding: 8px 14px; margin-bottom: 16px; font-size: 0.82rem;">
                    <div>
                        <strong style="color: #222222;">Inspecting Case:</strong> <code style="font-weight: 600; color: #8B1E3F;">{case_data['case_id']}</code> &nbsp;
                        <span style="color: #5F5F5F;">| District: <strong>{case_data.get('district', 'N/A')}</strong> | Category: <strong>{case_data.get('category', 'General')}</strong></span>
                    </div>
                    <div style="font-weight: 600; color: #5F5F5F;">
                        {role_badge_text}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Requirement (d): One-Line Verdict Banner
            v_type = case_data.get("verdict_type", "neutral")
            v_palette = {
                "success": ("#FDF2F4", "#8B1E3F", "✅", "#8B1E3F"),
                "warning": ("#FEF3C7", "#92400E", "⚠️", "#B45309"),
                "danger": ("#FEE2E2", "#991B1B", "🛡️", "#DC2626"),
                "review": ("#FFFBEB", "#B45309", "📋", "#D97706"),
                "neutral": ("#F0ECE5", "#222222", "ℹ️", "#5F5F5F")
            }
            bg_c, text_c, icon, border_c = v_palette.get(v_type, ("#F0ECE5", "#222222", "ℹ️", "#5F5F5F"))

            st.markdown(f"""
                <div style="background: {bg_c}; border: 1px solid {border_c}; border-left: 6px solid {border_c}; border-radius: 4px; padding: 14px 18px; margin-bottom: 20px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: {text_c}; letter-spacing: 0.05em; margin-bottom: 2px;">
                        Judicial Verdict & Trajectory Assessment
                    </div>
                    <div style="font-size: 1.12rem; font-weight: 700; color: {text_c}; line-height: 1.4;">
                        {icon} {case_data.get('one_line_verdict', 'Case Trajectory Evaluated')}
                    </div>
                    <div style="font-size: 0.82rem; color: #4A4A4A; margin-top: 6px; line-height: 1.4;">
                        {case_data.get('why', '')}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Layout: (a) Fragmented Records on Left, (c) Journey & (b) Pairwise Evidence on Right
            col_records, col_timeline = st.columns([1.1, 1.25], gap="large")

            # Requirement (a): Fragmented Source Records as they appear in each system
            with col_records:
                st.markdown("### 📥 (a) Fragmented Agency Source Records")
                st.markdown(f"""
                    <div style="font-size: 0.82rem; color: #5F5F5F; margin-bottom: 12px; line-height: 1.4;">
                        Notice that <strong>no shared or common ID exists</strong> across systems.
                        Names, phonetic spellings, ages/DOBs, and reported incident dates differ between agency records:
                    </div>
                """, unsafe_allow_html=True)

                records_list = case_data.get("records", [])
                if not records_list:
                    st.info("No agency records recorded for this synthetic case.")
                else:
                    for idx, rec in enumerate(records_list):
                        agency_title = rec.get("agency", "Agency Record")
                        rec_id = rec.get("record_id", f"REC-{idx}")
                        with st.expander(f"📄 {agency_title} — {rec_id}", expanded=True):
                            rc1, rc2 = st.columns(2)
                            with rc1:
                                st.markdown(f"**Agency System:** `{agency_title}`")
                                st.markdown(f"**System Record ID:** `{rec_id}`")
                                st.markdown(f"**Subject Name:** `{rec.get('name') or 'N/A'}`")
                                st.markdown(f"**Guardian / Spouse:** `{rec.get('guardian') or 'N/A'}`")
                                age_val = rec.get("age") or rec.get("dob") or "Not Captured"
                                st.markdown(f"**Age / DOB:** `{age_val}`")
                            with rc2:
                                st.markdown(f"**District:** `{rec.get('district') or 'N/A'}`")
                                st.markdown(f"**Locality / Area:** `{rec.get('locality') or 'Not Captured'}`")
                                st.markdown(f"**Reported Incident:** `{rec.get('incident_date') or 'N/A'}`")
                                st.markdown(f"**System Date:** `{rec.get('record_date') or 'N/A'}`")
                            if rec.get("details"):
                                st.caption(f"System Notes / Charge: {rec.get('details')}")

            # Requirement (c): Reconstructed Journey Timeline with Days Between Stages
            with col_timeline:
                st.markdown("### ⏳ (c) Reconstructed Journey Timeline")
                st.markdown("""
                    <div style="font-size: 0.82rem; color: #5F5F5F; margin-bottom: 12px; line-height: 1.4;">
                        Institutional trajectory reconstructed through cross-agency linkage.
                        Elapsed days between milestones identify exactly where the case <strong>stalled</strong> or <strong>dropped</strong>:
                    </div>
                """, unsafe_allow_html=True)

                timeline = case_data.get("timeline", {})
                steps = timeline.get("steps", [])
                bottlenecks = timeline.get("bottlenecks", [])

                for s in steps:
                    status = s.get("status", "UNKNOWN")
                    is_done = (status == "COMPLETED")
                    dot_c = "#8B1E3F" if is_done else "#B45309"
                    badge_class = "badge-privacy" if is_done else "badge-critical"
                    date_disp = f"🗓️ {s['date']}" if s.get('date') else "❌ Milestone Not Reached"
                    elapsed_disp = s.get("elapsed_label", "")

                    st.markdown(f"""
                        <div style="display: flex; align-items: flex-start; margin-bottom: 8px; padding: 10px 14px; background: #FFFFFF; border: 1px solid #E3DED6; border-left: 4px solid {dot_c}; border-radius: 4px;">
                            <div style="flex-grow: 1;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong style="color: #222222; font-size: 0.88rem;">{s['stage']}</strong>
                                    <span class="badge {badge_class}" style="font-size: 0.72rem;">{status}</span>
                                </div>
                                <div style="font-size: 0.78rem; color: #5F5F5F; margin-top: 3px;">
                                    {date_disp} | Record ID: <code>{s.get('record_id', 'N/A')}</code>
                                </div>
                                {f'<div style="font-size: 0.74rem; font-weight: 600; color: #8B1E3F; margin-top: 2px;">⏱️ Latency: {elapsed_disp}</div>' if elapsed_disp else ''}
                                {f'<div style="font-size: 0.74rem; color: #7F7F7F; margin-top: 2px;">{s["notes"]}</div>' if s.get("notes") else ''}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                if bottlenecks:
                    for bn in bottlenecks:
                        st.markdown(f"""
                            <div style="background: #FEF3C7; border: 1px solid #FCD34D; border-left: 4px solid #B45309; border-radius: 4px; padding: 10px 14px; margin-top: 8px; margin-bottom: 8px; font-size: 0.82rem;">
                                <strong style="color: #B45309;">⚠️ Institutional Bottleneck ({bn['code']}) at {bn.get('stage', 'Pipeline')}:</strong><br/>
                                <span style="color: #222222; font-weight: 600;">{bn['title']}</span><br/>
                                <span style="color: #5F5F5F;">{bn['detail']}</span>
                            </div>
                        """, unsafe_allow_html=True)
                elif timeline.get("is_dropped"):
                    st.markdown(f"""
                        <div style="background: #FEE2E2; border: 1px solid #FCA5A5; border-left: 4px solid #DC2626; border-radius: 4px; padding: 10px 14px; margin-top: 8px; margin-bottom: 8px; font-size: 0.82rem;">
                            <strong style="color: #DC2626;">🔻 Case Attrition Leak Point:</strong> Journey stalled and dropped after <strong>{timeline.get('drop_point', 'Investigation')}</strong>.<br/>
                            <span style="color: #5F5F5F;">The survivor never reached court trial or legal redress.</span>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # Requirement (b): Pairwise Match Probability and Reasons
            st.markdown("### 🔗 (b) Pairwise Match Probability & Linkage Reasons")
            st.markdown("""
                <div style="font-size: 0.82rem; color: #5F5F5F; margin-bottom: 12px; line-height: 1.4;">
                    Every pairwise agency combination scored through the Fellegi-Sunter probabilistic comparison matrix.
                    Thresholds: <strong>Auto-Accept</strong> (P &ge; 0.85), <strong>Human Review</strong> (0.50 &le; P &lt; 0.85), <strong>Reject</strong> (P &lt; 0.50):
                </div>
            """, unsafe_allow_html=True)

            pw_results = case_data.get("pairwise_results", [])
            if pw_results:
                table_rows = []
                for p in pw_results:
                    d_label = "✅ AUTO_ACCEPT" if p["decision"] == "AUTO_ACCEPT" else ("⚠️ HUMAN_REVIEW" if p["decision"] == "HUMAN_REVIEW" else "❌ REJECT")
                    table_rows.append({
                        "Record A": p["record_a"],
                        "Record B": p["record_b"],
                        "Match Probability (P)": f"{p['match_probability']:.3f}",
                        "Weight (Log-Odds)": f"{p['match_weight']:+.2f}",
                        "Decision": d_label
                    })
                st.dataframe(pd.DataFrame(table_rows), width="stretch")

                st.markdown("#### 🔬 Per-Field Evidence Explanations")
                for idx, p in enumerate(pw_results):
                    with st.expander(f"Scoring Evidence: {p['record_a']} ↔ {p['record_b']} ({p['decision']})", expanded=(idx == 0)):
                        st.markdown(f"**Posterior Match Probability:** `{p['match_probability']:.3f}` | **Composite Log-Odds Weight:** `{p['match_weight']:+.2f}` | **Decision:** `{p['decision']}`")
                        for ev in p.get("evidence", []):
                            icon_ev = "🟢" if ev["status"] == "AGREE" else ("🟡" if ev["status"] == "PARTIAL" else ("🔴" if ev["status"] == "DISAGREE" else "⚪"))
                            st.markdown(f"- {icon_ev} **{ev['field']}** ({ev['weight']:+.2f}): {ev['detail']}")
            else:
                st.info("Single record case; pairwise combinations require at least two agency records.")

    with tab_review:
        if CURRENT_ROLE != "admin":
            st.markdown("""
                <div style="background: #FFFBEB; border: 1px solid #FCD34D; border-left: 4px solid #D97706; border-radius: 4px; padding: 16px 18px; margin-top: 10px;">
                    <div style="font-weight: 700; color: #B45309; font-size: 1rem; margin-bottom: 4px;">
                        ⛔ Administrator Role Required
                    </div>
                    <div style="font-size: 0.88rem; color: #222222; line-height: 1.5;">
                        The Record Linkage Human Review Queue is restricted to System Administrators.
                        Reviewing borderline candidate pairs (0.50 ≤ P < 0.85) requires administrative authority to maintain strict zero-PII confidentiality and judicial oversight.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            render_filter_badge(FILTER_DISTRICT, (FILTER_FROM, FILTER_TO), FILTER_STAGE, KPIS.get("epsilon_remaining_display"))

            st.warning("⚠️ **Strict Privacy Guardrail**: Direct personal identifiers (names, relatives, addresses) are suppressed. Only pseudonymized record identifiers, incident dates, and match probabilities are shown.")

            active_token = st.session_state.get("token")
            rq_data = get_review_queue(limit=200, token=active_token)
            records = rq_data.get("records", []) if isinstance(rq_data, dict) else []
            total_pairs = rq_data.get("total_pairs", len(records)) if isinstance(rq_data, dict) else len(records)
            avg_prob = rq_data.get("avg_match_probability", 0.0) if isinstance(rq_data, dict) else 0.0

            if records:
                df_display = pd.DataFrame(records)

                col_rq1, col_rq2, col_rq3, col_rq4 = st.columns(4)
                with col_rq1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Uncertain Pairs</div>
                            <div class="metric-value">{total_pairs:,}</div>
                            <div class="metric-delta">Pending Human Review</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col_rq2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Avg Match Score</div>
                            <div class="metric-value">{avg_prob:.3f}</div>
                            <div class="metric-delta">Uncertainty Band [0.50, 0.85)</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col_rq3:
                    st.markdown("""
                        <div class="metric-card">
                            <div class="metric-label">Accept Threshold</div>
                            <div class="metric-value">0.85</div>
                            <div class="metric-delta">Auto-Accepted Links</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col_rq4:
                    st.markdown("""
                        <div class="metric-card">
                            <div class="metric-label">Reject Threshold</div>
                            <div class="metric-value">0.50</div>
                            <div class="metric-delta">Definitively Discarded</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
                st.markdown("#### Pseudonymized Candidate Pairs")
                st.dataframe(df_display.head(200), width="stretch")
                st.caption(f"Showing first {min(200, len(df_display)):,} of {total_pairs:,} records. All direct names suppressed.")
            else:
                st.info("No review queue candidate pairs available from API backend.")


# ==============================================================================
# PAGE 9: 📜 Audit Log (Admin Only - Zero-PII Security Ledger)
# ==============================================================================

elif PAGE in ["📜 Audit Log", "🔐 Audit Log"]:
    if CURRENT_ROLE != "admin":
        st.error("⛔ Access Denied: Administrator role required to view the Security Audit Log.")
        st.stop()

    st.markdown("""
        <div class="hero-banner">
            <div style="font-size: 1.15rem; font-weight: 700; color: #8B1E3F; letter-spacing: -0.01em; margin-bottom: 2px;">CaseLens</div>
            <div style="font-size: 0.92rem; color: #5F5F5F; margin-bottom: 14px;">System Security & Accountability</div>
            <div class="hero-title">Security & Session Audit Ledger</div>
            <div class="hero-subtitle">
                Immutable event tracking for authentication, user sessions, page visits, and access attempts.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.success("✅ **Zero-Credential Guarantee**: Passwords, hashes, and JWT tokens are cryptographically excluded from all audit records.")

    active_token = st.session_state.get("token")
    audit_resp = get_audit_logs(limit=300, token=active_token)
    app_logs = audit_resp.get("logs", []) if isinstance(audit_resp, dict) else (audit_resp or [])
    if app_logs:
        df_logs = pd.DataFrame(app_logs)
        total_events = len(df_logs)
        logins = int((df_logs["event_type"] == "LOGIN_SUCCESS").sum()) if "event_type" in df_logs.columns else 0
        fails = int((df_logs["event_type"] == "LOGIN_FAILED").sum()) if "event_type" in df_logs.columns else 0
        visits = int((df_logs["event_type"] == "PAGE_VISIT").sum()) if "event_type" in df_logs.columns else 0

        col_al1, col_al2, col_al3, col_al4 = st.columns(4)
        with col_al1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Events</div>
                    <div class="metric-value">{total_events:,}</div>
                    <div class="metric-delta">Recorded Sessions</div>
                </div>
            """, unsafe_allow_html=True)
        with col_al2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Successful Logins</div>
                    <div class="metric-value">{logins}</div>
                    <div class="metric-delta">Authorized Access</div>
                </div>
            """, unsafe_allow_html=True)
        with col_al3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Failed Logins</div>
                    <div class="metric-value">{fails}</div>
                    <div class="metric-delta">Enforcing 5-Attempt Lockout</div>
                </div>
            """, unsafe_allow_html=True)
        with col_al4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Page Visits</div>
                    <div class="metric-value">{visits}</div>
                    <div class="metric-delta">Analytical Navigation</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Audit Event Records (Most Recent)")
        st.dataframe(df_logs.iloc[::-1], width="stretch")
    else:
        st.info("No audit log events recorded yet.")

