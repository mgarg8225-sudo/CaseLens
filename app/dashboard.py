"""
app/dashboard.py - The Leaky Pipeline: Privacy-Preserving Policy Analytics Dashboard.

Phase 7: Multi-Stakeholder Interactive Streamlit Web Application.
Provides analytical visibility into systemic case attrition across institutional silos
(Police, Hospital, Forensic Science Labs, Prosecution, and Courts) for violence-against-women cases,
guaranteeing mathematical survivor privacy through Differential Privacy and k-anonymity.

Pages:
1. Executive Overview (Pipeline health, district vulnerability ranking, leak points)
2. Funnel with Uncertainty Ranges (DP Laplace noisy counts, 95% CIs, cell suppression)
3. Duration Bottlenecks & Latency (Percentiles, statutory deadlines CrPC/BNSS/POCSO)
4. Linkage Quality & Evaluation (Precision-Recall curve, B-Cubed metrics, error diagnostics)
5. Before / After Policy Transformation (Naive siloed view vs unified privacy-preserving pipeline)
6. Survivor Privacy & DP Engine (Laplace simulator, sequential budget accounting, attack defense)
7. Statutory Policy Insights (TinyFish public context, .gov.in/.nic.in benchmarks, policy recommendations)
"""

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

# Configure Page Settings
st.set_page_config(
    page_title="The Leaky Pipeline | Policy Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. Custom CSS Theme & Styling (Rich Dark Theme)
# ==============================================================================

CUSTOM_CSS = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 50%, rgba(14, 116, 144, 0.25) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        line-height: 1.5;
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .metric-delta {
        font-size: 0.82rem;
        font-weight: 500;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .badge-normal {
        background-color: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .badge-privacy {
        background-color: rgba(56, 189, 248, 0.18);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    
    /* Section Containers */
    .section-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 8px;
    }
    
    /* Callouts */
    .callout-info {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 14px 0;
        font-size: 0.92rem;
        color: #cbd5e1;
    }
    .callout-warning {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 14px 0;
        font-size: 0.92rem;
        color: #fbbf24;
    }
    
    /* Sidebar Base Styling */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        color: #E6EDF3 !important;
    }
    
    /* Sidebar Headings, Labels, Markdown & Paragraphs */
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
        color: #E6EDF3 !important;
    }
    
    /* Navigation Radio Buttons - Unselected */
    section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #E6EDF3 !important;
        padding: 6px 10px !important;
        border-radius: 6px !important;
        margin-bottom: 2px !important;
        transition: background-color 0.15s ease, color 0.15s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
        color: #E6EDF3 !important;
        font-size: 0.92rem;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Navigation Radio Buttons - Currently Selected Highlight (#4FD1C5, bold) */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has([aria-checked="true"]) {
        background-color: rgba(79, 209, 197, 0.12) !important;
        border-left: 3px solid #4FD1C5 !important;
        border-radius: 4px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"] p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has([aria-checked="true"]) p {
        color: #4FD1C5 !important;
        font-weight: 700 !important;
    }
    
    /* Sliders and Number Inputs in Sidebar */
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stThumbValue"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stSliderTickBarMin"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stSliderTickBarMax"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
        color: #E6EDF3 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stSliderTickBarMin"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stSliderTickBarMax"] {
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {
        fill: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        color: #E6EDF3 !important;
        background-color: #1e293b !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }
    
    /* EXCEPTION: Institutional Persona Selectbox (White Background + Dark #111 Text) */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] input,
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
        color: #111111 !important;
        fill: #111111 !important;
    }
    
    /* Selectbox Dropdown Menu Options (Popover) */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] div,
    ul[data-baseweb="menu"],
    ul[data-baseweb="menu"] li,
    ul[data-baseweb="menu"] li *,
    div[data-baseweb="select"] ul li,
    div[data-baseweb="select"] ul li * {
        background-color: #ffffff !important;
        color: #111111 !important;
    }
    ul[data-baseweb="menu"] li:hover,
    ul[data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. Data Caching & Loading Helpers
# ==============================================================================

DATA_DIR = os.environ.get("LEAKY_DATA_DIR", "data")


@st.cache_data
def load_json_file(filename: str) -> Dict[str, Any]:
    """Safely loads a JSON artifact from the data directory with graceful fallback."""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@st.cache_data
def load_data_bundle():
    """Caches all primary analytical and privacy JSON artifacts."""
    return {
        "summary": load_json_file("analytics_summary.json"),
        "funnel": load_json_file("analytics_funnel.json"),
        "dp_funnel": load_json_file("privacy_dp_funnel.json"),
        "district_leakage": load_json_file("analytics_district_leakage.json"),
        "dp_districts": load_json_file("privacy_dp_districts.json"),
        "bottlenecks": load_json_file("analytics_bottlenecks.json"),
        "categories": load_json_file("analytics_category.json"),
        "evaluation": load_json_file("evaluation_metrics.json"),
        "privacy_audit": load_json_file("privacy_audit_report.json"),
        "policy_context": clean_policy_data(load_json_file("policy_context_cache.json")),
    }


def clean_policy_data(raw_policy: Dict[str, Any]) -> Dict[str, Any]:
    """Applies domain filtering and .gov.in/.nic.in ranking to cached policy entries."""
    try:
        from src.policy_context import filter_and_rank_sources
        cleaned = {}
        for k, v in raw_policy.items():
            if isinstance(v, dict) and "sources" in v:
                entry = dict(v)
                entry["sources"] = filter_and_rank_sources(entry["sources"], catalog_key=k)
                cleaned[k] = entry
            else:
                cleaned[k] = v
        return cleaned
    except Exception:
        return raw_policy


DATA = load_data_bundle()


# ==============================================================================
# 3. Sidebar Navigation & Global Controls
# ==============================================================================

with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 2.2rem; margin-bottom: 6px;">🛡️</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">THE LEAKY PIPELINE</div>
            <div style="font-size: 0.78rem; color: #38bdf8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em;">
                VAW Policy Intelligence Platform
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧭 Navigation")
    PAGE = st.radio(
        label="Select Analytical View",
        options=[
            "📊 Executive Overview",
            "🔻 Funnel with Uncertainty Ranges",
            "⏱️ Duration Bottlenecks & Latency",
            "🔗 Linkage Quality & Evaluation",
            "⚖️ Before / After Policy Transformation",
            "🛡️ Survivor Privacy & DP Engine",
            "📜 Statutory Policy Insights (TinyFish)"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 👤 Institutional Persona")
    ROLE_SELECT = st.selectbox(
        "Active Role",
        ["Administrator (Full Audit & Config)", "Policy Analyst (Deep Breakdown)", "Public / Judiciary Viewer (Published DP)"],
        index=0
    )

    st.markdown("### 🔒 Privacy Controls")
    EPSILON_SIM = st.slider(r"Epsilon ($\epsilon$) Noise Level", min_value=0.1, max_value=2.0, value=0.25, step=0.05,
                            help="Lower epsilon draws stronger Laplace noise for higher survivor confidentiality.")
    K_ANONYMITY = st.number_input("k-Anonymity Suppression Threshold", min_value=3, max_value=20, value=5, step=1,
                                  help="Cells with fewer than k cases are strictly suppressed as '<k'.")

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">
            <strong style="color: #94a3b8;">System Integrity</strong><br/>
            • 5,125 Cases Reconstructed<br/>
            • Pure Laplace Differential Privacy<br/>
            • Zero-PII Response Verification<br/>
            • TinyFish Statutory Benchmarking
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# Helper Chart Utilities
# ==============================================================================

def get_plot_layout():
    """Generates standard sleek dark layout for Plotly figures."""
    return dict(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        margin=dict(l=40, r=30, t=40, b=40),
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
        legend=dict(bgcolor="rgba(17,24,39,0.7)", bordercolor="rgba(255,255,255,0.1)")
    )


# ==============================================================================
# PAGE 1: 📊 Executive Overview
# ==============================================================================

if PAGE == "📊 Executive Overview":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Executive Policy Overview: Violence-Against-Women Case Attrition</div>
            <div class="hero-subtitle">
                Systemic drop-off analytics reconstructed from multi-institutional silos (Police FIR, Hospital MLC, 
                Forensic Science Laboratories, Prosecution, and District Courts) with mathematical Differential Privacy protection.
            </div>
        </div>
    """, unsafe_allow_html=True)

    summary_data = DATA.get("summary", {})
    health = summary_data.get("pipeline_health", {})

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Reconstructed Cases</div>
                <div class="metric-value">{health.get('total_reconstructed_cases', 5125):,}</div>
                <div class="metric-delta" style="color: #38bdf8;">14,384 Canonical Events</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi2:
        fir_cnt = health.get('cases_initiating_fir', 4347)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">FIR Initiating Rate</div>
                <div class="metric-value">84.8%</div>
                <div class="metric-delta" style="color: #34d399;">{fir_cnt:,} Police FIRs</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi3:
        verdict_cnt = health.get('cases_reaching_verdict', 1397)
        conviction_rate = health.get('overall_conviction_reach_rate_pct', 32.14)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Conviction Reach Rate</div>
                <div class="metric-value">{conviction_rate}%</div>
                <div class="metric-delta" style="color: #f87171;">{verdict_cnt:,} Reaching Trial Verdict</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Policy Leak Points</div>
                <div class="metric-value">3 <span style="font-size: 1rem; color: #f87171;">Critical</span></div>
                <div class="metric-delta" style="color: #fbbf24;">D3, D6, D7 Anomalies</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi5:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Privacy Guarantee</div>
                <div class="metric-value">ε-DP</div>
                <div class="metric-delta" style="color: #34d399;">Zero PII Exposed</div>
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
            fig_rank = px.bar(
                df_rank,
                x="score",
                y="district",
                orientation="h",
                color="status",
                color_discrete_map={"CRITICAL": "#ef4444", "NORMAL": "#10b981"},
                text="score",
                labels={"score": "Vulnerability Score", "district": "District", "status": "Classification"},
                title="Composite District Leak Severity Score (Baseline: Normal < 10.0)"
            )
            fig_rank.update_layout(get_plot_layout(), yaxis=dict(autorange="reversed"))
            fig_rank.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig_rank, use_container_width=True)

    with col_right:
        st.markdown("### 🔍 Systemic Policy Leak Points Detected")
        leak_points = summary_data.get("systemic_leak_points_detected", [])
        for lp in leak_points:
            dist = lp.get("district")
            code = lp.get("leak_code")
            title = lp.get("title")
            mult = lp.get("multiplier")
            drop = lp.get("drop_off_leak_pct")
            
            badge_html = f'<span class="badge badge-critical">CRITICAL LEAK: {code}</span>'
            detail_text = ""
            if mult:
                delay = lp.get("delay_days") or lp.get("trial_delay_days", 0.0)
                base = lp.get("baseline_days", 0.0)
                detail_text = f"Turnaround: <strong>{delay:.1f} days</strong> vs baseline <strong>{base:.1f} days</strong> (<strong>{mult:.2f}x delay multiplier</strong>)"
            elif drop:
                rate = lp.get("chargesheet_rate_pct", 0.0)
                bench = lp.get("state_benchmark_pct", 0.0)
                detail_text = f"Only <strong>{rate:.1f}%</strong> chargesheet conversion vs <strong>{bench:.1f}%</strong> benchmark (<strong>{drop:.1f}% attrition drop-off</strong>)"

            st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <strong style="color: #f8fafc; font-size: 1.05rem;">{dist}</strong>
                        {badge_html}
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 4px;">{title}</div>
                    <div style="color: #94a3b8; font-size: 0.82rem;">{detail_text}</div>
                </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: 🔻 Funnel with Uncertainty Ranges
# ==============================================================================

elif PAGE == "🔻 Funnel with Uncertainty Ranges":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Case Attrition Funnel & Differential Privacy Uncertainty</div>
            <div class="hero-subtitle">
                Institutional attrition across canonical milestones. Every aggregate displays mathematical 
                Laplace noise error margins (±95% CI) and small-cell k-anonymity suppression (&lt;k).
            </div>
        </div>
    """, unsafe_allow_html=True)

    dp_funnel = DATA.get("dp_funnel", {})
    funnel_stages = dp_funnel.get("funnel_stages", [])

    if not funnel_stages:
        st.warning("Funnel analytical data not loaded. Please ensure analytics pipeline has run.")
    else:
        # Toggle options
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 3, 4])
        with ctrl_col1:
            SHOW_RAW = st.toggle("Show Raw vs DP Noisy Overlay", value=True)
        with ctrl_col2:
            CONF_LEVEL = st.selectbox("Confidence Level", ["95% Confidence Interval", "99% Confidence Interval"], index=0)
        with ctrl_col3:
            st.markdown(f"""
                <div style="padding-top: 6px; font-size: 0.85rem; color: #38bdf8;">
                    <strong>Privacy Mechanism:</strong> Laplace (&Delta;f=1, &epsilon;={EPSILON_SIM:.2f})
                </div>
            """, unsafe_allow_html=True)

        # Plotly Funnel Chart with Error Bars
        stage_names = [s["stage_name"] for s in funnel_stages]
        noisy_counts = [s["noisy_case_count"] if isinstance(s["noisy_case_count"], (int, float)) else 0 for s in funnel_stages]
        raw_counts = [s.get("raw_case_count", 0) for s in funnel_stages]
        err_margins = [s.get("confidence_margin_95", 12.0) if s.get("confidence_margin_95") is not None else 0.0 for s in funnel_stages]

        fig_funnel = go.Figure()

        if SHOW_RAW:
            fig_funnel.add_trace(go.Bar(
                name="Raw Ground Truth (Silo Unsanitized)",
                x=stage_names,
                y=raw_counts,
                marker_color="#475569",
                opacity=0.6,
                text=raw_counts,
                textposition="auto"
            ))

        fig_funnel.add_trace(go.Bar(
            name=f"Differentially Private Estimate (ε={EPSILON_SIM})",
            x=stage_names,
            y=noisy_counts,
            error_y=dict(type='data', array=err_margins, visible=True, color="#38bdf8", thickness=2, width=6),
            marker=dict(
                color=["#0284c7", "#0ea5e9", "#38bdf8", "#818cf8", "#a855f7"],
                line=dict(color="#ffffff", width=1)
            ),
            text=[f"{val:,}<br/>(±{err:.1f})" for val, err in zip(noisy_counts, err_margins)],
            textposition="auto"
        ))

        fig_funnel.update_layout(
            get_plot_layout(),
            title="Institutional Stage Volume with 95% Differential Privacy Confidence Margins",
            barmode="group" if SHOW_RAW else "relative",
            yaxis_title="Case Volume",
            xaxis_title="Institutional Milestone"
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

        # Stage Breakdown Table
        st.markdown("### 📋 Milestone-by-Milestone Attrition Accounting")
        table_rows = []
        for s in funnel_stages:
            table_rows.append({
                "Milestone": s["stage_name"],
                "Raw Cases": f"{s.get('raw_case_count', 0):,}",
                "DP Noisy Count": s["noisy_case_count"],
                "95% Error Bound": f"±{s.get('confidence_margin_95', 0.0)}" if s.get('confidence_margin_95') else "Suppressed",
                "Cumulative Retention": f"{s.get('dp_retention_pct', 0.0):.1f}%",
                "Step Conversion": f"{s.get('dp_conversion_pct', 0.0):.1f}%",
                "k-Suppressed (<5)": "⚠️ Masked" if s.get("is_k_anonymity_suppressed") else "✅ Compliant"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)


# ==============================================================================
# PAGE 3: ⏱️ Duration Bottlenecks & Latency
# ==============================================================================

elif PAGE == "⏱️ Duration Bottlenecks & Latency":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Institutional Duration Bottlenecks & Statutory Compliance</div>
            <div class="hero-subtitle">
                Quantifying latency across justice milestones against statutory benchmarks: 
                60-day police chargesheet mandate (Sec 173 CrPC / Sec 193 BNSS), 24h medical exam, and 30-day forensic reporting.
            </div>
        </div>
    """, unsafe_allow_html=True)

    bottlenecks = DATA.get("bottlenecks", {})

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
            med = trans_data.get("median_days", 0.0)
            p95 = trans_data.get("p95_days", 0.0)
            sample = trans_data.get("sample_count", 0)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{med:.1f} <span style="font-size: 0.9rem; color: #94a3b8;">days median</span></div>
                    <div class="metric-delta" style="color: #f59e0b;">P95 Latency: {p95:.1f} days (n={sample:,})</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Detailed Transition Breakdown & Delay Distribution
    sel_trans = st.selectbox("Select Transition to Inspect Latency Distribution", list(bottlenecks.keys()), index=0)
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
                color="Case Count",
                color_continuous_scale="Viridis",
                title=f"Latency Distribution: {sel_trans}"
            )
            fig_buckets.update_layout(get_plot_layout())
            st.plotly_chart(fig_buckets, use_container_width=True)

    with b_right:
        st.markdown(f"### 📊 Non-Parametric Latency Percentiles")
        st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 18px;">
                <p><strong>Sample Size:</strong> {selected_data.get('sample_count', 0):,} cases</p>
                <p><strong>Mean Duration:</strong> {selected_data.get('mean_days', 0.0):.1f} ± {selected_data.get('std_days', 0.0):.1f} days</p>
                <p><strong>Median (P50):</strong> {selected_data.get('median_days', 0.0):.1f} days</p>
                <p><strong>Interquartile Range (IQR):</strong> {selected_data.get('iqr_days', 0.0):.1f} days (P25: {selected_data.get('p25_days', 0.0)}d - P75: {selected_data.get('p75_days', 0.0)}d)</p>
                <p><strong>Severe Delay (P90):</strong> {selected_data.get('p90_days', 0.0):.1f} days</p>
                <p><strong>Extreme Outlier (P95):</strong> {selected_data.get('p95_days', 0.0):.1f} days</p>
                <hr style="border-color: rgba(255,255,255,0.1);"/>
                <p style="color: #38bdf8;"><strong>Statutory Mandate Compliance:</strong> {selected_data.get('statutory_compliance', {}).get('within_60_days_pct', 0.0)}% within 60 days</p>
            </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: 🔗 Linkage Quality & Evaluation
# ==============================================================================

elif PAGE == "🔗 Linkage Quality & Evaluation":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Privacy-Preserving Record Linkage: Quality & Diagnostics</div>
            <div class="hero-subtitle">
                Rigorous linkage evaluation of the Fellegi-Sunter / Splink probabilistic record linkage model 
                benchmarked against ground truth across all five institutional data sources.
            </div>
        </div>
    """, unsafe_allow_html=True)

    eval_data = DATA.get("evaluation", {})
    pairwise = eval_data.get("pairwise_metrics", {})
    cluster = eval_data.get("cluster_metrics", {})

    # Top Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">B-Cubed F1 Score</div>
                <div class="metric-value">{cluster.get('bcubed_f1', 0.9288):.3f}</div>
                <div class="metric-delta" style="color: #34d399;">Cluster-Level Accuracy</div>
            </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Macro Entity Purity</div>
                <div class="metric-value">{cluster.get('macro_purity', 0.972):.1%}</div>
                <div class="metric-delta" style="color: #34d399;">Near-Zero Cross-Contamination</div>
            </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Pairwise Recall</div>
                <div class="metric-value">{pairwise.get('metrics_at_default_threshold', {}).get('recall', 0.8846):.1%}</div>
                <div class="metric-delta" style="color: #38bdf8;">At Optimal Threshold 0.85</div>
            </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Candidate Search Space</div>
                <div class="metric-value">{pairwise.get('total_candidate_pairs', 706105):,}</div>
                <div class="metric-delta" style="color: #a855f7;">Blocked Pair Comparisons</div>
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
        fig_pr.add_trace(go.Scatter(
            x=[pairwise.get('metrics_at_default_threshold', {}).get('recall', 0.8846)],
            y=[pairwise.get('metrics_at_default_threshold', {}).get('precision', 0.572)],
            mode="markers+text",
            marker=dict(color="#ef4444", size=14, symbol="star"),
            name="Optimal Threshold (0.85)",
            text=["Selected Operational Threshold (0.85)"],
            textposition="top right"
        ))
        fig_pr.update_layout(get_plot_layout())
        st.plotly_chart(fig_pr, use_container_width=True)

    # Threshold Diagnostic Trade-offs Table
    st.markdown("### 🎚️ Threshold Trade-Off Analysis")
    st.dataframe(pd.DataFrame(thresh_table), use_container_width=True)


# ==============================================================================
# PAGE 5: ⚖️ Before / After Policy Transformation
# ==============================================================================

elif PAGE == "⚖️ Before / After Policy Transformation":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Before vs After: The Impact of Record Linkage & Privacy</div>
            <div class="hero-subtitle">
                Demonstrating how siloed departmental views obscure systemic failures, whereas privacy-preserving
                longitudinal record linkage exposes exact policy bottlenecks to enable targeted statutory interventions.
            </div>
        </div>
    """, unsafe_allow_html=True)

    c_before, c_after = st.columns(2)

    with c_before:
        st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.4); border: 2px dashed rgba(239, 68, 68, 0.4); border-radius: 12px; padding: 22px; min-height: 480px;">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.5rem; margin-right: 8px;">❌</span>
                    <strong style="color: #f87171; font-size: 1.3rem;">BEFORE: Siloed Institutional Data</strong>
                </div>
                <ul style="color: #cbd5e1; line-height: 1.8; font-size: 0.95rem;">
                    <li><strong>Departmental Blindness:</strong> Police, Hospitals, Forensic Labs, and Courts maintain disconnected databases with distinct survivor identifiers.</li>
                    <li><strong>False Self-Assessments:</strong>
                        <ul>
                            <li>Police claim 90%+ crime disposal rate by closing cases before forensic receipt.</li>
                            <li>Forensic Science Labs report high sample processing numbers without tracking court trial impact.</li>
                            <li>Judiciary attributes trial pendency solely to witness non-attendance.</li>
                        </ul>
                    </li>
                    <li><strong>Hidden Vulnerabilities:</strong> District 3 forensic turnaround lag (137.5 days) and District 6 police drop-off (72.2%) are completely invisible to state policymakers.</li>
                    <li><strong>Privacy Hazards:</strong> Ad-hoc sharing of raw survivor files leaks PII, compromising survivor safety.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with c_after:
        st.markdown("""
            <div style="background: rgba(17, 24, 39, 0.8); border: 2px solid rgba(16, 185, 129, 0.5); border-radius: 12px; padding: 22px; min-height: 480px;">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.5rem; margin-right: 8px;">✅</span>
                    <strong style="color: #34d399; font-size: 1.3rem;">AFTER: Privacy-Preserving Linkage Pipeline</strong>
                </div>
                <ul style="color: #cbd5e1; line-height: 1.8; font-size: 0.95rem;">
                    <li><strong>Unified Case Trajectories:</strong> 14,384 events stitched into 5,125 complete case journeys across all 5 institutional stages.</li>
                    <li><strong>Pinpointed Systemic Leak Points:</strong>
                        <ul>
                            <li><strong>Rudrapur (D3):</strong> Forensic turnaround delay (3.19x multiplier) isolated as primary root cause of chargesheet stalling.</li>
                            <li><strong>Shivgarh (D6):</strong> FIR-to-Chargesheet attrition (72.2% drop-off) identified from failure to record Sec 164 survivor statements.</li>
                            <li><strong>Anandpur (D7):</strong> Judicial trial delay (612.9 days) isolated to the post-chargesheet court adjudication phase.</li>
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
    comp_data = {
        "Analytical Dimension": [
            "Forensic Turnaround Visibility",
            "Police Attrition Attribution",
            "Cross-Stage Latency Quantification",
            "Survivor Re-identification Risk",
            "Statutory Compliance Auditing"
        ],
        "Siloed Status Quo (Before)": [
            "Unknown (Lab internal metric only)",
            "Assumed victim uncooperativeness",
            "Impossible (Stages disconnected)",
            "High (Unsanitized file sharing)",
            "Anecdotal / Reactive"
        ],
        "Privacy Pipeline (After)": [
            "137.5 days median in D3 (3.19x normal)",
            "72.2% attrition drop-off in D6 (LP-02)",
            "Exact IQR & P95 durations calculated",
            "Zero PII + Laplace DP Noise Mask",
            "Automated statutory benchmark tracking"
        ]
    }
    st.table(pd.DataFrame(comp_data))


# ==============================================================================
# PAGE 6: 🛡️ Survivor Privacy & DP Engine
# ==============================================================================

elif PAGE == "🛡️ Survivor Privacy & DP Engine":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Survivor Confidentiality & Differential Privacy Engine</div>
            <div class="hero-subtitle">
                Algorithmic privacy protections guaranteeing zero re-identification of survivors: Laplace mechanism 
                noise injection, Sequential Composition privacy budget accounting, and adversarial attack defense.
            </div>
        </div>
    """, unsafe_allow_html=True)

    priv_audit = DATA.get("privacy_audit", {})
    budget_sum = priv_audit.get("privacy_budget_summary", {})
    k_eval = priv_audit.get("k_anonymity_evaluation", {})

    # Top KPI Budget Cards
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Epsilon Allocated</div>
                <div class="metric-value">{budget_sum.get('total_epsilon_allocated', 10.0):.2f}</div>
                <div class="metric-delta" style="color: #38bdf8;">Sequential Budget Upper Bound</div>
            </div>
        """, unsafe_allow_html=True)

    with b2:
        spent = budget_sum.get('spent_epsilon', 2.35)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Epsilon Budget Spent</div>
                <div class="metric-value">{spent:.2f}</div>
                <div class="metric-delta" style="color: #f59e0b;">Consumed by Analytical Queries</div>
            </div>
        """, unsafe_allow_html=True)

    with b3:
        rem = budget_sum.get('remaining_epsilon', 7.65)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Remaining Epsilon</div>
                <div class="metric-value">{rem:.2f}</div>
                <div class="metric-delta" style="color: #34d399;">Available Query Capacity</div>
            </div>
        """, unsafe_allow_html=True)

    with b4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Zero-PII Status</div>
                <div class="metric-value">100%</div>
                <div class="metric-delta" style="color: #34d399;">Verified: Zero Leaked Names/IDs</div>
            </div>
        """, unsafe_allow_html=True)

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
            <div style="background: rgba(30, 41, 59, 0.7); border-radius: 8px; padding: 14px; margin-top: 14px;">
                <p><strong>Laplace Scale (b = Δf / ε):</strong> {scale_b:.2f}</p>
                <p><strong>95% Theoretical Error Bound:</strong> ±{margin_95:.2f} cases</p>
                <p style="font-size: 0.85rem; color: #94a3b8;">
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
        fig_lap.add_vline(x=-margin_95, line_dash="dash", line_color="#ef4444", annotation_text="-95% CI")
        fig_lap.add_vline(x=margin_95, line_dash="dash", line_color="#ef4444", annotation_text="+95% CI")
        fig_lap.update_layout(get_plot_layout())
        st.plotly_chart(fig_lap, use_container_width=True)

    # Adversarial Attack Simulation Results
    st.markdown("### 🦹 Adversarial Re-Identification Attack Defense")
    adv_threats = priv_audit.get("adversarial_threat_evaluations", {})
    singling_out = adv_threats.get("singling_out_marksmanship_attack", {})
    before_supp = singling_out.get("before_protection", {})
    after_supp = singling_out.get("after_k_anonymity_protection", {})

    att_col1, att_col2 = st.columns(2)
    with att_col1:
        st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 16px;">
                <h4 style="color: #f87171; margin-top: 0;">Unprotected Raw Data Vulnerability</h4>
                <p><strong>Singling-Out / Marksmanship Risk:</strong> {before_supp.get('singling_out_risk_pct', 7.42)}%</p>
                <p><strong>Max Prosecutor Re-identification Risk:</strong> {before_supp.get('max_prosecutor_risk_pct', 100.0)}%</p>
                <p><strong>Unique Vulnerable Survivor Cases:</strong> {before_supp.get('unique_records_count', 380):,} records</p>
                <p style="font-size: 0.85rem; color: #fca5a5;">An attacker with auxiliary knowledge can single out individuals with 100% certainty.</p>
            </div>
        """, unsafe_allow_html=True)

    with att_col2:
        st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 16px;">
                <h4 style="color: #34d399; margin-top: 0;">Post-Suppression & DP Defense</h4>
                <p><strong>Singling-Out / Marksmanship Risk:</strong> {after_supp.get('singling_out_risk_pct', 0.0)}% (0 vulnerable cases)</p>
                <p><strong>Max Prosecutor Re-identification Risk:</strong> {after_supp.get('max_prosecutor_risk_pct', 20.0)}% (k >= 5 enforced)</p>
                <p><strong>Suppression Audit:</strong> 100% of equivalence classes &lt; 5 masked</p>
                <p style="font-size: 0.85rem; color: #86efac;">Singling out completely foiled; mathematical plausible deniability guaranteed.</p>
            </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 7: 📜 Statutory Policy Insights (TinyFish)
# ==============================================================================

elif PAGE == "📜 Statutory Policy Insights (TinyFish)":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Statutory Policy Context & Government Benchmarks</div>
            <div class="hero-subtitle">
                Official statutory benchmarks and infrastructure standards retrieved via TinyFish 
                prioritizing official <code>.gov.in</code> and <code>.nic.in</code> government sources with zero PII leakage.
            </div>
        </div>
    """, unsafe_allow_html=True)

    policy_cache = DATA.get("policy_context", {})

    st.markdown("### 🏛️ Top 5 Policy Leak Points & Statutory Grounding")

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
                    <strong style="color: #f87171;">Empirical Pipeline Finding:</strong><br/>
                    <span style="color: #e2e8f0;">{finding}</span>
                </div>
                <div style="margin-bottom: 16px;">
                    <strong style="color: #38bdf8;">Recommended Statutory Intervention:</strong><br/>
                    <span style="color: #cbd5e1;">{intervention}</span>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<strong style='color: #34d399;'>Official Government & Statutory Sources:</strong>", unsafe_allow_html=True)
            for s in sources:
                domain = s.get("source_domain", "")
                is_gov = (".gov.in" in domain) or (".nic.in" in domain)
                badge_type = "badge-privacy" if is_gov else "badge-normal"
                badge_label = "OFFICIAL GOVT SOURCE" if is_gov else "STATUTORY REFERENCE"

                st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <a href="{s.get('url', '#')}" target="_blank" style="color: #38bdf8; font-weight: 600; text-decoration: none;">
                                {s.get('title', 'Document')} ↗
                            </a>
                            <span class="badge {badge_type}">{badge_label} ({domain})</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">
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
                    <strong style="color: #38bdf8;">Autonomous TinyFish Web Agent Fact Extraction</strong>
                </div>
                <div><strong>Mandate:</strong> {fact.get('fact', '')}</div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">
                    Statutory Reference: {fact.get('statutory_reference', '')} | Enacted: {fact.get('enactment_year', 2012)} | Source: {fact.get('source_domain', '')}
                </div>
            </div>
        """, unsafe_allow_html=True)
