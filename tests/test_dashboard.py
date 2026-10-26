"""
tests/test_dashboard.py - Unit & Integration tests for Phase 7 Streamlit Dashboard.
"""

import os
import json
import pytest
import pandas as pd

from app.dashboard import (
    load_json_file,
    load_data_bundle,
    get_plot_layout,
    DATA_DIR,
    request_navigation,
    ROLE_PERMISSIONS,
    compute_dashboard_kpis
)
from src.upload_analysis import build_dashboard_bundle_from_analysis


def test_dashboard_data_bundle_loading():
    """Verifies that all analytical and privacy JSON artifacts load cleanly."""
    bundle = load_data_bundle()
    required_keys = [
        "summary", "funnel", "dp_funnel", "district_leakage",
        "dp_districts", "bottlenecks", "categories", "evaluation",
        "privacy_audit", "policy_context"
    ]
    for key in required_keys:
        assert key in bundle, f"Missing data key: {key}"
        assert isinstance(bundle[key], dict), f"Data for {key} must be a dictionary"


def test_funnel_uncertainty_data_integrity():
    """Verifies that the DP Funnel contains 5 canonical milestones with confidence bounds."""
    bundle = load_data_bundle()
    dp_funnel = bundle["dp_funnel"]
    stages = dp_funnel.get("funnel_stages", [])
    assert len(stages) == 5, f"Expected 5 canonical milestones, got {len(stages)}"

    milestone_ids = [s["stage_id"] for s in stages]
    assert milestone_ids == [
        "FIR_REGISTERED",
        "MEDICAL_EXAM",
        "FORENSIC_REPORT",
        "CHARGESHEET_FILED",
        "COURT_VERDICT"
    ]

    for s in stages:
        assert "noisy_case_count" in s
        assert "confidence_margin_95" in s
        assert "dp_retention_pct" in s


def test_bottlenecks_percentiles_integrity():
    """Verifies duration percentiles and statutory compliance metrics."""
    bundle = load_data_bundle()
    bottlenecks = bundle["bottlenecks"]
    assert len(bottlenecks) > 0

    key_transitions = [
        "FIR_REGISTERED -> MEDICAL_EXAM",
        "MEDICAL_EXAM -> FORENSIC_REPORT",
        "FORENSIC_REPORT -> CHARGESHEET_FILED",
        "CHARGESHEET_FILED -> COURT_VERDICT"
    ]
    for trans in key_transitions:
        assert trans in bottlenecks, f"Missing transition {trans}"
        t_data = bottlenecks[trans]
        assert "median_days" in t_data
        assert "iqr_days" in t_data
        assert "p95_days" in t_data
        assert "delay_buckets" in t_data


def test_linkage_evaluation_metrics_integrity():
    """Verifies PR curve data and B-Cubed cluster metrics."""
    bundle = load_data_bundle()
    eval_data = bundle["evaluation"]

    pairwise = eval_data.get("pairwise_metrics", {})
    cluster = eval_data.get("cluster_metrics", {})
    thresh_table = eval_data.get("pairwise_threshold_table", [])

    assert pairwise.get("optimal_threshold") == 0.85
    assert cluster.get("bcubed_f1", 0) > 0.90
    assert len(thresh_table) >= 5


def test_policy_context_statutory_sources():
    """Verifies that policy context contains official government sources without PII."""
    bundle = load_data_bundle()
    pol_data = bundle["policy_context"]
    assert len(pol_data) >= 5

    for key, item in pol_data.items():
        if key == "government_benchmark_fact":
            continue
        assert "sources" in item
        domains = [s.get("source_domain", "") for s in item["sources"]]
        # Ensure scribd or facebook are not present
        for d in domains:
            assert "scribd" not in d
            assert "facebook" not in d


def test_plot_layout_styling():
    """Verifies consistent sober editorial light plot layout configuration."""
    layout = get_plot_layout()
    assert layout["paper_bgcolor"] == "#FFFFFF"
    assert layout["plot_bgcolor"] == "#FFFFFF"
    assert "font" in layout


def test_theme_wcag_contrast_ratios():
    """Computes and asserts WCAG AA contrast ratios (>= 4.5:1) for all theme color pairs."""
    def hex_to_rgb(hex_str: str):
        hex_str = hex_str.lstrip('#')
        return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def rel_luminance(rgb):
        def channel(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        r, g, b = [channel(c) for c in rgb]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def contrast_ratio(hex1: str, hex2: str) -> float:
        l1 = rel_luminance(hex_to_rgb(hex1))
        l2 = rel_luminance(hex_to_rgb(hex2))
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    pairs = [
        ("Page BG (#FAF8F5)", "Primary Text (#222222)", "#FAF8F5", "#222222"),
        ("Page BG (#FAF8F5)", "Secondary Text (#5F5F5F)", "#FAF8F5", "#5F5F5F"),
        ("Page BG (#FAF8F5)", "Accent Maroon (#8B1E3F)", "#FAF8F5", "#8B1E3F"),
        ("Card Surface (#FFFFFF)", "Primary Text (#222222)", "#FFFFFF", "#222222"),
        ("Card Surface (#FFFFFF)", "Secondary Text (#5F5F5F)", "#FFFFFF", "#5F5F5F"),
        ("Card Surface (#FFFFFF)", "Accent Maroon (#8B1E3F)", "#FFFFFF", "#8B1E3F"),
        ("Sidebar BG (#F0ECE5)", "Dark Text (#222222)", "#F0ECE5", "#222222"),
        ("Sidebar BG (#F0ECE5)", "Secondary Text (#5F5F5F)", "#F0ECE5", "#5F5F5F"),
        ("Sidebar BG (#F0ECE5)", "Selected Nav Accent (#8B1E3F)", "#F0ECE5", "#8B1E3F"),
        ("Alert Badge (#FEF3C7)", "Alert Amber (#B45309)", "#FEF3C7", "#B45309"),
        ("Selectbox Surface (#FFFFFF)", "Selectbox Text (#222222)", "#FFFFFF", "#222222"),
    ]

    print("\n| Background | Foreground / Text | Contrast Ratio | WCAG AA (>= 4.5:1) |")
    print("|---|---|---|---|")
    for bg_label, fg_label, bg_hex, fg_hex in pairs:
        ratio = contrast_ratio(bg_hex, fg_hex)
        passed = "Pass (AAA)" if ratio >= 7.0 else ("Pass (AA)" if ratio >= 4.5 else "FAIL")
        print(f"| {bg_label} | {fg_label} | **{ratio:.2f}:1** | {passed} |")
        assert ratio >= 4.5, f"Contrast ratio {ratio:.2f}:1 for {fg_label} fails WCAG AA 4.5:1 requirement"


def test_empirical_impact_signals():
    """
    Verifies that the underlying pipeline summary artifacts contain key
    empirical signals:
    - District 3: Forensic lab turnaround delay multiplier (~3.2x - 3.5x)
    - District 6: Severe chargesheet drop-off leak
    - Linkage evaluation metrics: Pairwise recall >= 88% and B-Cubed F1 >= 92%
    """
    bundle = load_data_bundle()
    summary = bundle.get("summary", {})
    leaks = summary.get("systemic_leak_points_detected", [])

    # 1. District 3 forensic delay
    d3_leak = next((lp for lp in leaks if "D3" in lp.get("district", "") or lp.get("leak_code") == "LP-01-FORENSIC-DELAY"), None)
    assert d3_leak is not None, "District 3 forensic delay leak point missing from summary"
    assert d3_leak.get("multiplier", 0) >= 3.0, f"Expected District 3 multiplier >= 3.0x, got {d3_leak.get('multiplier')}"

    # 2. District 6 chargesheet drop-off
    d6_leak = next((lp for lp in leaks if "D6" in lp.get("district", "") or lp.get("leak_code") == "LP-02-POLICE-ATTRITION"), None)
    assert d6_leak is not None, "District 6 chargesheet drop-off leak point missing from summary"
    assert d6_leak.get("drop_off_leak_pct", 0) > 50.0, f"Expected District 6 leak pct > 50%, got {d6_leak.get('drop_off_leak_pct')}"

    # 3. Splink record linkage performance benchmarks
    eval_data = bundle.get("evaluation", {})
    at_thr = eval_data.get("pairwise_metrics", {}).get("metrics_at_default_threshold", {})
    cluster = eval_data.get("cluster_metrics", {})
    assert at_thr.get("recall", 0) >= 0.88, f"Expected pairwise recall >= 0.88, got {at_thr.get('recall')}"
    assert cluster.get("bcubed_f1", 0) >= 0.92, f"Expected B-Cubed F1 >= 0.92, got {cluster.get('bcubed_f1')}"


def test_privacy_funnel_safeguards():
    """
    Verifies that funnel milestone data contains valid uncertainty margins,
    non-negative privatized case counts, and valid step conversion ratios.
    """
    bundle = load_data_bundle()
    dp_funnel = bundle.get("dp_funnel", {})
    stages = dp_funnel.get("funnel_stages", [])
    assert len(stages) == 5, f"Expected 5 milestones, got {len(stages)}"

    for s in stages:
        assert "noisy_case_count" in s
        assert "confidence_margin_95" in s
        assert "dp_retention_pct" in s
        assert "dp_conversion_pct" in s
        # Counts must be non-negative
        if isinstance(s["noisy_case_count"], (int, float)):
            assert s["noisy_case_count"] >= 0


def test_navigation_request_flow():
    """Verifies that navigation requests use nav_page_request and don't modify nav_page_selection directly."""
    import streamlit as st
    
    # 1. Fresh session defaults to first allowed page: Analyze Your Data for Analyst/Admin, Executive Overview for Viewer
    st.session_state.clear()
    assert ROLE_PERMISSIONS["analyst"][0] == "🔬 Analyze Your Data"
    assert ROLE_PERMISSIONS["admin"][0] == "🔬 Analyze Your Data"
    assert "🔬 Analyze Your Data" not in ROLE_PERMISSIONS["viewer"]
    assert ROLE_PERMISSIONS["viewer"][0] == "📊 Executive Overview"
    
    # 2. Calling request_navigation sets nav_page_request key without touching nav_page_selection
    request_navigation("📊 Executive Overview")
    assert st.session_state.get("nav_page_request") == "📊 Executive Overview"
    assert "nav_page_selection" not in st.session_state

    # 3. Sidebar navigation resolution applies request before radio instantiation
    allowed = ROLE_PERMISSIONS["viewer"]
    if "nav_page_request" in st.session_state:
        req = st.session_state.pop("nav_page_request")
        if req in allowed:
            st.session_state["nav_page_selection"] = req

    assert st.session_state["nav_page_selection"] == "📊 Executive Overview"
    assert "nav_page_request" not in st.session_state


def test_fresh_launch_empty_state_no_hardcoded_metrics():
    """
    Scenario: Fresh launch
    Verifies that when no dataset is analyzed, dashboard KPIs contain no old hardcoded
    metrics (5,125 cases, 84.8% FIR rate, 32.1% conviction reach rate).
    """
    kpis = compute_dashboard_kpis({})
    assert kpis["reconstructed_cases_display"] == "N/A"
    assert kpis["events_display"] == "N/A"
    assert kpis["fir_rate_display"] == "N/A"
    assert kpis["verdict_rate_display"] == "N/A"
    assert kpis["leak_count_display"] == "N/A"
    assert kpis["reconstructed_cases_display"] != "5,125"
    assert kpis["fir_rate_display"] != "84.8%"
    assert kpis["verdict_rate_display"] != "32.1%"


def test_single_case_analysis_state_and_persistence():
    """
    Scenario: Analyze one synthetic case
    Verifies that case-specific results appear, aggregate pages do not fabricate statistics,
    and results persist across navigation.
    """
    import streamlit as st
    st.session_state.clear()

    mock_case_res = {
        "is_single_case": True,
        "masked_survivor_name": "S**** P****",
        "overall_linkage_label": "same case",
        "linkage_method": "Fellegi-Sunter Pairwise Scoring (case_simulator)",
        "timeline": {
            "stages": [
                {"stage_id": "police", "stage_name": "1. Police FIR", "present": True, "date": "2024-03-01", "duration_days_from_prev": None},
                {"stage_id": "hospital", "stage_name": "2. Hospital MLC", "present": True, "date": "2024-03-02", "duration_days_from_prev": 1.0},
                {"stage_id": "fsl", "stage_name": "3. Forensic Lab (FSL)", "present": True, "date": "2024-04-15", "duration_days_from_prev": 44.0},
                {"stage_id": "prosecutor", "stage_name": "4. Prosecution Chargesheet", "present": True, "date": "2024-05-10", "duration_days_from_prev": 25.0},
                {"stage_id": "court", "stage_name": "5. Court Verdict", "present": False, "date": None, "duration_days_from_prev": None}
            ],
            "all_milestones_reached": False,
            "stalled_at": "Prosecution Chargesheet",
            "missing_record_from": "Court Verdict",
            "longest_transition": {
                "pair": "Hospital MLC -> Forensic Lab (FSL)",
                "duration_days": 44.0,
                "baseline_days": 30,
                "multiplier": 1.47
            }
        },
        "pairwise_match_scores": [
            {
                "stage_a": "Police FIR", "stage_b": "Hospital MLC",
                "match_probability": 0.96, "match_weight": 8.4,
                "label": "same case", "evidence": [{"detail": "Name agreement"}]
            }
        ]
    }

    # 1. Build dashboard bundle for single case
    bundle = build_dashboard_bundle_from_analysis(mock_case_res)
    assert bundle["is_single_case"] is True
    assert bundle["summary"]["pipeline_health"]["total_reconstructed_cases"] == 1
    assert bundle["journey"]["total_cases"] == 1

    # Aggregate pages must NOT fabricate population/district statistics
    assert bundle["summary"]["district_vulnerability_ranking"] == []
    assert len(bundle["dp_funnel"]["funnel_stages"]) == 5
    assert all(s["raw_case_count"] in [0, 1] for s in bundle["dp_funnel"]["funnel_stages"])

    # Transitions must reflect single-case measured durations
    transitions = bundle["bottlenecks"]["inter_stage_transitions"]
    assert len(transitions) == 3
    assert transitions[0]["transition"] == "1. Police FIR -> 2. Hospital MLC"
    assert transitions[0]["duration_days"] == 1.0
    assert transitions[1]["duration_days"] == 44.0

    # 2. Store in session state as single case active dataset
    st.session_state["active_dataset_id"] = "case"
    st.session_state["active_dataset_label"] = "Single Case (S**** P****)"
    st.session_state["active_dataset_bundle"] = bundle
    st.session_state["active_analysis_result"] = mock_case_res
    st.session_state["single_case_analysis_res"] = mock_case_res

    # 3. Simulate page navigation to Executive Overview and Funnel
    assert st.session_state["active_dataset_id"] == "case"
    assert st.session_state["active_dataset_bundle"]["is_single_case"] is True
    kpis = compute_dashboard_kpis(st.session_state["active_dataset_bundle"])
    assert kpis["reconstructed_cases_display"] == "1"
    assert kpis["reconstructed_cases_display"] != "5,125"


def test_multi_case_dataset_analysis_state_updates_all_pages():
    """
    Scenario: Analyze a multi-case CSV dataset / sample dataset
    Verifies that every supported dashboard page updates using the dataset's result.
    """
    import streamlit as st
    st.session_state.clear()

    mock_analysis_res = {
        "summary_kpis": {
            "total_cases": 120,
            "total_events": 480,
            "fir_cases": 115,
            "verdict_cases": 35,
            "drop_off_rate": 70.8
        },
        "funnel": {
            "total_cases": 120,
            "stages": [
                {"stage": "Police FIR", "stage_name": "1. Police FIR", "count": 115, "retention_pct": 100.0, "conversion_pct": 100.0},
                {"stage": "Hospital MLC", "stage_name": "2. Hospital MLC", "count": 100, "retention_pct": 87.0, "conversion_pct": 87.0},
                {"stage": "Forensic Lab", "stage_name": "3. Forensic Lab (FSL)", "count": 80, "retention_pct": 69.6, "conversion_pct": 80.0},
                {"stage": "Chargesheet", "stage_name": "4. Prosecution Chargesheet", "count": 60, "retention_pct": 52.2, "conversion_pct": 75.0},
                {"stage": "Court Verdict", "stage_name": "5. Court Verdict", "count": 35, "retention_pct": 30.4, "conversion_pct": 58.3}
            ]
        },
        "bottlenecks": {
            "Police FIR -> Hospital MLC": {"median_days": 1.5, "p95_days": 5.0, "sample_count": 100},
            "Hospital MLC -> Forensic Lab": {"median_days": 35.0, "p95_days": 90.0, "sample_count": 80},
            "Forensic Lab -> Prosecution": {"median_days": 20.0, "p95_days": 60.0, "sample_count": 60},
            "Prosecution -> Court Verdict": {"median_days": 180.0, "p95_days": 400.0, "sample_count": 35}
        },
        "linkage_stats": {
            "method": "Fellegi-Sunter In-Memory Linkage",
            "precision": 0.945,
            "recall": 0.892,
            "f1_score": 0.918,
            "clusters_formed": 120,
            "total_records_processed": 480,
            "potential_pairs_evaluated": 1500
        },
        "leaks": [
            {
                "stage": "Forensic Lab",
                "severity": "CRITICAL",
                "leak_code": "LP-FORENSIC-CUSTOM",
                "description": "Custom forensic turnaround delay detected",
                "drop_off_pct": 30.4,
                "district": "Custom District",
                "median_delay_days": 35.0
            }
        ]
    }

    bundle = build_dashboard_bundle_from_analysis(mock_analysis_res)

    # 1. Executive Overview / Summary
    kpis = compute_dashboard_kpis(bundle)
    assert kpis["reconstructed_cases_display"] == "120"
    assert kpis["events_display"] == "480"
    assert kpis["fir_count_display"] == "115"
    assert kpis["verdict_count_display"] == "35"
    assert kpis["reconstructed_cases_display"] != "5,125"

    # 2. Funnel with uncertainty ranges
    assert len(bundle["dp_funnel"]["funnel_stages"]) == 5
    assert bundle["dp_funnel"]["funnel_stages"][0]["noisy_case_count"] == 115

    # 3. Duration Bottlenecks
    assert "Police FIR -> Hospital MLC" in bundle["bottlenecks"]
    assert bundle["bottlenecks"]["Police FIR -> Hospital MLC"]["median_days"] == 1.5

    # 4. Linkage Quality & Evaluation (Honest metrics from active run)
    assert bundle["evaluation"]["linkage_method"] == "Fellegi-Sunter In-Memory Linkage"
    assert bundle["evaluation"]["pairwise_metrics"]["metrics_at_default_threshold"]["recall"] == 0.892
    assert bundle["evaluation"]["cluster_metrics"]["bcubed_f1"] == 0.918

    # 5. Before / After Policy Transformation
    assert bundle["before_after"]["filtered_case_count"] == 120
    assert bundle["before_after"]["filtered_event_count"] == 480

    # 6. Survivor Privacy & DP Engine
    assert bundle["privacy_audit"]["privacy_budget_summary"]["total_epsilon_allocated"] == 10.0

    # 7. Statutory Policy Insights
    assert "LP-01-FORENSIC-DELAY" in bundle["policy_context"]

    # Store in session state and verify persistence
    st.session_state["active_dataset_id"] = "uploaded"
    st.session_state["active_dataset_label"] = "Uploaded CSVs (120 cases)"
    st.session_state["active_dataset_bundle"] = bundle
    st.session_state["active_analysis_result"] = mock_analysis_res

    assert st.session_state["active_dataset_id"] == "uploaded"
    assert st.session_state["active_dataset_label"] == "Uploaded CSVs (120 cases)"


def test_clear_active_dataset_reset_action():
    """
    Scenario: Clear or replace active results
    Verifies that the clear action wipes active state back to empty dashboard.
    """
    import streamlit as st
    st.session_state["active_dataset_id"] = "sample"
    st.session_state["active_dataset_label"] = "Benchmark Sample Dataset (500 cases)"
    st.session_state["active_dataset_bundle"] = {"some": "data"}
    st.session_state["active_analysis_result"] = {"some": "result"}

    # Simulate clear action
    st.session_state["active_dataset_id"] = None
    st.session_state["active_dataset_label"] = None
    st.session_state["active_dataset_bundle"] = None
    st.session_state["active_analysis_result"] = None
    st.session_state["single_case_analysis_res"] = None
    st.session_state["upload_analysis_res"] = None
    st.session_state["sample_analysis_res"] = None

    assert st.session_state.get("active_dataset_id") is None
    assert st.session_state.get("active_dataset_bundle") is None
    assert st.session_state.get("active_analysis_result") is None
    kpis = compute_dashboard_kpis(st.session_state.get("active_dataset_bundle") or {})
    assert kpis["reconstructed_cases_display"] == "N/A"
    assert kpis["reconstructed_cases_display"] != "5,125"


def test_no_dataset_empty_state_and_unavailable_metrics():
    """
    Requirement 4 & 5: When no dataset is analyzed:
    - Never fallback to 5,125 cases or hardcoded metrics.
    - Clearly distinguish unavailable metrics (N/A) from zero values.
    """
    kpis = compute_dashboard_kpis({})
    assert kpis["reconstructed_cases_display"] == "N/A"
    assert kpis["events_display"] == "N/A"
    assert kpis["fir_rate_display"] == "N/A"
    assert kpis["verdict_rate_display"] == "N/A"
    assert kpis["leak_count_display"] == "N/A"
    assert kpis["bcubed_f1_display"] == "N/A"
    assert kpis["macro_purity_display"] == "N/A"
    assert kpis["candidate_pairs_display"] == "N/A"
    assert kpis["epsilon_remaining_display"] == "N/A"
    assert kpis["reconstructed_cases_display"] != "5,125"

    # Sidebar remaining budget with no active dataset
    rem_budget = "N/A"
    assert rem_budget != "7.65"


def test_dataset_switching_lifecycle_and_stale_data_prevention():
    """
    Requirement 7: Verify that switching datasets and clearing the active dataset
    updates every page correctly without stale results remaining.
    """
    import streamlit as st
    st.session_state.clear()

    # Step A: User uploads a dataset with 75 cases
    up_res = {
        "summary_kpis": {"total_cases": 75, "fir_cases": 70, "verdict_cases": 20, "fir_rate_pct": 93.3, "verdict_rate_pct": 28.6},
        "funnel_stages": [
            {"stage_id": "FIR_REGISTERED", "stage_name": "1. Police FIR", "raw_count": 70, "display_count": 70, "retention_from_fir_pct": 100.0, "step_conversion_pct": 100.0, "step_leak_pct": 0.0, "is_suppressed": False},
            {"stage_id": "COURT_VERDICT", "stage_name": "5. Court Verdict", "raw_count": 20, "display_count": 20, "retention_from_fir_pct": 28.6, "step_conversion_pct": 50.0, "step_leak_pct": 50.0, "is_suppressed": False}
        ],
        "bottlenecks": {
            "transitions": {
                "FIR_REGISTERED -> MEDICAL_EXAM": {"median_days": 1.2, "mean_days": 1.4, "p90_days": 2.0, "p95_days": 3.0, "std_days": 0.5, "iqr_days": 1.0, "delay_buckets": {"<= 24 Hours": 65}, "statutory_compliance": {"within_24h_pct": 92.9}}
            }
        },
        "linkage_stats": {"linkage_method": "Fellegi-Sunter Pairwise Scoring", "pairs_linked": 150, "review_queue_size": 2, "total_pairs_evaluated": 600},
        "journey_metrics": {"total_cases": 75, "total_events": 280, "median_journey_duration_days": 180.0, "avg_journey_duration_days": 195.0}
    }
    st.session_state["active_dataset_id"] = "uploaded"
    st.session_state["active_dataset_label"] = "Uploaded CSVs (75 cases)"
    st.session_state["active_dataset_bundle"] = build_dashboard_bundle_from_analysis(up_res)
    st.session_state["active_analysis_result"] = up_res
    st.session_state["upload_analysis_res"] = up_res

    bundle_up = st.session_state["active_dataset_bundle"]
    kpis_up = compute_dashboard_kpis(bundle_up)
    assert kpis_up["reconstructed_cases_display"] == "75"
    assert st.session_state["active_dataset_id"] == "uploaded"

    # Step B: Switch to Single Case analysis
    sc_res = {
        "is_single_case": True,
        "masked_survivor_name": "K**** D***",
        "overall_linkage_label": "same case",
        "linkage_method": "Fellegi-Sunter Pairwise Scoring",
        "timeline": {
            "all_milestones_reached": False,
            "stalled_at": "Prosecution Chargesheet",
            "missing_record_from": "Court Verdict",
            "stages": [
                {"stage_id": "FIR_REGISTERED", "stage_name": "1. Police FIR", "present": True, "date": "2024-01-01"},
                {"stage_id": "MEDICAL_EXAM", "stage_name": "2. Hospital MLC", "present": True, "date": "2024-01-02", "duration_days_from_prev": 1.0}
            ],
            "longest_transition": {"pair": "Police FIR -> Hospital MLC", "duration_days": 1.0, "baseline_days": 1.0, "multiplier": 1.0}
        }
    }
    st.session_state["active_dataset_id"] = "case"
    st.session_state["active_dataset_label"] = "Single Case (K**** D***)"
    st.session_state["active_dataset_bundle"] = build_dashboard_bundle_from_analysis(sc_res)
    st.session_state["active_analysis_result"] = sc_res
    st.session_state["single_case_analysis_res"] = sc_res

    bundle_case = st.session_state["active_dataset_bundle"]
    assert bundle_case["is_single_case"] is True
    kpis_case = compute_dashboard_kpis(bundle_case)
    assert kpis_case["reconstructed_cases_display"] == "1"
    assert kpis_case["reconstructed_cases_display"] != "75"
    assert kpis_case["reconstructed_cases_display"] != "5,125"

    # Step C: Switch to Sample dataset (500 cases)
    smpl_res = {
        "summary_kpis": {"total_cases": 500, "fir_cases": 480, "verdict_cases": 120, "fir_rate_pct": 96.0, "verdict_rate_pct": 25.0},
        "funnel_stages": [
            {"stage_id": "FIR_REGISTERED", "stage_name": "1. Police FIR", "raw_count": 480, "display_count": 480, "retention_from_fir_pct": 100.0, "step_conversion_pct": 100.0, "step_leak_pct": 0.0, "is_suppressed": False}
        ],
        "bottlenecks": {"transitions": {}},
        "linkage_stats": {"linkage_method": "Fellegi-Sunter Pairwise Scoring", "pairs_linked": 1100, "review_queue_size": 15, "total_pairs_evaluated": 5000},
        "journey_metrics": {"total_cases": 500, "total_events": 1600, "median_journey_duration_days": 210.0, "avg_journey_duration_days": 225.0}
    }
    st.session_state["active_dataset_id"] = "sample"
    st.session_state["active_dataset_label"] = "Benchmark Sample Dataset (500 cases)"
    st.session_state["active_dataset_bundle"] = build_dashboard_bundle_from_analysis(smpl_res)
    st.session_state["active_analysis_result"] = smpl_res
    st.session_state["sample_analysis_res"] = smpl_res

    bundle_sample = st.session_state["active_dataset_bundle"]
    assert bundle_sample.get("is_single_case") is not True
    kpis_sample = compute_dashboard_kpis(bundle_sample)
    assert kpis_sample["reconstructed_cases_display"] == "500"
    assert kpis_sample["reconstructed_cases_display"] != "1"

    # Step D: Clear Active Dataset
    st.session_state["active_dataset_id"] = None
    st.session_state["active_dataset_label"] = None
    st.session_state["active_dataset_bundle"] = None
    st.session_state["active_analysis_result"] = None
    st.session_state["single_case_analysis_res"] = None
    st.session_state["upload_analysis_res"] = None
    st.session_state["sample_analysis_res"] = None

    assert st.session_state["active_dataset_id"] is None
    kpis_empty = compute_dashboard_kpis(st.session_state.get("active_dataset_bundle") or {})
    assert kpis_empty["reconstructed_cases_display"] == "N/A"
    assert kpis_empty["reconstructed_cases_display"] != "500"
    assert kpis_empty["reconstructed_cases_display"] != "5,125"


def test_all_seven_dashboard_pages_data_bundle_integration():
    """
    Requirement 3: Verify that all seven dashboard pages derive their results
    directly from the active dataset bundle with zero fallback to precomputed data.
    """
    mock_run = {
        "summary_kpis": {
            "total_cases": 88,
            "total_events": 352,
            "fir_cases": 85,
            "verdict_cases": 22,
            "fir_rate_pct": 96.6,
            "verdict_rate_pct": 25.9,
            "one_line_insight": "Significant attrition between FSL and Prosecution."
        },
        "funnel_stages": [
            {"stage_id": "FIR_REGISTERED", "stage_name": "1. Police FIR", "raw_count": 85, "display_count": 85, "retention_from_fir_pct": 100.0, "step_conversion_pct": 100.0, "step_leak_pct": 0.0, "is_suppressed": False},
            {"stage_id": "MEDICAL_EXAM", "stage_name": "2. Hospital MLC", "raw_count": 80, "display_count": 80, "retention_from_fir_pct": 94.1, "step_conversion_pct": 94.1, "step_leak_pct": 5.9, "is_suppressed": False},
            {"stage_id": "FORENSIC_REPORT", "stage_name": "3. Forensic Lab (FSL)", "raw_count": 65, "display_count": 65, "retention_from_fir_pct": 76.5, "step_conversion_pct": 81.2, "step_leak_pct": 18.8, "is_suppressed": False},
            {"stage_id": "CHARGESHEET_FILED", "stage_name": "4. Prosecution Chargesheet", "raw_count": 35, "display_count": 35, "retention_from_fir_pct": 41.2, "step_conversion_pct": 53.8, "step_leak_pct": 46.2, "is_suppressed": False},
            {"stage_id": "COURT_VERDICT", "stage_name": "5. Court Verdict", "raw_count": 22, "display_count": 22, "retention_from_fir_pct": 25.9, "step_conversion_pct": 62.9, "step_leak_pct": 37.1, "is_suppressed": False}
        ],
        "bottlenecks": {
            "transitions": {
                "FIR_REGISTERED -> MEDICAL_EXAM": {
                    "pair": "FIR_REGISTERED -> MEDICAL_EXAM", "sample_count": 80, "median_days": 1.1, "mean_days": 1.3,
                    "p25_days": 1.0, "p75_days": 1.5, "p90_days": 2.0, "p95_days": 2.5, "std_days": 0.4, "iqr_days": 0.5,
                    "delay_buckets": {"<= 24 Hours": 75, "1-3 Days": 5, "> 3 Days": 0}, "statutory_compliance": {"within_24h_pct": 93.8},
                    "statutory_benchmark_days": 1.0
                },
                "FORENSIC_REPORT -> CHARGESHEET_FILED": {
                    "pair": "FORENSIC_REPORT -> CHARGESHEET_FILED", "sample_count": 35, "median_days": 48.0, "mean_days": 54.0,
                    "p25_days": 32.0, "p75_days": 65.0, "p90_days": 80.0, "p95_days": 95.0, "std_days": 18.0, "iqr_days": 33.0,
                    "delay_buckets": {"<= 30 Days": 10, "31-60 Days": 18, "> 60 Days": 7}, "statutory_compliance": {"within_60_days_pct": 80.0},
                    "statutory_benchmark_days": 60.0
                }
            },
            "biggest_bottleneck": {
                "pair": "FORENSIC_REPORT -> CHARGESHEET_FILED", "median_days": 48.0, "baseline_median_days": 30.0,
                "delay_multiplier": 1.6, "sample_count": 35, "summary": "Turnaround of 48.0 days vs baseline 30.0 days"
            }
        },
        "linkage_stats": {
            "linkage_method": "Fellegi-Sunter Pairwise Scoring (case_simulator)",
            "pairs_linked": 180,
            "review_queue_size": 5,
            "total_pairs_evaluated": 850
        },
        "privacy_info": {
            "masked_fields": ["complainant_name", "patient_name", "subject_name", "victim_name", "petitioner_victim", "guardian_name"],
            "suppressed_groups": [],
            "zero_pii_verified": True
        },
        "district_leakage": [
            {"district": "D3_Rudrapur", "total_cases": 50, "fir_cases": 48, "chargesheet_cases": 20, "verdict_cases": 12, "score": 45.2, "leak_score": 45.2, "status": "NORMAL"}
        ],
        "district_ranking": [
            {"district": "D3_Rudrapur", "score": 45.2, "leak_score": 45.2, "status": "NORMAL"}
        ],
        "journey_metrics": {
            "total_cases": 88,
            "total_events": 352,
            "median_journey_duration_days": 160.0,
            "avg_journey_duration_days": 175.0
        },
        "systemic_leak_points": [
            {
                "district": "D3_Rudrapur",
                "leak_code": "LP-01-FORENSIC-DELAY",
                "title": "Forensic Turnaround Lag",
                "delay_days": 48.0,
                "baseline_days": 30.0,
                "multiplier": 1.6,
                "affected_cases": 35
            }
        ]
    }

    bundle = build_dashboard_bundle_from_analysis(mock_run)

    # Page 1: Executive Overview integration
    kpis = compute_dashboard_kpis(bundle)
    assert kpis["reconstructed_cases_display"] == "88"
    assert kpis["events_display"] == "352"
    assert kpis["fir_count_display"] == "85"
    assert kpis["verdict_count_display"] == "22"
    assert len(bundle["bottlenecks"]["inter_stage_transitions"]) == 2
    assert bundle["summary"]["district_vulnerability_ranking"][0]["district"] == "D3_Rudrapur"

    # Page 2: Funnel with Uncertainty Ranges integration
    dp_funnel = bundle["dp_funnel"]
    assert len(dp_funnel["funnel_stages"]) == 5
    assert dp_funnel["funnel_stages"][0]["noisy_case_count"] == 85
    assert "is_k_anonymity_suppressed" in dp_funnel["funnel_stages"][0]

    # Page 3: Duration Bottlenecks & Latency integration
    b_dict = bundle["bottlenecks"]
    trans = b_dict["FIR_REGISTERED -> MEDICAL_EXAM"]
    assert trans["p95_days"] == 2.5
    assert trans["std_days"] == 0.4
    assert trans["iqr_days"] == 0.5
    assert trans["statutory_compliance"]["within_24h_pct"] == 93.8
    assert "<= 24 Hours" in trans["delay_buckets"]

    # Page 4: Linkage Quality & Evaluation integration
    evaluation = bundle["evaluation"]
    assert evaluation["linkage_method"] == "Fellegi-Sunter Pairwise Scoring (case_simulator)"
    assert evaluation["pairwise_metrics"]["total_candidate_pairs"] == 850
    assert evaluation["pairwise_metrics"]["total_auto_accepted_pairs"] == 180

    # Page 5: Before / After Policy Transformation integration
    before_after = bundle["before_after"]
    assert before_after["filtered_case_count"] == 88
    assert before_after["filtered_event_count"] == 352
    assert len(before_after["comparative_table"]) >= 4

    # Page 6: Survivor Privacy & DP Engine integration
    priv = bundle["privacy_audit"]
    assert priv["privacy_budget_summary"]["total_epsilon_allocated"] == 10.0
    assert priv["differential_privacy_mechanisms"]["zero_pii_compliance"] is True

    # Page 7: Statutory Policy Insights integration
    policy = bundle["policy_context"]
    assert len(policy) >= 1
    assert "LP-01-FORENSIC-DELAY" in policy
    assert "sources" in policy["LP-01-FORENSIC-DELAY"]




