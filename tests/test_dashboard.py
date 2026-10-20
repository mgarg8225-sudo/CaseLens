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
    DATA_DIR
)


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


