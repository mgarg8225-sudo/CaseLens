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
    """Verifies consistent sleek dark plot layout configuration."""
    layout = get_plot_layout()
    assert layout["paper_bgcolor"] == "#111827"
    assert layout["plot_bgcolor"] == "#111827"
    assert "font" in layout
