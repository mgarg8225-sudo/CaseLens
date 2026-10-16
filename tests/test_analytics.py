"""
test_analytics.py - Unit tests for Phase 4 Longitudinal Case Journey Analytics Engine.
"""

import os
import shutil
import pytest
import pandas as pd
from src.generate_data import generate_dataset
from src.linkage import run_linkage_pipeline
from src.journey import build_event_log
from src.analytics import (
    compute_funnel_metrics,
    compute_district_leakage_diagnostics,
    compute_category_analytics,
    compute_transition_bottlenecks,
    run_duckdb_analytics,
    apply_privacy_k_anonymity,
    run_full_analytics_pipeline,
    CANONICAL_STAGES
)

TEST_ANALYTICS_DIR = "tests/test_analytics_data"


@pytest.fixture(scope="module")
def setup_analytics_data():
    """Sets up a synthetic dataset and runs linkage and journey event logging for analytics testing."""
    if os.path.exists(TEST_ANALYTICS_DIR):
        shutil.rmtree(TEST_ANALYTICS_DIR)

    generate_dataset(num_cases=140, output_dir=TEST_ANALYTICS_DIR)
    run_linkage_pipeline(
        data_dir=TEST_ANALYTICS_DIR,
        high_threshold=0.80,
        low_threshold=0.45,
        max_training_pairs=5000
    )

    df_events = build_event_log(data_dir=TEST_ANALYTICS_DIR)
    df_events.to_csv(os.path.join(TEST_ANALYTICS_DIR, "event_log.csv"), index=False)

    yield TEST_ANALYTICS_DIR

    if os.path.exists(TEST_ANALYTICS_DIR):
        shutil.rmtree(TEST_ANALYTICS_DIR)


def test_compute_funnel_metrics(setup_analytics_data):
    """Validates multi-stage case funnel calculations, retention rates, and step leak rates."""
    df_events = pd.read_csv(os.path.join(setup_analytics_data, "event_log.csv"))
    funnel = compute_funnel_metrics(df_events)

    assert funnel["total_reconstructed_cases"] > 0
    assert funnel["fir_registered_cases"] > 0
    assert len(funnel["funnel_stages"]) == len(CANONICAL_STAGES)

    # Check stage invariants
    for stage in funnel["funnel_stages"]:
        assert stage["case_count"] >= 0
        assert 0.0 <= stage["cumulative_retention_pct"] <= 100.0
        assert stage["step_conversion_pct"] >= 0.0
        assert stage["step_leak_pct"] >= 0.0

    # District and category sub-funnels
    assert len(funnel["district_funnels"]) > 0
    assert len(funnel["category_funnels"]) > 0


def test_compute_district_leakage_diagnostics(setup_analytics_data):
    """Validates district performance evaluation and leak vulnerability scoring."""
    df_events = pd.read_csv(os.path.join(setup_analytics_data, "event_log.csv"))
    diagnostics = compute_district_leakage_diagnostics(df_events)

    assert "districts" in diagnostics
    assert "state_benchmarks" in diagnostics
    assert len(diagnostics["districts"]) > 0

    for dist, metrics in diagnostics["districts"].items():
        assert "vulnerability_score" in metrics
        assert 0.0 <= metrics["vulnerability_score"] <= 100.0
        assert metrics["status"] in ["NORMAL", "CRITICAL"]
        assert metrics["total_cases"] > 0


def test_compute_category_analytics(setup_analytics_data):
    """Validates category-level breakdown of attrition and durations."""
    df_events = pd.read_csv(os.path.join(setup_analytics_data, "event_log.csv"))
    cat_summary = compute_category_analytics(df_events)

    assert len(cat_summary) > 0
    for cat, metrics in cat_summary.items():
        assert metrics["total_cases"] > 0
        assert metrics["avg_duration_days"] >= 0
        assert metrics["median_duration_days"] >= 0
        assert 0.0 <= metrics["fir_to_chargesheet_rate_pct"] <= 100.0


def test_compute_transition_bottlenecks(setup_analytics_data):
    """Validates non-parametric percentiles and delay histograms."""
    df_events = pd.read_csv(os.path.join(setup_analytics_data, "event_log.csv"))
    bottlenecks = compute_transition_bottlenecks(df_events)

    assert len(bottlenecks) > 0
    for pair, stats in bottlenecks.items():
        assert stats["sample_count"] >= 5
        assert stats["min_days"] <= stats["median_days"] <= stats["p90_days"] <= stats["max_days"]
        assert stats["iqr_days"] >= 0

        # Verify delay bucket counts sum to sample count
        bucket_sum = sum(stats["delay_buckets"].values())
        assert bucket_sum == stats["sample_count"]


def test_run_duckdb_analytics(setup_analytics_data):
    """Validates DuckDB OLAP query execution on event log and case entities."""
    duck_res = run_duckdb_analytics(data_dir=setup_analytics_data)

    assert "monthly_trend_records" in duck_res
    assert duck_res["monthly_trend_records"] > 0
    assert "cube_slice" in duck_res
    assert len(duck_res["cube_slice"]) > 0


def test_apply_privacy_k_anonymity():
    """Validates that cells with counts < k are masked to protect survivor privacy."""
    test_data = {
        "group_a": {
            "case_count": 3,
            "avg_duration": 45.2
        },
        "group_b": {
            "case_count": 25,
            "avg_duration": 30.1
        }
    }

    anonymized = apply_privacy_k_anonymity(test_data, k=5)
    assert anonymized["group_a"]["case_count"] == "<5"
    assert anonymized["group_a"]["avg_duration"] == 45.2
    assert anonymized["group_b"]["case_count"] == 25


def test_end_to_end_analytics_pipeline(setup_analytics_data):
    """Validates end-to-end analytics execution and file persistence."""
    summary = run_full_analytics_pipeline(data_dir=setup_analytics_data)

    assert "pipeline_health" in summary
    assert "district_vulnerability_ranking" in summary
    assert summary["pipeline_health"]["total_reconstructed_cases"] > 0

    # Verify generated JSON artifacts
    assert os.path.exists(os.path.join(setup_analytics_data, "analytics_funnel.json"))
    assert os.path.exists(os.path.join(setup_analytics_data, "analytics_district_leakage.json"))
    assert os.path.exists(os.path.join(setup_analytics_data, "analytics_category.json"))
    assert os.path.exists(os.path.join(setup_analytics_data, "analytics_bottlenecks.json"))
    assert os.path.exists(os.path.join(setup_analytics_data, "analytics_summary.json"))
