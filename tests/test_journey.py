"""
test_journey.py - Unit tests for Phase 4 Case Journey Reconstruction & Process Mining.
"""

import os
import shutil
import pytest
import pandas as pd
from src.generate_data import generate_dataset
from src.linkage import run_linkage_pipeline
from src.journey import (
    build_event_log,
    discover_process_variants,
    compute_directly_follows_graph,
    run_journey_analysis
)

TEST_JOURNEY_DIR = "tests/test_journey_data"


@pytest.fixture(scope="module")
def setup_journey_data():
    """Sets up a small end-to-end dataset with linked entities for journey testing."""
    if os.path.exists(TEST_JOURNEY_DIR):
        shutil.rmtree(TEST_JOURNEY_DIR)

    generate_dataset(num_cases=120, output_dir=TEST_JOURNEY_DIR)
    run_linkage_pipeline(
        data_dir=TEST_JOURNEY_DIR,
        high_threshold=0.80,
        low_threshold=0.45,
        max_training_pairs=5000
    )

    yield TEST_JOURNEY_DIR

    if os.path.exists(TEST_JOURNEY_DIR):
        shutil.rmtree(TEST_JOURNEY_DIR)


def test_build_event_log(setup_journey_data):
    """Validates that the event log contains required process mining fields and chronological sorting."""
    df_events = build_event_log(data_dir=setup_journey_data)
    assert len(df_events) > 0

    required_cols = ["case_id", "activity", "timestamp", "agency", "district", "case_category", "record_id"]
    for col in required_cols:
        assert col in df_events.columns, f"Missing required column {col} in event log"

    # Verify chronological ordering per case
    for _, grp in df_events.groupby("case_id"):
        dts = pd.to_datetime(grp["timestamp"])
        assert dts.is_monotonic_increasing, "Events for a case entity must be sorted chronologically"


def test_discover_process_variants(setup_journey_data):
    """Validates trace variant discovery, path lengths, and entry/exit point distributions."""
    df_events = build_event_log(data_dir=setup_journey_data)
    df_variants, metrics = discover_process_variants(df_events)

    assert len(df_variants) > 0
    assert metrics["total_cases_reconstructed"] > 0
    assert metrics["unique_journey_variants"] > 0
    assert "start_activities_pct" in metrics
    assert "exit_points_pct" in metrics

    # The sum of variant shares should be ~100%
    total_share = df_variants["share_pct"].sum()
    assert 99.0 <= total_share <= 101.0, f"Variant shares should sum to ~100%, got {total_share}"


def test_compute_directly_follows_graph(setup_journey_data):
    """Validates transition matrix calculations and duration metrics."""
    df_events = build_event_log(data_dir=setup_journey_data)
    df_dfg, summary = compute_directly_follows_graph(df_events)

    assert not df_dfg.empty
    assert "transition_count" in df_dfg.columns
    assert "avg_duration_days" in df_dfg.columns
    assert (df_dfg["avg_duration_days"] >= 0).all(), "Durations must be non-negative"

    assert "overall_dfg" in summary
    assert "district_dfg" in summary


def test_end_to_end_journey_pipeline(setup_journey_data):
    """Tests the full run_journey_analysis pipeline and artifact persistence."""
    summary = run_journey_analysis(data_dir=setup_journey_data)

    assert summary["total_cases"] > 0
    assert summary["total_events"] > 0
    assert summary["avg_duration_days"] >= 0

    # Check generated files
    assert os.path.exists(os.path.join(setup_journey_data, "event_log.csv"))
    assert os.path.exists(os.path.join(setup_journey_data, "journey_dfg.json"))
    assert os.path.exists(os.path.join(setup_journey_data, "journey_variants.json"))
    assert os.path.exists(os.path.join(setup_journey_data, "journey_metrics.json"))
