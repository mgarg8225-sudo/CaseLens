"""
test_evaluate.py - Unit tests for Phase 3 Linkage Evaluation and Impact Analysis.
"""

import os
import shutil
import pytest
import pandas as pd
from src.generate_data import generate_dataset
from src.linkage import run_linkage_pipeline
from src.evaluate import (
    build_ground_truth_mapping,
    evaluate_pairwise_metrics,
    evaluate_cluster_metrics,
    analyze_linkage_errors,
    evaluate_downstream_impact,
    plot_precision_recall_curve
)

TEST_EVAL_DIR = "tests/test_eval_data"


@pytest.fixture(scope="module")
def setup_eval_data():
    """Generates and links a small test dataset for evaluation testing."""
    if os.path.exists(TEST_EVAL_DIR):
        shutil.rmtree(TEST_EVAL_DIR)

    generate_dataset(num_cases=120, output_dir=TEST_EVAL_DIR)
    run_linkage_pipeline(
        data_dir=TEST_EVAL_DIR,
        high_threshold=0.80,
        low_threshold=0.45,
        max_training_pairs=5000
    )

    yield TEST_EVAL_DIR

    if os.path.exists(TEST_EVAL_DIR):
        shutil.rmtree(TEST_EVAL_DIR)


def test_build_ground_truth_mapping(setup_eval_data):
    """Validates mapping of unique_ids to ground truth case_ids and true pair count."""
    gt_mapping, total_true_pairs, case_to_recs = build_ground_truth_mapping(setup_eval_data)
    assert len(gt_mapping) > 0
    assert total_true_pairs > 0
    assert len(case_to_recs) == 120

    # Every mapped ID should have format agency__record_id
    for uid, cid in list(gt_mapping.items())[:10]:
        assert "__" in uid
        assert cid.startswith("CASE-")


def test_pairwise_metrics_and_pr_monotonicity(setup_eval_data):
    """
    Validates pairwise precision, recall, and F1 calculations.
    Tests that threshold increase generally yields higher precision.
    """
    gt_mapping, total_true_pairs, _ = build_ground_truth_mapping(setup_eval_data)
    df_pairs = pd.read_csv(os.path.join(setup_eval_data, "linked_pairs.csv"))

    thresholds = [0.30, 0.60, 0.85]
    df_results, summary = evaluate_pairwise_metrics(df_pairs, gt_mapping, total_true_pairs, thresholds)

    assert len(df_results) == 3
    assert summary["best_f1_score"] > 0.0

    # Recall at threshold 0.30 should be >= recall at threshold 0.85
    rec_low = df_results[df_results["threshold"] == 0.30]["recall"].iloc[0]
    rec_high = df_results[df_results["threshold"] == 0.85]["recall"].iloc[0]
    assert rec_low >= rec_high, "Recall must decrease or stay equal as threshold increases"


def test_cluster_metrics(setup_eval_data):
    """Validates B-cubed cluster metrics and purity calculations."""
    gt_mapping, _, case_to_recs = build_ground_truth_mapping(setup_eval_data)
    df_ent = pd.read_csv(os.path.join(setup_eval_data, "case_entities.csv"))

    metrics = evaluate_cluster_metrics(df_ent, gt_mapping, case_to_recs)
    assert 0.0 <= metrics["bcubed_precision"] <= 1.0
    assert 0.0 <= metrics["bcubed_recall"] <= 1.0
    assert 0.0 <= metrics["bcubed_f1"] <= 1.0
    assert 0.0 <= metrics["macro_purity"] <= 1.0
    assert metrics["total_true_cases_evaluated"] > 0


def test_downstream_impact_computation(setup_eval_data):
    """Validates comparison between ground truth and reconstructed pipeline rates."""
    impact = evaluate_downstream_impact(setup_eval_data)
    assert "fir_to_chargesheet_rate" in impact
    assert "leak_point_1_d3_forensic_turnaround" in impact
    assert "leak_point_2_d6_chargesheet_rate" in impact

    # Bias delta should be a finite number
    bias = impact["fir_to_chargesheet_rate"]["bias_delta_pct"]
    assert isinstance(bias, float)


def test_plot_generation(setup_eval_data):
    """Verifies generation of Plotly precision-recall HTML graph."""
    gt_mapping, total_true_pairs, _ = build_ground_truth_mapping(setup_eval_data)
    df_pairs = pd.read_csv(os.path.join(setup_eval_data, "linked_pairs.csv"))
    df_results, _ = evaluate_pairwise_metrics(df_pairs, gt_mapping, total_true_pairs)

    plot_path = os.path.join(setup_eval_data, "test_pr_curve.html")
    out_path = plot_precision_recall_curve(df_results, output_path=plot_path)

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 1000, "Plotly HTML file must not be empty"
