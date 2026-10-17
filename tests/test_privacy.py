"""
test_privacy.py - Unit tests for Phase 5 Survivor Privacy Protection & Differential Privacy Engine.
"""

import os
import shutil
import pytest
import numpy as np
import pandas as pd

from src.privacy import (
    LaplaceMechanism,
    GaussianMechanism,
    DifferentiallyPrivateMean,
    PrivacyBudgetTracker,
    PrivacyBudgetExceededError,
    evaluate_k_anonymity,
    evaluate_l_diversity,
    evaluate_t_closeness,
    apply_kanonymity_suppression,
    simulate_reidentification_attack,
    simulate_differencing_attack,
    generate_dp_funnel,
    generate_dp_district_leakage,
    verify_zero_pii_leakage,
    run_privacy_pipeline
)
from src.generate_data import generate_dataset
from src.linkage import run_linkage_pipeline
from src.journey import build_event_log
from src.analytics import run_full_analytics_pipeline


TEST_PRIVACY_DATA_DIR = "tests/test_privacy_data"


@pytest.fixture(scope="module")
def setup_privacy_test_env():
    """Sets up a minimal end-to-end dataset for testing the privacy pipeline."""
    if os.path.exists(TEST_PRIVACY_DATA_DIR):
        shutil.rmtree(TEST_PRIVACY_DATA_DIR)

    generate_dataset(num_cases=80, output_dir=TEST_PRIVACY_DATA_DIR)
    run_linkage_pipeline(
        data_dir=TEST_PRIVACY_DATA_DIR,
        high_threshold=0.80,
        low_threshold=0.45,
        max_training_pairs=3000
    )
    df_events = build_event_log(data_dir=TEST_PRIVACY_DATA_DIR)
    df_events.to_csv(os.path.join(TEST_PRIVACY_DATA_DIR, "event_log.csv"), index=False)
    run_full_analytics_pipeline(data_dir=TEST_PRIVACY_DATA_DIR)

    yield TEST_PRIVACY_DATA_DIR

    if os.path.exists(TEST_PRIVACY_DATA_DIR):
        shutil.rmtree(TEST_PRIVACY_DATA_DIR)


# ==============================================================================
# Unit Tests for DP Mechanisms
# ==============================================================================

def test_laplace_mechanism_properties():
    """Validates scale, confidence intervals, and bounds of Laplace Mechanism."""
    laplace = LaplaceMechanism(epsilon=0.5, sensitivity=1.0, random_state=42)
    assert laplace.scale == 2.0  # Delta f / epsilon = 1.0 / 0.5 = 2.0
    
    # 95% Confidence Interval margin
    margin = laplace.confidence_bound(0.95)
    expected_margin = -2.0 * np.log(0.05)
    assert pytest.approx(margin, rel=1e-3) == expected_margin

    # Non-negative count noise
    noisy_count = laplace.add_noise_int(count=3, lower_bound=0)
    assert isinstance(noisy_count, int)
    assert noisy_count >= 0

    with pytest.raises(ValueError):
        LaplaceMechanism(epsilon=-0.1)


def test_gaussian_mechanism_properties():
    """Validates Gaussian mechanism standard deviation formula and confidence bounds."""
    gaussian = GaussianMechanism(epsilon=1.0, delta=1e-4, sensitivity=1.0, random_state=42)
    expected_sigma = (1.0 * np.sqrt(2.0 * np.log(1.25 / 1e-4))) / 1.0
    assert pytest.approx(gaussian.sigma, rel=1e-3) == expected_sigma

    margin_95 = gaussian.confidence_bound(0.95)
    assert pytest.approx(margin_95, rel=1e-3) == 1.96 * expected_sigma

    noisy_val = gaussian.add_noise(100.0)
    assert isinstance(noisy_val, float)


def test_differentially_private_mean():
    """Validates bounded sensitivity clipping and DP noisy mean estimation."""
    dp_mean = DifferentiallyPrivateMean(lower_bound=0.0, upper_bound=100.0, epsilon=1.0, random_state=42)
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    res = dp_mean.compute(samples)

    assert res["sample_size"] == 5
    assert res["raw_mean"] == 30.0
    assert 0.0 <= res["noisy_mean"] <= 100.0
    assert res["error_margin_95"] > 0


def test_privacy_budget_tracker_accounting():
    """Validates sequential composition budget tracking and exhaustion enforcement."""
    tracker = PrivacyBudgetTracker(total_epsilon=3.0, total_delta=1e-3)
    assert tracker.remaining_epsilon == 3.0
    assert tracker.spent_epsilon == 0.0
    assert not tracker.is_exhausted

    tracker.allocate(query_id="query_1", epsilon=1.0)
    assert tracker.spent_epsilon == 1.0
    assert tracker.remaining_epsilon == 2.0

    tracker.allocate(query_id="query_2", epsilon=1.5)
    assert tracker.spent_epsilon == 2.5
    assert tracker.remaining_epsilon == 0.5

    # Exceeding budget must raise PrivacyBudgetExceededError
    with pytest.raises(PrivacyBudgetExceededError):
        tracker.allocate(query_id="query_3", epsilon=1.0)

    # Ledger audit history
    ledger = tracker.get_ledger()
    assert len(ledger) == 2
    assert ledger[0]["query_id"] == "query_1"


# ==============================================================================
# Unit Tests for k-Anonymity, l-Diversity, and t-Closeness
# ==============================================================================

def test_evaluate_k_anonymity_and_suppression():
    """Validates equivalence class grouping, violation detection, and row suppression."""
    df_sample = pd.DataFrame({
        "district": ["D1", "D1", "D1", "D1", "D1", "D2", "D2"],
        "category": ["A", "A", "A", "A", "A", "B", "B"],
        "case_id": [f"C{i}" for i in range(7)]
    })

    audit = evaluate_k_anonymity(df_sample, quasi_identifiers=["district", "category"], k=5)
    assert audit["total_records"] == 7
    assert audit["total_equivalence_classes"] == 2
    assert audit["min_observed_k"] == 2  # D2-B has 2 rows
    assert audit["violating_classes_count"] == 1
    assert audit["violating_records_count"] == 2
    assert not audit["is_compliant"]

    # Apply suppression
    df_sanitized, supp_audit = apply_kanonymity_suppression(df_sample, quasi_identifiers=["district", "category"], k=5)
    assert len(df_sanitized) == 5
    assert supp_audit["suppressed_rows"] == 2
    assert (df_sanitized["district"] == "D1").all()

    # Re-evaluating sanitized data must be 100% compliant
    sanitized_audit = evaluate_k_anonymity(df_sanitized, quasi_identifiers=["district", "category"], k=5)
    assert sanitized_audit["is_compliant"]
    assert sanitized_audit["violating_records_count"] == 0


def test_evaluate_l_diversity():
    """Validates detection of homogeneous sensitive attributes within equivalence classes."""
    # Homogeneous class (all same terminal outcome -> l=1)
    df_homo = pd.DataFrame({
        "district": ["D1", "D1", "D1"],
        "category": ["A", "A", "A"],
        "terminal_stage": ["CONVICTION", "CONVICTION", "CONVICTION"]
    })
    audit_homo = evaluate_l_diversity(df_homo, quasi_identifiers=["district", "category"], sensitive_col="terminal_stage", l=2)
    assert audit_homo["min_observed_l"] == 1
    assert not audit_homo["is_compliant"]

    # Diverse class (2 distinct outcomes -> l=2)
    df_diverse = pd.DataFrame({
        "district": ["D1", "D1", "D1"],
        "category": ["A", "A", "A"],
        "terminal_stage": ["CONVICTION", "ACQUITTAL", "CONVICTION"]
    })
    audit_div = evaluate_l_diversity(df_diverse, quasi_identifiers=["district", "category"], sensitive_col="terminal_stage", l=2)
    assert audit_div["min_observed_l"] == 2
    assert audit_div["is_compliant"]


def test_evaluate_t_closeness():
    """Validates Total Variation Distance computation against global distribution."""
    df = pd.DataFrame({
        "district": ["D1", "D1", "D2", "D2"],
        "category": ["A", "A", "B", "B"],
        "stage": ["FIR", "VERDICT", "FIR", "VERDICT"]
    })
    audit = evaluate_t_closeness(df, quasi_identifiers=["district", "category"], sensitive_col="stage", t=0.1)
    assert audit["total_equivalence_classes"] == 2
    assert audit["max_observed_tvd"] == 0.0  # exactly identical distribution
    assert audit["is_compliant"]


# ==============================================================================
# Unit Tests for Attack Simulations
# ==============================================================================

def test_simulate_reidentification_attack():
    """Validates singling-out marksmanship, prosecutor, and journalist risk metrics."""
    df = pd.DataFrame({
        "district": ["D1", "D1", "D2"],  # D2 is a unique record (k=1)
        "category": ["A", "A", "B"],
        "case_id": ["C1", "C2", "C3"]
    })
    risk = simulate_reidentification_attack(df, quasi_identifiers=["district", "category"])
    assert risk["unique_records_count"] == 1
    assert pytest.approx(risk["singling_out_risk_pct"], rel=1e-2) == 33.33
    assert risk["max_prosecutor_risk_pct"] == 100.0  # 1/1 = 100%
    assert risk["risk_classification"] in ["CRITICAL", "MODERATE"]


def test_simulate_differencing_attack():
    """Validates that DP Laplace noise neutralizes membership differencing attacks."""
    df = pd.DataFrame({
        "case_id": [f"C{i}" for i in range(10)],
        "district": ["D1"] * 10,
        "case_category": ["domestic_violence"] * 10
    })
    res = simulate_differencing_attack(
        df=df,
        target_case_id="C0",
        query_filter={"district": "D1", "case_category": "domestic_violence"},
        epsilon=0.5,
        random_state=42
    )

    assert res["target_in_query_cohort"]
    assert res["raw_counts"]["raw_difference"] == 1  # Unprotected raw leak
    assert res["differential_privacy"]["protection_active"]
    assert "Laplace Noise Mask" in res["differential_privacy"]["plausible_deniability"]


# ==============================================================================
# Unit Tests for Zero-PII Verification & End-to-End Pipeline
# ==============================================================================

def test_verify_zero_pii_leakage():
    """Validates recursive PII detection and ensures sanitized data passes."""
    dirty_data = {
        "funnel": [{"stage_name": "FIR", "victim_name": "Survivor A"}]
    }
    audit_dirty = verify_zero_pii_leakage(dirty_data)
    assert not audit_dirty["zero_pii_verified"]
    assert audit_dirty["leaked_fields_count"] > 0

    clean_data = {
        "stage_id": "FIR_REGISTERED",
        "stage_name": "1. Police FIR",
        "case_count": 450,
        "retention_pct": 100.0
    }
    audit_clean = verify_zero_pii_leakage(clean_data)
    assert audit_clean["zero_pii_verified"]
    assert audit_clean["leaked_fields_count"] == 0


def test_end_to_end_privacy_pipeline(setup_privacy_test_env):
    """Validates complete Phase 5 pipeline execution, DP generation, and report persistence."""
    audit_report = run_privacy_pipeline(
        data_dir=setup_privacy_test_env,
        output_dir=setup_privacy_test_env,
        total_epsilon_budget=15.0,
        k_threshold=5,
        random_state=42
    )

    assert "privacy_budget_summary" in audit_report
    assert "k_anonymity_evaluation" in audit_report
    assert "adversarial_threat_evaluations" in audit_report
    assert audit_report["privacy_budget_summary"]["remaining_epsilon"] >= 0

    # Verify generated artifact files exist on disk
    assert os.path.exists(os.path.join(setup_privacy_test_env, "privacy_dp_funnel.json"))
    assert os.path.exists(os.path.join(setup_privacy_test_env, "privacy_dp_districts.json"))
    assert os.path.exists(os.path.join(setup_privacy_test_env, "privacy_audit_report.json"))
