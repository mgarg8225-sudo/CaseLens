"""
test_generate_data.py - Unit tests for Phase 1 Synthetic Data Generator.
"""

import os
import shutil
import pytest
import pandas as pd
from src.generate_data import generate_dataset, perturb_name, perturb_date

TEST_OUTPUT_DIR = "tests/test_data"


@pytest.fixture(scope="module")
def sample_dataset():
    """Generates a small test dataset for fast, deterministic unit testing."""
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)
    
    summary = generate_dataset(num_cases=500, output_dir=TEST_OUTPUT_DIR)
    yield summary
    
    # Cleanup after test module finishes
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)


def test_ground_truth_isolation(sample_dataset):
    """
    CRITICAL PRIVACY TEST:
    Verifies that 'case_id' and 'true_*' ground truth identifiers are strictly isolated
    in ground_truth.csv and NEVER leak into any agency tables.
    """
    agency_files = [
        "police_fir.csv",
        "hospital_mlc.csv",
        "forensic_lab.csv",
        "prosecutor.csv",
        "court.csv"
    ]
    
    gt_df = pd.read_csv(os.path.join(TEST_OUTPUT_DIR, "ground_truth.csv"))
    assert "case_id" in gt_df.columns, "ground_truth.csv must contain case_id"
    assert len(gt_df) == 500, "Should generate exactly 500 true cases"

    for file_name in agency_files:
        path = os.path.join(TEST_OUTPUT_DIR, file_name)
        assert os.path.exists(path), f"{file_name} must exist"
        df = pd.read_csv(path)
        
        # Ensure no ground truth identifiers are present
        assert "case_id" not in df.columns, f"{file_name} leaked case_id!"
        for col in df.columns:
            assert not col.startswith("true_"), f"{file_name} contains ground truth column {col}!"


def test_agency_schemas(sample_dataset):
    """Validates that each agency table conforms to its institutional schema."""
    fir_df = pd.read_csv(os.path.join(TEST_OUTPUT_DIR, "police_fir.csv"))
    mlc_df = pd.read_csv(os.path.join(TEST_OUTPUT_DIR, "hospital_mlc.csv"))
    fsl_df = pd.read_csv(os.path.join(TEST_OUTPUT_DIR, "forensic_lab.csv"))
    pros_df = pd.read_csv(os.path.join(TEST_OUTPUT_DIR, "prosecutor.csv"))
    court_df = pd.read_csv(os.path.join(TEST_OUTPUT_DIR, "court.csv"))

    # Distinct agency-specific ID columns
    assert "fir_number" in fir_df.columns
    assert "mlc_number" in mlc_df.columns
    assert "fsl_case_number" in fsl_df.columns
    assert "prosecution_file_no" in pros_df.columns
    assert "cnr_number" in court_df.columns

    # Date and name fields
    assert "complainant_name" in fir_df.columns and "fir_date" in fir_df.columns
    assert "patient_name" in mlc_df.columns and "examination_date" in mlc_df.columns
    assert "subject_name" in fsl_df.columns and "report_dispatch_date" in fsl_df.columns
    assert "victim_name" in pros_df.columns and "chargesheet_date" in pros_df.columns
    assert "petitioner_victim" in court_df.columns and "verdict_date" in court_df.columns


def test_injected_leak_points(sample_dataset):
    """
    Validates that the 2 key synthetic leak points are demonstrably present in the output:
    1. District 3 (Rudrapur) forensic delays are ~3x higher than other districts.
    2. District 6 (Shivgarh) FIR -> Chargesheet rate is significantly lower than average.
    """
    fsl_delays = sample_dataset["fsl_delay_by_district"]
    d3_delay = fsl_delays.get("D3_Rudrapur", 0)
    other_delays = [v for k, v in fsl_delays.items() if k != "D3_Rudrapur"]
    avg_other_delay = sum(other_delays) / len(other_delays)

    assert d3_delay > 2.2 * avg_other_delay, (
        f"District 3 delay ({d3_delay:.1f}d) should be >2.2x normal ({avg_other_delay:.1f}d)"
    )

    cs_rates = sample_dataset["fir_to_chargesheet_rate_by_district"]
    d6_rate = cs_rates.get("D6_Shivgarh", 100.0)
    other_rates = [v for k, v in cs_rates.items() if k != "D6_Shivgarh"]
    avg_other_rate = sum(other_rates) / len(other_rates)

    assert d6_rate < 40.0, f"District 6 chargesheet rate ({d6_rate}%) must be < 40%"
    assert avg_other_rate > 55.0, f"Other districts chargesheet rate ({avg_other_rate}%) must be > 55%"


def test_realistic_dropoff_funnel(sample_dataset):
    """Checks that case counts strictly drop off as cases progress through the pipeline."""
    funnel = sample_dataset["funnel"]
    assert funnel["Total Cases (Reported)"] == 500
    assert funnel["FIR Registered"] <= 500
    assert funnel["Chargesheet Filed"] < funnel["FIR Registered"]
    assert funnel["Court Verdict Reached"] < funnel["Chargesheet Filed"]


def test_noise_and_perturbations():
    """Validates the perturbation and transliteration utilities."""
    f, l, full = perturb_name("Sunita", "Sharma")
    assert isinstance(f, str) and len(f) > 0
    assert isinstance(l, str) and len(l) > 0
    assert isinstance(full, str) and len(full) > 0

    # Perturb date
    d = perturb_date(pd.to_datetime("2024-05-10").date())
    assert abs((d - pd.to_datetime("2024-05-10").date()).days) <= 5
