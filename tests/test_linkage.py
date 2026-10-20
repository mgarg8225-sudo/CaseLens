"""
test_linkage.py - Unit tests for Phase 2 Record Linkage Engine.
"""

import os
import shutil
import pytest
import pandas as pd
from src.generate_data import generate_dataset
from src.linkage import (
    normalize_name,
    normalize_locality,
    load_and_harmonize_records,
    run_linkage_pipeline
)

TEST_DIR = "tests/test_linkage_data"


@pytest.fixture(scope="module")
def setup_test_linkage_data():
    """Generates a small test dataset for linkage testing."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    generate_dataset(num_cases=100, output_dir=TEST_DIR)
    yield TEST_DIR

    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def test_normalize_name_titles_and_transliteration():
    """Validates that titles are stripped and transliterations normalized."""
    full, first, last, p_first, p_last = normalize_name("Smt. Suneeta Sharma")
    assert first == "sunita", f"Expected 'sunita', got {first}"
    assert last == "sharma", f"Expected 'sharma', got {last}"
    assert p_first is not None and p_last is not None

    full2, first2, last2, p_first2, p_last2 = normalize_name("Mohd. Ramesh Yadav")
    assert first2 == "mohammad"
    assert p_first2 is not None


def test_normalize_name_swapped_tokens():
    """Validates that surname-first ordering is corrected."""
    full, first, last, p_first, p_last = normalize_name("Sharma Sunita")
    assert first == "sunita", "Given name should be prioritized as first_name"
    assert last == "sharma", "Surname should be placed as last_name"
    assert full == "sunita sharma"


def test_normalize_locality():
    """Validates abbreviation expansions and noise stripping in addresses."""
    loc = normalize_locality("Near Kali Mandir, Shanti Ngr, Sec 12")
    assert "nagar" in loc, "Expected 'ngr' to expand to 'nagar'"
    assert "sector" in loc, "Expected 'sec' to expand to 'sector'"
    assert "near" not in loc, "Expected filler word 'near' to be stripped"


def test_load_and_harmonize_records(setup_test_linkage_data):
    """Verifies that all 5 agencies are harmonized into a uniform schema with unique IDs."""
    df_harm = load_and_harmonize_records(data_dir=setup_test_linkage_data)
    assert len(df_harm) > 0
    assert "unique_id" in df_harm.columns
    assert "source_agency" in df_harm.columns
    assert "norm_full_name" in df_harm.columns
    assert "phonetic_first" in df_harm.columns
    assert "district" in df_harm.columns

    # Check distinct agencies present
    agencies = set(df_harm["source_agency"].unique())
    assert "police_fir" in agencies
    assert "hospital_mlc" in agencies
    assert "court" in agencies

    # Verify unique_id uniqueness
    assert df_harm["unique_id"].is_unique, "unique_id must be strictly unique"

    # Privacy check: No case_id leakage
    assert "case_id" not in df_harm.columns

    # FSL incident_date must be the incident, not sample receipt; record_date is dispatch
    df_fsl_raw = pd.read_csv(os.path.join(setup_test_linkage_data, "forensic_lab.csv"))
    assert "incident_date" in df_fsl_raw.columns
    fsl_harm = df_harm[df_harm["source_agency"] == "forensic_lab"].copy()
    assert len(fsl_harm) == len(df_fsl_raw)
    fsl_merged = fsl_harm.merge(
        df_fsl_raw,
        left_on="record_id",
        right_on="fsl_case_number",
        suffixes=("_harm", "_raw"),
    )
    assert (fsl_merged["incident_date_harm"].astype(str) == fsl_merged["incident_date_raw"].astype(str)).all()
    copied_receipt = (
        fsl_merged["incident_date_harm"].astype(str) == fsl_merged["sample_receipt_date"].astype(str)
    ).all()
    assert not copied_receipt, "Harmonized FSL incident_date must not be sample_receipt_date"
    assert (fsl_merged["record_date"].astype(str) == fsl_merged["report_dispatch_date"].astype(str)).all()


def test_linkage_pipeline_execution(setup_test_linkage_data):
    """Tests the full Splink linkage, EM training, review queue, and clustering."""
    summary = run_linkage_pipeline(
        data_dir=setup_test_linkage_data,
        high_threshold=0.80,
        low_threshold=0.45,
        max_training_pairs=5000
    )

    assert summary["total_records_harmonized"] > 0
    assert summary["total_pairs_evaluated"] > 0
    assert summary["total_case_entities"] > 0

    # Verify review queue file
    review_path = os.path.join(setup_test_linkage_data, "review_queue.csv")
    assert os.path.exists(review_path), "review_queue.csv must be created"
    df_review = pd.read_csv(review_path)
    if len(df_review) > 0:
        assert (df_review["match_probability"] >= 0.45).all()
        assert "review_status" in df_review.columns
        assert (df_review["review_status"] == "PENDING_INVESTIGATION").all()

    # Verify case entities file
    entities_path = os.path.join(setup_test_linkage_data, "case_entities.csv")
    assert os.path.exists(entities_path), "case_entities.csv must be created"
    df_entities = pd.read_csv(entities_path)
    assert "entity_id" in df_entities.columns
    assert "unique_id" in df_entities.columns
