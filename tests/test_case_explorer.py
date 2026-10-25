"""
tests/test_case_explorer.py - Unit & Integration Tests for Case Explorer.

Tests requirements:
1. Curated benchmark selector:
   - clean_match
   - spelling_variants
   - uncertain_review
   - similar_names_kept_apart
   - missed_link
   - district_3_forensic_delay
   - district_6_chargesheet_drop
2. Random case picking from the 5,000 generated synthetic cases.
3. Fragmented source records from police, hospital, forensic, prosecution, and court:
   - Displayed exactly as they appear in each system (different IDs, spellings, ages, dates).
4. Pairwise Fellegi-Sunter match probabilities and reasons for each link.
5. Reconstructed journey timeline with days between stages and stalled/dropped flags.
6. One-line judicial verdict for every case.
7. Server-side role privacy enforcement:
   - Viewer sees masked names (e.g. S***a S***a).
   - Analyst sees pseudonymised IDs (e.g. SUBJ-8B3E / ANON-...).
8. Strictly read-only with zero storage side-effects.
9. Mandatory synthetic label: "All data shown is synthetic. No real person or case."
"""

import os
import pytest
from typing import Dict, Any

from src.case_explorer import (
    get_case_explorer_service,
    mask_name,
    pseudonymise_name,
    apply_role_privacy,
    CURATED_EXAMPLES
)


# ==============================================================================
# 1. Masking & Pseudonymisation Privacy Tests
# ==============================================================================

def test_mask_name_viewer_privacy():
    """Verifies that viewer masking properly obfuscates raw names with asterisks."""
    assert mask_name("Sunita Sharma") == "S****a S****a"
    assert mask_name("Ramesh") == "R****h"
    assert mask_name("Al") == "A*"
    assert mask_name("") == "Not Disclosed"
    assert mask_name(None) == "Not Disclosed"


def test_pseudonymise_name_analyst_privacy():
    """Verifies that analyst pseudonymisation produces deterministic anonymized tokens."""
    token1 = pseudonymise_name("Sunita Sharma", prefix="SUBJ")
    token2 = pseudonymise_name("Sunita Sharma", prefix="SUBJ")
    token_diff = pseudonymise_name("Meena Kumari", prefix="SUBJ")

    assert token1.startswith("SUBJ-")
    assert token1 == token2  # Deterministic
    assert token1 != token_diff  # Distinct subjects receive distinct tokens
    assert "Sunita" not in token1
    assert "Sharma" not in token1


def test_apply_role_privacy():
    """Verifies role-based branching between masking and pseudonymisation."""
    raw = "Pooja Verma"
    viewer_view = apply_role_privacy(raw, role="viewer")
    analyst_view = apply_role_privacy(raw, role="analyst", prefix="SUBJ")
    admin_view = apply_role_privacy(raw, role="admin", prefix="SUBJ")

    assert "*" in viewer_view
    assert "Pooja" not in viewer_view
    assert analyst_view.startswith("SUBJ-")
    assert "Pooja" not in analyst_view
    assert admin_view.startswith("SUBJ-")


# ==============================================================================
# 2. Curated Examples Benchmark Tests
# ==============================================================================

def test_all_seven_curated_examples_exist():
    """Verifies all 7 required curated judicial scenarios exist in metadata."""
    svc = get_case_explorer_service()
    examples = svc.get_curated_examples_summary()
    example_ids = {ex["id"] for ex in examples}

    required_ids = {
        "clean_match",
        "spelling_variants",
        "uncertain_review",
        "similar_names_kept_apart",
        "missed_link",
        "district_3_forensic_delay",
        "district_6_chargesheet_drop"
    }

    assert required_ids.issubset(example_ids), f"Missing required curated examples: {required_ids - example_ids}"
    assert len(examples) == 7


def test_curated_clean_match():
    """Verifies clean match produces 5 completed stages, high match probabilities, and success verdict."""
    svc = get_case_explorer_service()
    case = svc.build_case_explorer_data("clean_match", role="analyst")

    assert case["case_id"] == "clean_match"
    assert case["verdict_type"] == "success"
    assert "Fully Linked" in case["one_line_verdict"]

    # 5 fragmented records from distinct agency systems
    records = case["records"]
    assert len(records) == 5
    agency_codes = [r["agency_code"] for r in records]
    assert agency_codes == ["police_fir", "hospital_mlc", "forensic_lab", "prosecutor", "court"]

    # Verify no common ID exists across agencies
    ids = [r["record_id"] for r in records]
    assert len(set(ids)) == 5
    assert ids[0].startswith("FIR-")
    assert ids[1].startswith("MLC-")
    assert ids[2].startswith("FSL-")
    assert ids[3].startswith("PP-CHG-")
    assert ids[4].startswith("CNR-")

    # Pairwise results exist
    assert len(case["pairwise_results"]) > 0
    for p in case["pairwise_results"]:
        assert "match_probability" in p
        assert "match_weight" in p
        assert len(p["evidence"]) > 0

    # Timeline with days between stages
    timeline = case["timeline"]
    assert timeline["completed_count"] == 5
    assert timeline["is_dropped"] is False
    assert len(timeline["steps"]) == 5


def test_curated_spelling_variants():
    """Verifies phonetic/transliteration discrepancies resolve cleanly."""
    svc = get_case_explorer_service()
    case = svc.build_case_explorer_data("spelling_variants", role="viewer")

    assert case["verdict_type"] == "success"
    # Auto-accept links present despite Suneeta vs Sunita
    pw = case["pairwise_results"]
    assert any(p["decision"] == "AUTO_ACCEPT" for p in pw)


def test_curated_uncertain_review():
    """Verifies uncertain pair (0.50 <= P < 0.85) is held for review."""
    svc = get_case_explorer_service()
    case = svc.build_case_explorer_data("uncertain_review", role="analyst")

    assert case["verdict_type"] == "review"
    assert "Review Queue" in case["one_line_verdict"]
    # At least one pair falls in [0.50, 0.85)
    pw = case["pairwise_results"]
    assert any(0.50 <= p["match_probability"] < 0.85 for p in pw)
    assert any(p["decision"] == "HUMAN_REVIEW" for p in pw)


def test_curated_similar_names_kept_apart():
    """Verifies false-link protection separates two different people with common names."""
    svc = get_case_explorer_service()
    case = svc.build_case_explorer_data("similar_names_kept_apart", role="analyst")

    assert case["verdict_type"] == "danger"
    assert "Definitively Disconnected" in case["one_line_verdict"]
    pw = case["pairwise_results"]
    assert all(p["decision"] == "REJECT" for p in pw)
    assert all(p["match_probability"] < 0.50 for p in pw)


def test_curated_missed_link():
    """Verifies missed link / false negative warning when metadata is omitted."""
    svc = get_case_explorer_service()
    case = svc.build_case_explorer_data("missed_link", role="viewer")

    assert case["verdict_type"] == "warning"
    assert "Missed" in case["one_line_verdict"]


def test_curated_district_3_forensic_delay():
    """Verifies District 3 FSL latency bottleneck detection (> 60 days)."""
    svc = get_case_explorer_service()
    case = svc.build_case_explorer_data("district_3_forensic_delay", role="analyst")

    assert case["district"] == "D3_Rudrapur"
    assert case["verdict_type"] == "warning"
    assert "Forensic" in case["one_line_verdict"]

    bottlenecks = case["timeline"]["bottlenecks"]
    assert any(b["code"] == "LP-01-FORENSIC-DELAY" for b in bottlenecks)
    fsl_bn = [b for b in bottlenecks if b["code"] == "LP-01-FORENSIC-DELAY"][0]
    assert fsl_bn["delay_days"] == 276


def test_curated_district_6_chargesheet_drop():
    """Verifies District 6 chargesheet attrition leak point identification."""
    svc = get_case_explorer_service()
    case = svc.build_case_explorer_data("district_6_chargesheet_drop", role="viewer")

    assert case["district"] == "D6_Shivgarh"
    assert case["verdict_type"] == "warning"
    assert "Critical Policy Leak" in case["one_line_verdict"]
    assert case["timeline"]["is_dropped"] is True


# ==============================================================================
# 3. Generated Dataset Loading & Random Case Tests
# ==============================================================================

def test_random_case_retrieval_from_generated_dataset():
    """Verifies that random case selector retrieves a valid case from the 5,000 cases."""
    svc = get_case_explorer_service()
    rand_id = svc.get_random_case_id()
    assert rand_id.startswith("CASE-")

    case = svc.build_case_explorer_data(rand_id, role="viewer")
    assert case["case_id"] == rand_id
    assert case["synthetic_data"] is True
    assert case["synthetic_label"] == "All data shown is synthetic. No real person or case."
    assert "records" in case
    assert len(case["records"]) >= 1
    assert "timeline" in case
    assert len(case["timeline"]["steps"]) == 5
    assert len(case["one_line_verdict"]) > 0


def test_case_detail_from_dataset_case_00001():
    """Verifies that CASE-00001 loads fragmented records from the CSVs."""
    svc = get_case_explorer_service()
    case = svc.build_case_explorer_data("CASE-00001", role="analyst")

    assert case["case_id"] == "CASE-00001"
    assert case["district"] == "D2_Chandranagar"
    records = case["records"]
    assert len(records) >= 3

    # Check elapsed days are computed in timeline
    timeline_steps = case["timeline"]["steps"]
    completed_steps = [s for s in timeline_steps if s["status"] == "COMPLETED"]
    assert len(completed_steps) >= 3


# ==============================================================================
# 4. Server-Side Role Enforcement Tests
# ==============================================================================

def test_server_side_role_enforcement_viewer_vs_analyst():
    """
    Verifies server-side RBAC:
    - Viewers see masked names with asterisks (S***a S***a).
    - Analysts see pseudonymised IDs (SUBJ-8B3E / ANON-...).
    """
    svc = get_case_explorer_service()

    case_viewer = svc.build_case_explorer_data("clean_match", role="viewer")
    case_analyst = svc.build_case_explorer_data("clean_match", role="analyst")

    viewer_name = case_viewer["records"][0]["name"]
    analyst_name = case_analyst["records"][0]["name"]

    # Viewer check
    assert "*" in viewer_name
    assert "Shashi" not in viewer_name
    assert "Choudhary" not in viewer_name
    assert case_viewer["role_view"] == "masked"

    # Analyst check
    assert analyst_name.startswith("SUBJ-")
    assert "Shashi" not in analyst_name
    assert case_analyst["role_view"] == "pseudonymised"


# ==============================================================================
# 5. Zero Side-Effects & Synthetic Label Tests
# ==============================================================================

def test_zero_side_effects_on_data_files():
    """Verifies that exploring cases never modifies CSV files or review_queue."""
    rq_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "review_queue.csv")
    mtime_before = os.path.getmtime(rq_path) if os.path.exists(rq_path) else None

    svc = get_case_explorer_service()
    for ex_id in CURATED_EXAMPLES.keys():
        svc.build_case_explorer_data(ex_id, role="viewer")
        svc.build_case_explorer_data(ex_id, role="analyst")

    if os.path.exists(rq_path):
        mtime_after = os.path.getmtime(rq_path)
        assert mtime_before == mtime_after, "Case Explorer must never write to review_queue.csv!"


def test_mandatory_synthetic_label_present():
    """Verifies exact user requirement: 'All data shown is synthetic. No real person or case.'"""
    svc = get_case_explorer_service()
    case = svc.build_case_explorer_data("clean_match", role="viewer")

    assert case["synthetic_data"] is True
    assert case["synthetic_label"] == "All data shown is synthetic. No real person or case."
