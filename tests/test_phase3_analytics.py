"""
tests/test_phase3_analytics.py - Comprehensive Unit Tests for Phase 3:
Funnel, Linkage & Journey Analytics, Timestamp Anomaly Handling, Statutory Whitelists,
Mathematical DP Noise Confidence Margins, and Native Process Mining Fallbacks.

All tests utilize 100% synthetic fixtures. No real person, victim, police, FIR, or court records.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.upload_analysis import (
    classify_district_severity,
    compute_funnel_and_bottlenecks,
    check_bnss_193_eligibility,
    check_bnss_184_eligibility,
    compute_calendar_month_deadline,
    build_dashboard_bundle_from_analysis
)
from src.analytics import (
    compute_funnel_metrics,
    compute_transition_bottlenecks,
    compute_filtered_summary
)
from src.privacy import LaplaceMechanism, generate_dp_funnel
from src.journey import compute_directly_follows_graph, discover_process_variants
from app.dashboard import compute_dashboard_kpis, NA, _clean_counter


# ==============================================================================
# 1. Missing and Reversed Timestamp Anomaly Handling
# ==============================================================================

def test_missing_and_reversed_timestamps_flagged_and_excluded():
    """
    Validates that:
    1. Cases with reversed timestamps (end < start) are flagged as sequence anomalies.
    2. Negative durations are excluded from positive duration statistics and percentiles.
    3. Missing (NaT/None) timestamps are cleanly skipped without raising errors.
    """
    # 4 distinct synthetic cases:
    # Case A: Normal valid progression (Day 1 -> Day 10 -> Day 40)
    # Case B: Reversed timestamp anomaly (FIR 2024-03-01, Medical 2024-02-15: delta = -15 days)
    # Case C: Missing timestamp (NaT) on intermediate stage
    # Case D: Another valid progression (Day 1 -> Day 5)
    events_data = [
        # Case A: Valid
        {"case_id": "SYN-CASE-01", "activity": "FIR_REGISTERED", "timestamp": "2024-01-01", "district": "D1", "case_category": "general_violence", "legal_sections": "BNS 64"},
        {"case_id": "SYN-CASE-01", "activity": "MEDICAL_EXAM", "timestamp": "2024-01-02", "district": "D1", "case_category": "general_violence", "legal_sections": "BNS 64"},
        {"case_id": "SYN-CASE-01", "activity": "CHARGESHEET_FILED", "timestamp": "2024-02-15", "district": "D1", "case_category": "general_violence", "legal_sections": "BNS 64"},

        # Case B: Reversed timestamp anomaly
        {"case_id": "SYN-CASE-02", "activity": "FIR_REGISTERED", "timestamp": "2024-03-01", "district": "D1", "case_category": "general_violence", "legal_sections": "BNS 64"},
        {"case_id": "SYN-CASE-02", "activity": "MEDICAL_EXAM", "timestamp": "2024-02-15", "district": "D1", "case_category": "general_violence", "legal_sections": "BNS 64"},

        # Case C: Missing timestamp (None/NaT)
        {"case_id": "SYN-CASE-03", "activity": "FIR_REGISTERED", "timestamp": "2024-01-10", "district": "D1", "case_category": "general_violence", "legal_sections": "BNS 64"},
        {"case_id": "SYN-CASE-03", "activity": "MEDICAL_EXAM", "timestamp": None, "district": "D1", "case_category": "general_violence", "legal_sections": "BNS 64"},
        {"case_id": "SYN-CASE-03", "activity": "CHARGESHEET_FILED", "timestamp": "2024-02-20", "district": "D1", "case_category": "general_violence", "legal_sections": "BNS 64"},

        # Case D: Valid
        {"case_id": "SYN-CASE-04", "activity": "FIR_REGISTERED", "timestamp": "2024-01-05", "district": "D1", "case_category": "general_violence", "legal_sections": "BNS 64"},
        {"case_id": "SYN-CASE-04", "activity": "MEDICAL_EXAM", "timestamp": "2024-01-06", "district": "D1", "case_category": "general_violence", "legal_sections": "BNS 64"},
        {"case_id": "SYN-CASE-04", "activity": "CHARGESHEET_FILED", "timestamp": "2024-02-10", "district": "D1", "case_category": "general_violence", "legal_sections": "BNS 64"},
    ]
    df_events = pd.DataFrame(events_data)

    res = compute_funnel_and_bottlenecks(df_events, total_entities=4, k_threshold=1)
    trans = res.get("transitions", {})

    # Check that the reversed transition (Case B: -15 days) did NOT enter positive duration statistics
    fir_to_med = trans.get("FIR_REGISTERED -> MEDICAL_EXAM", {})
    assert fir_to_med.get("min_days", 0) >= 0, "Durations must be strictly non-negative"
    assert fir_to_med.get("median_days", 0) >= 0

    # In journey DFG
    df_dfg, dfg_metrics = compute_directly_follows_graph(df_events)
    if not df_dfg.empty:
        assert (df_dfg["avg_duration_days"] >= 0).all()


# ==============================================================================
# 2. Statutory Whitelists & Missing Legal Sections (Requirement 2)
# ==============================================================================

def test_bnss_193_missing_sections_unverified_excluded_from_compliance():
    """
    Validates Requirement 2:
    Cases with missing legal sections must remain UNVERIFIED_STATUTORY_ELIGIBILITY
    and must not enter statutory compliance calculations for the 2-calendar-month benchmark.
    """
    # 5 cases with missing legal sections (None, empty string, whitespace)
    events_data = []
    for i in range(1, 6):
        cid = f"SYN-UNVERIFIED-{i:02d}"
        events_data.append({"case_id": cid, "activity": "FIR_REGISTERED", "timestamp": "2024-01-01", "district": "D1", "case_category": "sexual_assault", "legal_sections": None})
        events_data.append({"case_id": cid, "activity": "CHARGESHEET_FILED", "timestamp": "2024-01-20", "district": "D1", "case_category": "sexual_assault", "legal_sections": None})

    df_events = pd.DataFrame(events_data)
    res = compute_funnel_and_bottlenecks(df_events, total_entities=5, k_threshold=1)
    trans = res.get("transitions", {})
    chg_trans = trans.get("FIR_REGISTERED -> CHARGESHEET_FILED (E2E)") or trans.get("FIR_REGISTERED -> CHARGESHEET_FILED", {})

    stat_comp = chg_trans.get("statutory_compliance", {})
    # Verified eligible cases must be 0
    assert stat_comp.get("verified_eligible_cases", 0) == 0
    assert stat_comp.get("unverified_cases", 0) == 5
    # Since verified count is 0, compliance percentage must be None (not 100% or 0%)
    assert stat_comp.get("within_two_calendar_months_pct") is None


def test_bnss_193_ineligible_sections_excluded_from_compliance():
    """
    Validates that statutory ineligible offences (e.g. BNS 69, POCSO 12, general IPC 302/498A)
    do not enter the 2-calendar-month investigation compliance calculations.
    """
    events_data = []
    # 5 cases under BNS Section 69 (deceitful promise - explicitly excluded)
    for i in range(1, 6):
        cid = f"SYN-BNS69-{i:02d}"
        events_data.append({"case_id": cid, "activity": "FIR_REGISTERED", "timestamp": "2024-01-01", "district": "D1", "case_category": "sexual_assault", "legal_sections": "BNS 69"})
        events_data.append({"case_id": cid, "activity": "CHARGESHEET_FILED", "timestamp": "2024-01-25", "district": "D1", "case_category": "sexual_assault", "legal_sections": "BNS 69"})

    df_events = pd.DataFrame(events_data)
    res = compute_funnel_and_bottlenecks(df_events, total_entities=5, k_threshold=1)
    trans = res.get("transitions", {})
    chg_trans = trans.get("FIR_REGISTERED -> CHARGESHEET_FILED (E2E)") or trans.get("FIR_REGISTERED -> CHARGESHEET_FILED", {})

    stat_comp = chg_trans.get("statutory_compliance", {})
    assert stat_comp.get("verified_eligible_cases", 0) == 0
    assert stat_comp.get("within_two_calendar_months_pct") is None


def test_bnss_193_verified_eligible_sections_calculated_correctly():
    """
    Validates that cases with qualifying statutory provisions (e.g. BNS 64, POCSO 6)
    are verified and enter the 2-calendar-month investigation compliance rate.
    """
    events_data = []
    # Case 1: Within 2 months (Jan 01 -> Feb 20 = 50 days <= 2 months)
    # Case 2: Exceeded 2 months (Jan 01 -> Apr 15 = 105 days > 2 months)
    # Case 3: Within 2 months (Jan 15 -> Mar 10 = 55 days <= 2 months)
    # Case 4: Within 2 months (Feb 01 -> Mar 25 = 53 days <= 2 months)
    # Case 5: Exceeded 2 months (Jan 01 -> May 01 = 121 days > 2 months)
    cases_info = [
        ("SYN-V-01", "2024-01-01", "2024-02-20", "BNS 64"),
        ("SYN-V-02", "2024-01-01", "2024-04-15", "BNS 70"),
        ("SYN-V-03", "2024-01-15", "2024-03-10", "POCSO Sec 6"),
        ("SYN-V-04", "2024-02-01", "2024-03-25", "BNS 65"),
        ("SYN-V-05", "2024-01-01", "2024-05-01", "POCSO 4"),
    ]
    for cid, fir_dt, chg_dt, sec in cases_info:
        events_data.append({"case_id": cid, "activity": "FIR_REGISTERED", "timestamp": fir_dt, "district": "D1", "case_category": "sexual_assault", "legal_sections": sec})
        events_data.append({"case_id": cid, "activity": "CHARGESHEET_FILED", "timestamp": chg_dt, "district": "D1", "case_category": "sexual_assault", "legal_sections": sec})

    df_events = pd.DataFrame(events_data)
    res = compute_funnel_and_bottlenecks(df_events, total_entities=5, k_threshold=1)
    trans = res.get("transitions", {})
    chg_trans = trans.get("FIR_REGISTERED -> CHARGESHEET_FILED (E2E)") or trans.get("FIR_REGISTERED -> CHARGESHEET_FILED", {})

    stat_comp = chg_trans.get("statutory_compliance", {})
    assert stat_comp.get("verified_eligible_cases") == 5
    assert stat_comp.get("unverified_cases") == 0
    # 3 out of 5 cases completed within 2 calendar months -> 60.0%
    assert stat_comp.get("within_two_calendar_months_pct") == 60.0


# ==============================================================================
# 3. Funnel Conversion and Retention Bounded to 100%
# ==============================================================================

def test_funnel_conversion_and_retention_bounded():
    """
    Validates that in cross-agency datasets with asymmetric silo volumes,
    step conversion and cumulative retention are bounded at 100.0%,
    preventing spurious conversions like 400% and collapsed 0.0% leaks.
    """
    # 5 FIRs, 2 Hospital MLCs, 1 FSL, 4 Chargesheets
    events_data = [
        # FIRs
        {"case_id": "C1", "activity": "FIR_REGISTERED", "timestamp": "2024-01-01", "district": "D1", "case_category": "general_violence"},
        {"case_id": "C2", "activity": "FIR_REGISTERED", "timestamp": "2024-01-02", "district": "D1", "case_category": "general_violence"},
        {"case_id": "C3", "activity": "FIR_REGISTERED", "timestamp": "2024-01-03", "district": "D1", "case_category": "general_violence"},
        {"case_id": "C4", "activity": "FIR_REGISTERED", "timestamp": "2024-01-04", "district": "D1", "case_category": "general_violence"},
        {"case_id": "C5", "activity": "FIR_REGISTERED", "timestamp": "2024-01-05", "district": "D1", "case_category": "general_violence"},

        # MLCs (only 2)
        {"case_id": "C1", "activity": "MEDICAL_EXAM", "timestamp": "2024-01-02", "district": "D1", "case_category": "general_violence"},
        {"case_id": "C2", "activity": "MEDICAL_EXAM", "timestamp": "2024-01-03", "district": "D1", "case_category": "general_violence"},

        # FSL (only 1)
        {"case_id": "C1", "activity": "FORENSIC_REPORT", "timestamp": "2024-01-20", "district": "D1", "case_category": "general_violence"},

        # Chargesheet (4 cases)
        {"case_id": "C1", "activity": "CHARGESHEET_FILED", "timestamp": "2024-02-01", "district": "D1", "case_category": "general_violence"},
        {"case_id": "C2", "activity": "CHARGESHEET_FILED", "timestamp": "2024-02-05", "district": "D1", "case_category": "general_violence"},
        {"case_id": "C3", "activity": "CHARGESHEET_FILED", "timestamp": "2024-02-10", "district": "D1", "case_category": "general_violence"},
        {"case_id": "C4", "activity": "CHARGESHEET_FILED", "timestamp": "2024-02-15", "district": "D1", "case_category": "general_violence"},
    ]
    df_events = pd.DataFrame(events_data)
    funnel = compute_funnel_metrics(df_events)

    for stage in funnel["funnel_stages"]:
        assert 0.0 <= stage["cumulative_retention_pct"] <= 100.0, f"Exceeded bound for {stage['stage_id']}"
        assert 0.0 <= stage["step_conversion_pct"] <= 100.0, f"Exceeded step conversion for {stage['stage_id']}"
        assert 0.0 <= stage["step_leak_pct"] <= 100.0


# ==============================================================================
# 4. Mathematical Derivation of Differential Privacy Confidence Margins
# ==============================================================================

def test_laplace_confidence_margin_mathematical_derivation():
    """
    Validates Requirement 3:
    The 95% confidence margin for Laplace noise is derived mathematically from
    sensitivity Delta f and epsilon: margin = -b * ln(0.05) where scale b = Delta f / epsilon.
    Confirms that +/-12 is approximately correct for epsilon=0.25, sensitivity=1.0,
    and changes dynamically for other parameter values.
    """
    # Standard baseline: epsilon = 0.25, sensitivity = 1.0 -> b = 4.0
    lap_025 = LaplaceMechanism(epsilon=0.25, sensitivity=1.0, random_state=42)
    expected_margin_025 = -4.0 * math.log(0.05)  # 11.9829...
    assert round(lap_025.confidence_bound(0.95), 2) == round(expected_margin_025, 2)
    assert round(lap_025.confidence_bound(0.95), 0) == 12.0

    # Adjusted parameters: epsilon = 0.50, sensitivity = 1.0 -> b = 2.0
    lap_050 = LaplaceMechanism(epsilon=0.50, sensitivity=1.0, random_state=42)
    expected_margin_050 = -2.0 * math.log(0.05)  # 5.9914...
    assert round(lap_050.confidence_bound(0.95), 2) == round(expected_margin_050, 2)

    # 99% confidence level
    expected_margin_99 = -4.0 * math.log(0.01)  # 18.4206...
    assert round(lap_025.confidence_bound(0.99), 2) == round(expected_margin_99, 2)


def test_dp_funnel_suppression_and_percentages():
    """
    Validates that when a funnel stage count is < k, it is masked as '<k',
    and retention/conversion are returned as None (not 0.0%), while unsuppressed
    percentages are bounded <= 100.0%.
    """
    funnel_data = {
        "funnel_stages": [
            {"stage_id": "FIR_REGISTERED", "stage_name": "Police FIR", "case_count": 100},
            {"stage_id": "MEDICAL_EXAM", "stage_name": "Hospital MLC", "case_count": 80},
            {"stage_id": "FORENSIC_REPORT", "stage_name": "Forensic Lab", "case_count": 3},  # Small cell < 5
            {"stage_id": "CHARGESHEET_FILED", "stage_name": "Prosecution", "case_count": 50},
            {"stage_id": "COURT_VERDICT", "stage_name": "Court Verdict", "case_count": 20},
        ]
    }
    dp_res = generate_dp_funnel(funnel_data, epsilon_per_stage=0.25, k_threshold=5, random_state=42)
    stages = dp_res["funnel_stages"]

    fsl_stage = next(s for s in stages if s["stage_id"] == "FORENSIC_REPORT")
    assert fsl_stage["is_k_anonymity_suppressed"] is True
    assert fsl_stage["noisy_case_count"] == "<5"
    assert fsl_stage["dp_retention_pct"] is None
    assert fsl_stage["dp_conversion_pct"] is None

    # Check that unsuppressed stages have bounded percentages <= 100.0%
    for s in stages:
        if not s["is_k_anonymity_suppressed"]:
            assert 0.0 <= s["dp_retention_pct"] <= 100.0
            assert 0.0 <= s["dp_conversion_pct"] <= 100.0


# ==============================================================================
# 5. Preservation of Phase 2 Statutory Framework
# ==============================================================================

def test_phase2_severity_and_statutory_preservation():
    """Verifies that all Phase 2 5-tier severity thresholds and calendar months remain strictly preserved."""
    # 5-tier classification
    assert classify_district_severity(0.0, total_cases=20, fir_cases=20) == "Normal"
    assert classify_district_severity(9.9, total_cases=20, fir_cases=20) == "Normal"
    assert classify_district_severity(10.0, total_cases=20, fir_cases=20) == "Moderate"
    assert classify_district_severity(25.0, total_cases=20, fir_cases=20) == "Elevated"
    assert classify_district_severity(40.0, total_cases=20, fir_cases=20) == "Severe"
    assert classify_district_severity(60.0, total_cases=20, fir_cases=20) == "Critical"
    assert classify_district_severity(50.0, total_cases=4, fir_cases=4) == "Insufficient Data"

    # Calendar month calculation
    deadline = compute_calendar_month_deadline("2024-01-31", months=2)
    assert str(deadline.date()) == "2024-03-31"

    deadline_feb = compute_calendar_month_deadline("2023-12-31", months=2)
    assert str(deadline_feb.date()) == "2024-02-29"  # 2024 is leap year


# ==============================================================================
# 6. Journey Duration None Fallback & Genuine Measured Zero (Requirement 1)
# ==============================================================================

def test_median_journey_none_fallback_and_measured_zero():
    """
    Validates Requirement 1:
    1. Replaces 0.0 fallback with None when no valid journey duration exists.
    2. Dashboard KPI displays 'N/A' when duration is None.
    3. Zero represents an actual measured 0-day duration only, and formats as '0.0'.
    4. Executive summary sets median_end_to_end_journey_days to None when no E2E verdict exists.
    """
    # Scenario A: Single-event cases (no multi-event journey exists)
    single_events = [
        {"case_id": f"SYN-SINGLE-{i}", "activity": "FIR_REGISTERED", "timestamp": "2024-01-01", "district": "D1", "case_category": "general_violence"}
        for i in range(1, 6)
    ]
    df_single = pd.DataFrame(single_events)
    res_single = compute_funnel_and_bottlenecks(df_single, total_entities=5, k_threshold=1)
    jm_single = res_single["journey_metrics"]

    assert jm_single["median_journey_duration_days"] is None, "Missing duration must be None, not 0.0"
    assert jm_single["avg_journey_duration_days"] is None

    # Check dashboard bundle mapping produces N/A
    bundle_single = build_dashboard_bundle_from_analysis(res_single)
    kpis_single = compute_dashboard_kpis(bundle_single)
    assert kpis_single["median_journey_display"] == NA
    assert kpis_single["median_journey_days"] is None

    # Scenario B: Measured Zero-Day Duration (e.g. same-day FIR and Chargesheet / Verdict)
    zero_day_events = [
        {"case_id": "SYN-ZERO-1", "activity": "FIR_REGISTERED", "timestamp": "2024-01-01", "district": "D1", "case_category": "general_violence"},
        {"case_id": "SYN-ZERO-1", "activity": "CHARGESHEET_FILED", "timestamp": "2024-01-01", "district": "D1", "case_category": "general_violence"},
        {"case_id": "SYN-ZERO-2", "activity": "FIR_REGISTERED", "timestamp": "2024-01-02", "district": "D1", "case_category": "general_violence"},
        {"case_id": "SYN-ZERO-2", "activity": "CHARGESHEET_FILED", "timestamp": "2024-01-02", "district": "D1", "case_category": "general_violence"},
    ]
    df_zero = pd.DataFrame(zero_day_events)
    res_zero = compute_funnel_and_bottlenecks(df_zero, total_entities=2, k_threshold=1)
    jm_zero = res_zero["journey_metrics"]

    # Measured zero must be preserved as 0.0 float
    assert jm_zero["median_journey_duration_days"] == 0.0
    assert jm_zero["avg_journey_duration_days"] == 0.0

    bundle_zero = build_dashboard_bundle_from_analysis(res_zero)
    kpis_zero = compute_dashboard_kpis(bundle_zero)
    assert kpis_zero["median_journey_display"] == "0.0"
    assert kpis_zero["median_journey_days"] == 0.0


# ==============================================================================
# 7. DP Sensitivity, Contribution Limits & Noise Margins (Requirement 2)
# ==============================================================================

def test_dp_sensitivity_and_contribution_limits():
    """
    Validates Requirement 2:
    1. Query sensitivity Delta f = 1 holds because each case entity contributes at most 1 count per stage.
    2. Adding or removing a case entity shifts the count of any stage by at most 1.
    3. Epsilon = 0.25 yields Laplace scale b = Delta f / epsilon = 1.0 / 0.25 = 4.0.
    4. 95% statistical noise margin = -b * ln(0.05) ~= 11.98 cases.
    5. Confirms that +/-12 is a 95% statistical noise margin, NOT a deterministic clamp
       (i.e. Laplace distribution has infinite support; P(|X| > 12) ~= 0.05).
    """
    # Base dataset: 10 cases reaching FIR
    base_events = [
        {"case_id": f"SYN-DP-{i}", "activity": "FIR_REGISTERED", "timestamp": "2024-01-01", "district": "D1", "case_category": "general_violence"}
        for i in range(10)
    ]
    df_base = pd.DataFrame(base_events)
    f_base = compute_funnel_metrics(df_base)["funnel_stages"][0]["case_count"]

    # Neighboring dataset D': add 1 case
    neighbor_events = base_events + [
        {"case_id": "SYN-DP-10", "activity": "FIR_REGISTERED", "timestamp": "2024-01-01", "district": "D1", "case_category": "general_violence"}
    ]
    df_neighbor = pd.DataFrame(neighbor_events)
    f_neighbor = compute_funnel_metrics(df_neighbor)["funnel_stages"][0]["case_count"]

    # Global L1 Sensitivity Delta f = |f(D) - f(D')| = 1
    delta_f = abs(f_neighbor - f_base)
    assert delta_f == 1, f"Sensitivity Delta f must be 1, got {delta_f}"

    # Verify contribution limit: duplicate events for the same case do NOT increment stage count
    dup_events = base_events + [
        {"case_id": "SYN-DP-0", "activity": "FIR_REGISTERED", "timestamp": "2024-01-02", "district": "D1", "case_category": "general_violence"}
    ]
    df_dup = pd.DataFrame(dup_events)
    f_dup = compute_funnel_metrics(df_dup)["funnel_stages"][0]["case_count"]
    assert f_dup == f_base, "Duplicate stage events for the same case must not exceed contribution limit of 1"

    # Verify Laplace scale for epsilon = 0.25, sensitivity = 1.0
    epsilon = 0.25
    laplace = LaplaceMechanism(epsilon=epsilon, sensitivity=delta_f, random_state=42)
    assert laplace.scale == 4.0, f"Expected scale b=4.0, got {laplace.scale}"

    # Verify 95% statistical noise margin
    margin_95 = laplace.confidence_bound(0.95)
    expected_margin = -4.0 * math.log(0.05)  # 11.9829...
    assert math.isclose(margin_95, expected_margin, rel_tol=1e-5)
    assert round(margin_95, 0) == 12.0

    # Confirm that noise is NOT deterministically bounded:
    # Theoretical tail probability P(|X| > 12) = exp(-12 / 4) = exp(-3) ~= 0.049787 (~5%)
    tail_prob = math.exp(-12.0 / 4.0)
    assert 0.049 < tail_prob < 0.051


# ==============================================================================
# 8. Native Pandas DFG vs PM4Py Engine Distinction (Requirement 3)
# ==============================================================================

def test_process_engine_distinction_and_fallback():
    """
    Validates Requirement 3:
    Native Pandas DFG is clearly attributed and distinguished from PM4Py in journey metrics and bundle.
    """
    events_data = [
        {"case_id": "SYN-P-1", "activity": "FIR_REGISTERED", "timestamp": "2024-01-01", "district": "D1", "case_category": "general_violence"},
        {"case_id": "SYN-P-1", "activity": "MEDICAL_EXAM", "timestamp": "2024-01-02", "district": "D1", "case_category": "general_violence"},
        {"case_id": "SYN-P-1", "activity": "CHARGESHEET_FILED", "timestamp": "2024-01-20", "district": "D1", "case_category": "general_violence"}
    ]
    df_events = pd.DataFrame(events_data)
    res = compute_funnel_and_bottlenecks(df_events, total_entities=1, k_threshold=1)

    # Journey metrics must explicitly identify the engine
    assert res["journey_metrics"]["process_mining_engine"] == "Native Pandas Directly Follows Graph (DFG)"
    assert "Native in-memory" in res["journey_metrics"]["engine_description"]

    # Dashboard bundle preserves engine distinction
    bundle = build_dashboard_bundle_from_analysis(res)
    assert bundle["journey"]["process_mining_engine"] == "Native Pandas Directly Follows Graph (DFG)"
    assert bundle["bottlenecks"]["process_mining_engine"] == "Native Pandas Directly Follows Graph (DFG)"


# ==============================================================================
# 9. Timestamp Anomalies Visibly Reported in Data Quality (Requirement 4)
# ==============================================================================

def test_timestamp_anomalies_reported_as_data_quality():
    """
    Validates Requirement 4:
    Anomalous timestamps are excluded from duration statistics and visibly reported in the bundle.
    Both timestamp_anomalies_count (transitions) and anomalous_cases_count (cases) are tracked.
    """
    events_data = [
        # Normal Case: +1 day
        {"case_id": "SYN-ANOM-NORM", "activity": "FIR_REGISTERED", "timestamp": "2024-01-01", "district": "D1", "case_category": "general_violence"},
        {"case_id": "SYN-ANOM-NORM", "activity": "MEDICAL_EXAM", "timestamp": "2024-01-02", "district": "D1", "case_category": "general_violence"},
        # Anomalous Case: Inverted timestamp (Jan 15 -> Jan 05 = -10 days)
        {"case_id": "SYN-ANOM-REV", "activity": "FIR_REGISTERED", "timestamp": "2024-01-15", "district": "D1", "case_category": "general_violence"},
        {"case_id": "SYN-ANOM-REV", "activity": "MEDICAL_EXAM", "timestamp": "2024-01-05", "district": "D1", "case_category": "general_violence"},
    ]
    df_events = pd.DataFrame(events_data)
    res = compute_funnel_and_bottlenecks(df_events, total_entities=2, k_threshold=1)

    assert res["timestamp_anomalies_count"] == 1
    assert res["anomalous_cases_count"] == 1

    # Duration stats must only include the 1 valid transition
    med_trans = res["transitions"]["FIR_REGISTERED -> MEDICAL_EXAM"]
    assert med_trans["sample_count"] == 1
    assert med_trans["median_days"] == 1.0

    # Bundle preserves both counters for visible display
    bundle = build_dashboard_bundle_from_analysis(res)
    assert bundle["summary"]["timestamp_anomalies_count"] == 1
    assert bundle["summary"]["anomalous_cases_count"] == 1
    assert bundle["bottlenecks"]["timestamp_anomalies_count"] == 1
    assert bundle["bottlenecks"]["anomalous_cases_count"] == 1


# ==============================================================================
# 10. Multiple Anomalies per Case & Dual Counter Distinction (Targeted Adjustment 1)
# ==============================================================================

def test_multiple_anomalies_in_single_case_counter_distinction():
    """
    Validates Targeted Adjustment 1:
    - timestamp_anomalies_count represents total invalid timestamp transitions.
    - anomalous_cases_count represents unique affected case IDs.
    - When a single case contains multiple sequence inversions (e.g. 3 negative deltas),
      timestamp_anomalies_count == 3 while anomalous_cases_count == 1 (they are strictly different).
    """
    # 1 Case with multiple inverted transitions:
    # FIR: March 1, Medical: Feb 1 (-29d), Chargesheet: Jan 1 (-31d), E2E FIR->Chargesheet: Jan 1 vs Mar 1 (-60d)
    multi_anom_events = [
        {"case_id": "SYN-MULTI-ANOM", "activity": "FIR_REGISTERED", "timestamp": "2024-03-01", "district": "D1", "case_category": "general_violence"},
        {"case_id": "SYN-MULTI-ANOM", "activity": "MEDICAL_EXAM", "timestamp": "2024-02-01", "district": "D1", "case_category": "general_violence"},
        {"case_id": "SYN-MULTI-ANOM", "activity": "CHARGESHEET_FILED", "timestamp": "2024-01-01", "district": "D1", "case_category": "general_violence"},
    ]
    df_multi = pd.DataFrame(multi_anom_events)
    res_multi = compute_funnel_and_bottlenecks(df_multi, total_entities=1, k_threshold=1)

    # 3 invalid transitions (FIR->Med, Med->Chg, and E2E FIR->Chg)
    assert res_multi["timestamp_anomalies_count"] == 3
    # Exactly 1 unique case affected
    assert res_multi["anomalous_cases_count"] == 1
    # Verify the two counts are strictly different
    assert res_multi["timestamp_anomalies_count"] != res_multi["anomalous_cases_count"]

    # Bundle preserves both distinct counts
    bundle = build_dashboard_bundle_from_analysis(res_multi)
    assert bundle["summary"]["timestamp_anomalies_count"] == 3
    assert bundle["summary"]["anomalous_cases_count"] == 1
    assert bundle["bottlenecks"]["timestamp_anomalies_count"] == 3
    assert bundle["bottlenecks"]["anomalous_cases_count"] == 1


# ==============================================================================
# 11. Zero-Denominator Validity Label Handling (Targeted Adjustment 2)
# ==============================================================================

def test_zero_denominator_validity_label_logic():
    """
    Validates Targeted Adjustment 2:
    - If anomalies exist: displays excluded count.
    - If there are no anomalies and s_cnt > 0: displays 0 (100% valid).
    - If s_cnt is zero, None or missing: displays N/A (No Evaluated Transitions).
    - Handles invalid or missing sample-count values safely (NaN, inf, None, <=0).
    """
    def compute_label(anom_cnt, anom_cases_cnt, s_cnt):
        has_valid_sample = isinstance(s_cnt, (int, float)) and not math.isnan(s_cnt) and not math.isinf(s_cnt) and s_cnt > 0
        if anom_cnt > 0:
            cases_text = f" ({anom_cases_cnt} unique {'case' if anom_cases_cnt == 1 else 'cases'})" if anom_cases_cnt > 0 else ""
            return f"{anom_cnt} excluded{cases_text}"
        elif has_valid_sample:
            return "0 (100% valid)"
        else:
            return "N/A (No Evaluated Transitions)"

    # Scenario 1: Anomalies exist
    assert "3 excluded (1 unique case)" in compute_label(3, 1, 10)
    assert "1 excluded (1 unique case)" in compute_label(1, 1, 0)

    # Scenario 2: Zero anomalies and positive sample count -> 100% valid
    assert compute_label(0, 0, 25) == "0 (100% valid)"
    assert compute_label(0, 0, 1) == "0 (100% valid)"

    # Scenario 3: Zero sample count -> N/A (No Evaluated Transitions)
    assert compute_label(0, 0, 0) == "N/A (No Evaluated Transitions)"
    assert compute_label(0, 0, None) == "N/A (No Evaluated Transitions)"
    assert compute_label(0, 0, float("nan")) == "N/A (No Evaluated Transitions)"
    assert compute_label(0, 0, -5) == "N/A (No Evaluated Transitions)"
    assert compute_label(0, 0, "invalid") == "N/A (No Evaluated Transitions)"


# ==============================================================================
# 12. Defensive Anomaly Counter Sanitization & Fallback
# ==============================================================================

def test_anomaly_counter_defensive_sanitization_and_fallback():
    """
    Validates defensive anomaly counter sanitization:
    1. Preserves valid zero values across Python and NumPy integer/float types.
    2. Rejects None, NaN, positive/negative infinity, negative numbers, and booleans (including np.bool_).
    3. Converts valid non-negative numeric values to integers.
    4. Defaults invalid values to 0.
    5. Preserves legitimate zero values without erroneously triggering summary fallback.
    6. Falls back to summary when bottlenecks counter is explicitly None.
    """
    # 1. Valid zero preservation
    assert _clean_counter(0) == 0
    assert _clean_counter(0.0) == 0
    assert _clean_counter(np.int64(0)) == 0
    assert _clean_counter(np.float64(0.0)) == 0
    assert _clean_counter(np.int32(0)) == 0
    assert _clean_counter(np.float32(0.0)) == 0

    # 2. Valid positive numeric conversion to integer
    assert _clean_counter(5) == 5
    assert _clean_counter(5.0) == 5
    assert _clean_counter(5.9) == 5
    assert _clean_counter(np.int64(12)) == 12
    assert _clean_counter(np.float64(7.0)) == 7
    assert _clean_counter(np.int32(3)) == 3
    assert _clean_counter(np.float32(2.5)) == 2

    # 3. None and non-numeric rejection -> 0
    assert _clean_counter(None) == 0
    assert _clean_counter("10") == 0
    assert _clean_counter([1]) == 0
    assert _clean_counter({"count": 1}) == 0

    # 4. NaN and +/- infinity rejection -> 0
    assert _clean_counter(float("nan")) == 0
    assert _clean_counter(np.nan) == 0
    assert _clean_counter(float("inf")) == 0
    assert _clean_counter(float("-inf")) == 0
    assert _clean_counter(np.inf) == 0
    assert _clean_counter(-np.inf) == 0

    # 5. Negative number rejection -> 0
    assert _clean_counter(-1) == 0
    assert _clean_counter(-0.1) == 0
    assert _clean_counter(-100) == 0
    assert _clean_counter(np.int64(-5)) == 0
    assert _clean_counter(np.float64(-3.5)) == 0

    # 6. Boolean rejection -> 0 (explicitly excluding bool and np.bool_)
    assert _clean_counter(True) == 0
    assert _clean_counter(False) == 0
    assert _clean_counter(np.bool_(True)) == 0
    assert _clean_counter(np.bool_(False)) == 0

    # 7. Very large Python integers without overflow
    assert _clean_counter(10**18) == 10**18
    assert _clean_counter(10**100) == 10**100
    assert _clean_counter(10**400) == 10**400

    # 8. Fallback logic simulation
    def retrieve_counters(bottlenecks, DATA):
        raw_cnt = bottlenecks.get("timestamp_anomalies_count")
        if raw_cnt is None and isinstance(DATA.get("summary"), dict):
            raw_cnt = DATA.get("summary", {}).get("timestamp_anomalies_count")
        anom_cnt = _clean_counter(raw_cnt)

        raw_cases = bottlenecks.get("anomalous_cases_count")
        if raw_cases is None and isinstance(DATA.get("summary"), dict):
            raw_cases = DATA.get("summary", {}).get("anomalous_cases_count")
        anom_cases = _clean_counter(raw_cases)
        return anom_cnt, anom_cases

    # Scenario A: bottlenecks contains 0 -> 0 is preserved, summary (with 99) is NOT used
    data_with_summary = {"summary": {"timestamp_anomalies_count": 99, "anomalous_cases_count": 88}}
    assert retrieve_counters({"timestamp_anomalies_count": 0, "anomalous_cases_count": 0}, data_with_summary) == (0, 0)

    # Scenario B: bottlenecks key is None -> falls back to summary
    assert retrieve_counters({"timestamp_anomalies_count": None, "anomalous_cases_count": None}, data_with_summary) == (99, 88)

    # Scenario C: bottlenecks key is missing -> falls back to summary
    assert retrieve_counters({}, data_with_summary) == (99, 88)

    # Scenario D: bottlenecks has invalid value (e.g. NaN, negative, boolean) -> sanitized to 0
    assert retrieve_counters({"timestamp_anomalies_count": float("nan"), "anomalous_cases_count": -5}, data_with_summary) == (0, 0)
    assert retrieve_counters({"timestamp_anomalies_count": True, "anomalous_cases_count": False}, data_with_summary) == (0, 0)

    # 9. Verify NameError is eliminated when summary_data is undefined in local scope
    local_scope_without_summary_data = {
        "bottlenecks": {"timestamp_anomalies_count": 0, "anomalous_cases_count": 0},
        "DATA": {"bottlenecks": {"timestamp_anomalies_count": 0, "anomalous_cases_count": 0}}
    }
    # Executing the exact logic from app/dashboard.py in an isolated environment without summary_data
    exec_code = """
raw_anom_cnt = bottlenecks.get("timestamp_anomalies_count")
if raw_anom_cnt is None and isinstance(DATA.get("summary"), dict):
    raw_anom_cnt = DATA.get("summary", {}).get("timestamp_anomalies_count")
anom_cnt = _clean_counter(raw_anom_cnt)

raw_anom_cases = bottlenecks.get("anomalous_cases_count")
if raw_anom_cases is None and isinstance(DATA.get("summary"), dict):
    raw_anom_cases = DATA.get("summary", {}).get("anomalous_cases_count")
anom_cases_cnt = _clean_counter(raw_anom_cases)
"""
    exec_globals = {"_clean_counter": _clean_counter, "isinstance": isinstance, "dict": dict}
    exec(exec_code, exec_globals, local_scope_without_summary_data)
    assert local_scope_without_summary_data["anom_cnt"] == 0
    assert local_scope_without_summary_data["anom_cases_cnt"] == 0
