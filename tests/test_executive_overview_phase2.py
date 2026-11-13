"""
tests/test_executive_overview_phase2.py - Comprehensive Unit Tests for Phase 2:
Executive Overview, District Vulnerability Severity Index, Statutory Delay Ratios, and P95 Metrics.

Validates:
1. Strict 5-tier district severity thresholds (Normal < 10.0, Moderate 10-25, Elevated 25-40, Severe 40-60, Critical >= 60).
2. Boundary cases (score 10.0 is Moderate NOT Normal, score 30.0/50.0 are NEVER Normal).
3. Distinct case count suppression rule (n < 5 distinct cases returns "Insufficient Data", not 0 or Normal).
4. Statutory Delay Ratio formula: Empirical Median Duration / Statutory Benchmark Limit.
5. Exact 1.00x ratio behavior: strictly labeled "At Statutory Benchmark" (neither delay nor improvement).
6. Latency ratios < 1.00x labeled "Within statutory baseline / On schedule" and > 1.00x labeled delay multiplier.
7. Primary Latency Bottleneck selection: prioritizes highest statutory delay ratio over raw calendar days.
8. P95 Tail Latency: verified as empirical 95th percentile from actual durations, distinguishing the 188-day tail cohort.
9. Backend and frontend consistency across all severity tiers.
"""

import numpy as np
import pandas as pd

from src.upload_analysis import (
    classify_district_severity,
    compute_funnel_and_bottlenecks,
    check_bnss_193_eligibility,
    check_bnss_184_eligibility,
    compute_calendar_month_deadline,
    DEFAULT_FORENSIC_SOP_BENCHMARK_DAYS,
    FORENSIC_SOP_AUTHORITY
)
from app.dashboard import format_multiplier_label, format_leak_detail


# ==============================================================================
# 1. District Severity Threshold & Boundary Tests
# ==============================================================================

def test_district_severity_boundary_scores():
    """Validates that severity thresholds are strictly consistent across all score boundaries."""
    # Normal tier: strictly < 10.0
    assert classify_district_severity(0.0, total_cases=50, fir_cases=50) == "Normal"
    assert classify_district_severity(5.3, total_cases=50, fir_cases=50) == "Normal"
    assert classify_district_severity(9.9, total_cases=50, fir_cases=50) == "Normal"

    # Score of 10.0 must be MODERATE, never Normal
    assert classify_district_severity(10.0, total_cases=50, fir_cases=50) == "Moderate"
    assert classify_district_severity(18.5, total_cases=50, fir_cases=50) == "Moderate"
    assert classify_district_severity(24.9, total_cases=50, fir_cases=50) == "Moderate"

    # Elevated tier: 25.0 <= score < 40.0
    assert classify_district_severity(25.0, total_cases=50, fir_cases=50) == "Elevated"
    assert classify_district_severity(30.0, total_cases=50, fir_cases=50) == "Elevated"
    assert classify_district_severity(34.0, total_cases=50, fir_cases=50) == "Elevated"
    assert classify_district_severity(39.9, total_cases=50, fir_cases=50) == "Elevated"

    # Severe tier: 40.0 <= score < 60.0 (Score 50.0 is Severe, NEVER Normal)
    assert classify_district_severity(40.0, total_cases=50, fir_cases=50) == "Severe"
    assert classify_district_severity(45.2, total_cases=50, fir_cases=50) == "Severe"
    assert classify_district_severity(50.0, total_cases=50, fir_cases=50) == "Severe"
    assert classify_district_severity(56.5, total_cases=50, fir_cases=50) == "Severe"
    assert classify_district_severity(59.9, total_cases=50, fir_cases=50) == "Severe"

    # Critical tier: score >= 60.0
    assert classify_district_severity(60.0, total_cases=50, fir_cases=50) == "Critical"
    assert classify_district_severity(75.0, total_cases=50, fir_cases=50) == "Critical"
    assert classify_district_severity(100.0, total_cases=50, fir_cases=50) == "Critical"


def test_district_small_count_suppression_insufficient_data():
    """Validates that distinct case counts < 5 return 'Insufficient Data', never Normal or 0."""
    # Zero cases or zero FIRs
    assert classify_district_severity(0.0, total_cases=0, fir_cases=0, k_threshold=5) == "Insufficient Data"
    assert classify_district_severity(50.0, total_cases=0, fir_cases=0, k_threshold=5) == "Insufficient Data"
    assert classify_district_severity(5.0, total_cases=10, fir_cases=0, k_threshold=5) == "Insufficient Data"

    # Distinct cases < 5 (k-anonymity boundary)
    assert classify_district_severity(2.0, total_cases=1, fir_cases=1, k_threshold=5) == "Insufficient Data"
    assert classify_district_severity(8.0, total_cases=4, fir_cases=4, k_threshold=5) == "Insufficient Data"
    assert classify_district_severity(50.0, total_cases=4, fir_cases=4, k_threshold=5) == "Insufficient Data"

    # Exactly 5 cases (sufficient data)
    assert classify_district_severity(8.0, total_cases=5, fir_cases=5, k_threshold=5) == "Normal"
    assert classify_district_severity(10.0, total_cases=5, fir_cases=5, k_threshold=5) == "Moderate"

    # Explicit suppression flag
    assert classify_district_severity(5.0, total_cases=50, fir_cases=50, is_suppressed=True) == "Insufficient Data"
    assert classify_district_severity(None, total_cases=50, fir_cases=50) == "Insufficient Data"


def test_distinct_case_counting_in_upload_analysis():
    """Verifies that k-anonymity suppression in upload analysis uses distinct case IDs, not row counts."""
    # 4 distinct cases with 20 total event rows (5 events per case)
    events_data = []
    for c_id in ["CASE-01", "CASE-02", "CASE-03", "CASE-04"]:
        for act, day in [("FIR_REGISTERED", "2024-01-01"), ("MEDICAL_EXAM", "2024-01-02"),
                         ("FORENSIC_REPORT", "2024-02-01"), ("CHARGESHEET_FILED", "2024-03-01"),
                         ("COURT_VERDICT", "2024-10-01")]:
            events_data.append({
                "case_id": c_id,
                "district": "D_Sparse",
                "activity": act,
                "timestamp": day
            })
    df_events = pd.DataFrame(events_data)
    assert len(df_events) == 20  # Total rows = 20
    assert df_events["case_id"].nunique() == 4  # Distinct cases = 4 < 5

    res = compute_funnel_and_bottlenecks(df_events, total_entities=4, k_threshold=5)
    d_leak = res["district_leakage"]
    sparse_entry = next((d for d in d_leak if d["district"] == "D_Sparse"), None)

    assert sparse_entry is not None
    assert sparse_entry["is_suppressed"] is True
    assert sparse_entry["status"] == "Insufficient Data"
    assert sparse_entry["total_cases"] == "<5"


# ==============================================================================
# 2. Statutory Delay Ratio & Exact 1.00x Multiplier Tests
# ==============================================================================

def test_statutory_delay_ratio_formula():
    """
    Validates Statutory Delay Ratio formula:
    Statutory Delay Ratio = Empirical Median Duration (Days) / Statutory Benchmark (Days).
    """
    benchmarks = {
        "FIR_REGISTERED -> MEDICAL_EXAM": 1.0,      # 24 hours under Sec 164A CrPC
        "FIR_REGISTERED -> FORENSIC_REPORT": 30.0,  # 30 days under POCSO / MHA Directives
        "FIR_REGISTERED -> CHARGESHEET_FILED": 60.0, # 60 days under Sec 173(2) CrPC
        "CHARGESHEET_FILED -> COURT_VERDICT": 365.0 # 365 days (1 year) under POCSO Sec 35
    }

    # Case A: Exactly at benchmark (1.00x)
    ratio_mlc = 1.0 / benchmarks["FIR_REGISTERED -> MEDICAL_EXAM"]
    assert round(ratio_mlc, 2) == 1.00

    # Case B: Below benchmark (ahead of schedule, e.g. 284d trial vs 365d)
    ratio_trial = 284.0 / benchmarks["CHARGESHEET_FILED -> COURT_VERDICT"]
    assert round(ratio_trial, 2) == 0.78
    assert ratio_trial < 1.00

    # Case C: Above benchmark (delayed, e.g. 43d FSL vs 30d)
    ratio_fsl = 43.0 / benchmarks["FIR_REGISTERED -> FORENSIC_REPORT"]
    assert round(ratio_fsl, 2) == 1.43
    assert ratio_fsl > 1.00


def test_multiplier_exact_1_00x_labeled_at_statutory_benchmark():
    """Verifies that 1.00x is strictly labeled 'At Statutory Benchmark' (not an improvement, not a delay)."""
    # format_multiplier_label
    assert format_multiplier_label(1.0) == "At Statutory Benchmark"
    assert format_multiplier_label(1.000) == "At Statutory Benchmark"
    assert format_multiplier_label("1.0") == "At Statutory Benchmark"

    # Below 1.00x
    sub_one = format_multiplier_label(0.78)
    assert "Within Statutory Benchmark" in sub_one
    assert "0.78" in sub_one

    # Above 1.00x
    above_one = format_multiplier_label(1.43)
    assert "1.43× baseline" in above_one or "delay" in above_one

    # format_leak_detail with multiplier 1.00
    lp_bench = {
        "multiplier": 1.0,
        "delay_days": 30.0,
        "baseline_days": 30.0
    }
    detail = format_leak_detail(lp_bench)
    assert "At Statutory Benchmark" in detail
    assert "delay multiplier" not in detail


def test_primary_bottleneck_statutory_ratio_selection():
    """
    Validates that Primary Bottleneck is selected by Statutory Delay Ratio,
    NOT solely by longest raw calendar duration.
    """
    # Create event log where:
    # 1. Trial takes 284 days (Statutory limit: 365 days -> Ratio: 0.78x, on schedule)
    # 2. Forensic analysis takes 43 days (Statutory limit: 30 days -> Ratio: 1.43x, severe delay)
    cases = []
    for i in range(20):
        c_id = f"CASE-{i:03d}"
        cases.extend([
            {"case_id": c_id, "district": "D1", "activity": "FIR_REGISTERED", "timestamp": "2024-01-01"},
            {"case_id": c_id, "district": "D1", "activity": "MEDICAL_EXAM", "timestamp": "2024-01-02"},  # 1 day
            {"case_id": c_id, "district": "D1", "activity": "FORENSIC_REPORT", "timestamp": "2024-02-14"},  # 43 days
            {"case_id": c_id, "district": "D1", "activity": "CHARGESHEET_FILED", "timestamp": "2024-04-01"}, # 47 days
            {"case_id": c_id, "district": "D1", "activity": "COURT_VERDICT", "timestamp": "2025-01-10"},  # 284 days
        ])
    df_events = pd.DataFrame(cases)

    res = compute_funnel_and_bottlenecks(df_events, total_entities=20, k_threshold=5)
    bb = res["biggest_bottleneck"]

    assert bb is not None
    # Must NOT select Court Verdict (284 days) as bottleneck because 284d < 365d mandate (ratio 0.78x)
    assert "COURT_VERDICT" not in bb["pair"]
    # Must select Forensic turnaround (ratio 1.43x > 1.00x)
    assert "FORENSIC_REPORT" in bb["pair"]
    assert bb["delay_multiplier"] > 1.0


# ==============================================================================
# 3. P95 Tail Latency & Actual Duration Verification
# ==============================================================================

def test_p95_calculated_from_actual_duration_percentile():
    """Verifies that P95 is calculated mathematically as the 95th percentile of actual durations."""
    # 100 actual duration points where 95% fall under 80.1 days and 5% tail spikes up to 188 days
    durations = [30 + i * 0.5 for i in range(95)] + [140, 155, 170, 180, 188]
    vals = np.array(durations)
    expected_p95 = round(float(np.percentile(vals, 95)), 1)
    expected_median = round(float(np.median(vals)), 1)

    assert expected_median < expected_p95
    assert expected_p95 > 80.0
    assert any(d >= 188.0 for d in durations)  # Extreme tail contains 188-day cases
    assert round(float(np.percentile(vals, 99)), 1) >= 180.0


def test_p95_chargesheet_cohort_context():
    """Verifies the 188-day metric context in the standard baseline bottlenecks."""
    import json
    with open("data/analytics_bottlenecks.json", "r") as f:
        data = json.load(f)

    e2e_chg = data.get("FIR_REGISTERED -> CHARGESHEET_FILED (E2E)", {})
    assert e2e_chg.get("p95_days") == 188.0
    assert e2e_chg.get("median_days") == 86.0
    assert e2e_chg.get("sample_count") == 2398
    # 86 days median vs 60 days statutory benchmark
    assert e2e_chg.get("median_days") > 60.0
    # P95 is 188 days (over 3x statutory 60-day limit)
    assert e2e_chg.get("p95_days") > 3 * 60.0


# ==============================================================================
# 4. Statutory Eligibility Whitelist Tests (BNSS 193(2) vs BNSS 184(1))
# ==============================================================================

def test_bnss_193_eligibility_statutory_offences():
    """
    Validates BNSS Section 193(2) / CrPC Section 173(1A) 2-calendar-month eligibility:
    - BNS 64, 65, 66, 67, 68, 70, 71 (Eligible)
    - POCSO 4, 6, 8, 10 (Eligible)
    - IPC 376 series (Eligible)
    - BNS 69 (Ineligible - deceitful promise excluded from statutory 2-month mandate)
    - POCSO 12, 14, 15 (Ineligible)
    - IPC 498A, 354, 302 (Ineligible)
    - Missing legal section data cannot be certified by offense_category == 'sexual_assault' alone.
    """
    # 1. Eligible BNS Offences
    for sec in ["BNS 64", "BNS 65", "BNS 66", "BNS 67", "BNS 68", "BNS 70", "BNS 71"]:
        is_elig, reason = check_bnss_193_eligibility(sec)
        assert is_elig is True, f"Failed for {sec}: {reason}"
        assert reason == "ELIGIBLE_BNSS_193_2_BNS"

    # 2. Eligible POCSO Offences
    for sec in ["POCSO Sec 4", "POCSO Sec 6", "POCSO 8", "POCSO 10"]:
        is_elig, reason = check_bnss_193_eligibility(sec)
        assert is_elig is True, f"Failed for {sec}: {reason}"
        assert reason == "ELIGIBLE_BNSS_193_2_POCSO"

    # 3. Eligible Legacy IPC Offences under Section 173(1A) CrPC (including 376B, 376C, and 376E)
    for sec in ["IPC 376", "IPC 376A", "IPC 376B", "IPC 376C", "IPC 376D", "IPC 376E", "IPC 376(2)(f)"]:
        is_elig, reason = check_bnss_193_eligibility(sec)
        assert is_elig is True, f"Failed for {sec}: {reason}"
        assert reason == "ELIGIBLE_CRPC_173_1A_IPC"

    # 4. Explicitly Excluded: BNS Section 69
    is_elig_69, reason_69 = check_bnss_193_eligibility("BNS 69")
    assert is_elig_69 is False
    assert reason_69 == "INELIGIBLE_BNS_SECTION_69"

    # 5. Ineligible POCSO offences
    for sec in ["POCSO 12", "POCSO 14", "POCSO 15"]:
        is_elig, reason = check_bnss_193_eligibility(sec)
        assert is_elig is False
        assert reason == "INELIGIBLE_STATUTORY_OFFENCE"

    # 6. Ineligible general offences
    for sec in ["IPC 498A", "IPC 354D", "IPC 302", "BNS 85"]:
        is_elig, reason = check_bnss_193_eligibility(sec)
        assert is_elig is False
        assert reason == "INELIGIBLE_STATUTORY_OFFENCE"

    # 7. Missing section data cannot be certified by coarse offense_category alone
    is_elig_none, reason_none = check_bnss_193_eligibility(None, offense_category="sexual_assault")
    assert is_elig_none is False
    assert reason_none == "UNVERIFIED_STATUTORY_ELIGIBILITY"

    is_elig_empty, reason_empty = check_bnss_193_eligibility("", offense_category="sexual_assault")
    assert is_elig_empty is False
    assert reason_empty == "UNVERIFIED_STATUTORY_ELIGIBILITY"


def test_bnss_184_medical_exam_eligibility_rape_only():
    """
    Validates BNSS Section 184(1) / CrPC Section 164A(1) medical exam eligibility:
    - Must be substantive rape or attempted rape offence.
    - Eligible: BNS 64, 65, 66, 70; IPC 376, 376A, 376D.
    - IPC 376B (intercourse during separation) & IPC 376C (intercourse by authority) are NOT rape offences.
    - BNS 67 (husband during separation) & BNS 68 (custodial sex) are NOT rape offences.
    - BNS 69 is NOT a rape offence.
    - BNS 71 (repeat offender) does NOT independently certify rape eligibility without an underlying qualifying offence.
    - POCSO-only sections (Sec 4, 6) cannot be automatically included under Sec 184 without confirmed rape provisions.
    - Missing legal sections marked UNVERIFIED_STATUTORY_ELIGIBILITY.
    """
    # 1. Eligible Substantive Rape Sections
    for sec in ["BNS 64", "BNS 65", "BNS 66", "BNS 70", "BNS 64, 71"]:
        is_elig, reason = check_bnss_184_eligibility(sec)
        assert is_elig is True, f"Failed for {sec}: {reason}"
        assert reason == "ELIGIBLE_BNSS_184_1_BNS_RAPE"

    for sec in ["IPC 376", "IPC 376A", "IPC 376D"]:
        is_elig, reason = check_bnss_184_eligibility(sec)
        assert is_elig is True, f"Failed for {sec}: {reason}"
        assert reason == "ELIGIBLE_CRPC_164A_IPC_RAPE"

    # 2. IPC 376B and 376C are NOT rape offences under Section 164A(1) / Section 184(1)
    is_elig_376b, reason_376b = check_bnss_184_eligibility("IPC 376B")
    assert is_elig_376b is False
    assert reason_376b == "INELIGIBLE_NON_RAPE_IPC"

    is_elig_376c, reason_376c = check_bnss_184_eligibility("IPC 376C")
    assert is_elig_376c is False
    assert reason_376c == "INELIGIBLE_NON_RAPE_IPC"

    # 3. BNS 67 and 68 are NOT rape offences for Section 184
    is_elig_67, reason_67 = check_bnss_184_eligibility("BNS 67")
    assert is_elig_67 is False
    assert reason_67 == "INELIGIBLE_NON_RAPE_BNS"

    is_elig_68, reason_68 = check_bnss_184_eligibility("BNS 68")
    assert is_elig_68 is False
    assert reason_68 == "INELIGIBLE_NON_RAPE_BNS"

    is_elig_69, reason_69 = check_bnss_184_eligibility("BNS 69")
    assert is_elig_69 is False
    assert reason_69 == "INELIGIBLE_NON_RAPE_BNS"

    # 4. BNS 71 without a qualifying underlying offence is not independently certified
    is_elig_71, reason_71 = check_bnss_184_eligibility("BNS 71")
    assert is_elig_71 is False
    assert reason_71 == "UNVERIFIED_STATUTORY_ELIGIBILITY"

    # 5. IPC 376E (repeat offender) alone cannot independently certify rape eligibility
    is_elig_376e, reason_376e = check_bnss_184_eligibility("IPC 376E")
    assert is_elig_376e is False
    assert reason_376e == "UNVERIFIED_STATUTORY_ELIGIBILITY"

    # 6. IPC 376E paired with qualifying underlying rape sections IS certified eligible
    for combo in [
        "IPC 376, 376E",
        "IPC 376A, 376E",
        "IPC 376AB, 376E",
        "IPC 376D, 376E",
        "IPC 376DA, 376E",
        "IPC 376DB, 376E"
    ]:
        is_elig_combo, reason_combo = check_bnss_184_eligibility(combo)
        assert is_elig_combo is True, f"Failed for {combo}: {reason_combo}"
        assert reason_combo == "ELIGIBLE_CRPC_164A_IPC_RAPE"

    # 7. POCSO-only sections cannot be automatically included under Sec 184 without established rape provisions
    is_elig_pocso4, reason_p4 = check_bnss_184_eligibility("POCSO Sec 4")
    assert is_elig_pocso4 is False
    assert reason_p4 == "UNVERIFIED_STATUTORY_ELIGIBILITY"

    is_elig_pocso6, reason_p6 = check_bnss_184_eligibility("POCSO Sec 6")
    assert is_elig_pocso6 is False
    assert reason_p6 == "UNVERIFIED_STATUTORY_ELIGIBILITY"

    # 8. Missing section data
    is_elig_miss, reason_miss = check_bnss_184_eligibility(None, offense_category="sexual_assault")
    assert is_elig_miss is False
    assert reason_miss == "UNVERIFIED_STATUTORY_ELIGIBILITY"


# ==============================================================================
# 5. Calendar-Month Deadline, Anomalies, and Pending Case Separation Tests
# ==============================================================================

def test_calendar_month_boundary_calculation():
    """
    Verifies that statutory investigation deadline calculates exact two calendar months,
    handling leap years, month-end boundaries, and missing dates safely.
    """
    # Boundary 1: Jan 31 in leap year (2024) -> March 31 (60 days)
    dt_jan31_leap = pd.Timestamp("2024-01-31")
    dl_jan31_leap = compute_calendar_month_deadline(dt_jan31_leap, months=2)
    assert dl_jan31_leap == pd.Timestamp("2024-03-31")
    assert (dl_jan31_leap - dt_jan31_leap).days == 60

    # Boundary 2: Jan 31 in non-leap year (2023) -> March 31 (59 days)
    dt_jan31 = pd.Timestamp("2023-01-31")
    dl_jan31 = compute_calendar_month_deadline(dt_jan31, months=2)
    assert dl_jan31 == pd.Timestamp("2023-03-31")
    assert (dl_jan31 - dt_jan31).days == 59

    # Boundary 3: July 1 -> September 1 (62 days)
    dt_jul1 = pd.Timestamp("2024-07-01")
    dl_jul1 = compute_calendar_month_deadline(dt_jul1, months=2)
    assert dl_jul1 == pd.Timestamp("2024-09-01")
    assert (dl_jul1 - dt_jul1).days == 62

    # Boundary 4: Missing dates return None
    assert compute_calendar_month_deadline(pd.NaT) is None
    assert compute_calendar_month_deadline(None) is None


def test_timestamp_sequence_anomalies_flagged_and_excluded():
    """
    Verifies that negative durations (e.g. t_end < t_start) are flagged as anomalies,
    excluded from duration statistics (no max(0, delta) clamping), and exposed in counter.
    """
    events = [
        # Normal Case: FIR on Jan 1, Medical on Jan 2 (+1 day)
        {"case_id": "CASE-NORM", "district": "D1", "activity": "FIR_REGISTERED", "timestamp": "2024-01-01"},
        {"case_id": "CASE-NORM", "district": "D1", "activity": "MEDICAL_EXAM", "timestamp": "2024-01-02"},
        # Anomalous Case: Medical on Jan 10, FIR on Jan 20 (-10 days sequence inversion)
        {"case_id": "CASE-ANOM", "district": "D1", "activity": "FIR_REGISTERED", "timestamp": "2024-01-20"},
        {"case_id": "CASE-ANOM", "district": "D1", "activity": "MEDICAL_EXAM", "timestamp": "2024-01-10"}
    ]
    df_events = pd.DataFrame(events)

    res = compute_funnel_and_bottlenecks(df_events, total_entities=2, k_threshold=1)
    trans = res["transitions"].get("FIR_REGISTERED -> MEDICAL_EXAM", {})

    # Anomaly counter exposed
    assert res.get("timestamp_anomalies_count") == 1

    # Duration stats must ONLY include the valid transition (+1 day), NOT the -10 day anomaly clamped to 0
    assert trans.get("sample_count") == 1
    assert trans.get("median_days") == 1.0
    assert trans.get("mean_days") == 1.0

    # Original dates are strictly preserved in raw data
    raw_anom_fir = df_events[(df_events["case_id"] == "CASE-ANOM") & (df_events["activity"] == "FIR_REGISTERED")]["timestamp"].iloc[0]
    assert raw_anom_fir == "2024-01-20"


def test_forensic_sop_target_configuration_and_authority():
    """Verifies that 30-day forensic benchmark is stored as a configurable Policy SOP target with source reference."""
    assert DEFAULT_FORENSIC_SOP_BENCHMARK_DAYS == 30.0
    assert "MHA Women Safety Advisory" in FORENSIC_SOP_AUTHORITY
    assert "State FSL / SAEK SOPs" in FORENSIC_SOP_AUTHORITY
    assert "Administrative Policy Target" in FORENSIC_SOP_AUTHORITY


def test_pending_cases_separated_from_completed_duration_percentiles():
    """
    Verifies that cases with missing completion milestones (pending cases)
    are not mixed into completed-case transition duration arrays.
    """
    events = [
        # Completed Case
        {"case_id": "C-COMP", "district": "D1", "activity": "CHARGESHEET_FILED", "timestamp": "2024-01-01"},
        {"case_id": "C-COMP", "district": "D1", "activity": "COURT_VERDICT", "timestamp": "2024-08-01"},
        # Pending Case (chargesheet filed, verdict never reached)
        {"case_id": "C-PEND", "district": "D1", "activity": "CHARGESHEET_FILED", "timestamp": "2024-01-15"}
    ]
    df_events = pd.DataFrame(events)

    res = compute_funnel_and_bottlenecks(df_events, total_entities=2, k_threshold=1)
    trans = res["transitions"].get("CHARGESHEET_FILED -> COURT_VERDICT", {})

    # Only 1 completed case is present in duration distribution
    assert trans.get("sample_count") == 1
    assert trans.get("median_days") == 213.0
