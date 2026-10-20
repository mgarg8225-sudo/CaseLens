"""
tests/test_case_simulator.py - Unit & Integration Tests for Part B Case Simulator.

Requirements Covered:
1. Each of the 8 ready-made scenarios executes cleanly and produces structured results.
2. Scenario 6 (two distinct individuals with similar names in same district) is NOT auto-merged (REJECT).
3. Scenario 7 flags an attrition drop-off after investigation without reaching chargesheet/trial.
4. Scenario 8 flags severe forensic turnaround delay (LP-01).
5. Zero side-effects: Running simulations never writes records into data/ or review_queue.csv.
6. Audit logging never records personal values (only scenario title/ID).
7. Custom edited scenario runs and produces valid scoring.
"""

import os
import json
import pytest
import pandas as pd

from src.case_simulator import (
    SCENARIOS,
    score_case_pair,
    reconstruct_simulated_journey,
    run_simulation
)


def test_all_eight_scenarios_execute():
    """Verifies that all 8 ready-made synthetic judicial scenarios execute without errors."""
    assert len(SCENARIOS) == 8, f"Expected 8 scenarios, found {len(SCENARIOS)}"

    for sc_key in SCENARIOS:
        result = run_simulation(sc_key)
        assert result is not None
        assert "title" in result
        assert "records" in result
        assert "pairwise_results" in result
        assert "journey" in result
        assert "verdict" in result
        assert "why" in result
        assert "summary" in result

        summary = result["summary"]
        assert summary["total_records"] >= 2
        assert summary["total_candidate_pairs"] >= 1
        assert len(result["pairwise_results"]) == summary["total_candidate_pairs"]


def test_scenario_6_false_link_protection():
    """
    Verifies that Scenario 6 (two different individuals with similar names in the same district)
    is strictly rejected and NOT auto-merged, validating false-positive protection.
    """
    res = run_simulation("scenario_6_different_people")
    summary = res["summary"]

    # Must have 0 auto-accepts
    assert summary["auto_accept_pairs"] == 0, "Scenario 6 candidate pair must not be auto-accepted!"
    assert summary["reject_pairs"] >= 1, "Scenario 6 pair must be definitively rejected!"

    pair = res["pairwise_results"][0]
    assert pair["decision"] == "REJECT", f"Expected REJECT decision, got '{pair['decision']}'"
    assert pair["match_probability"] < 0.50, f"Expected probability < 0.50, got {pair['match_probability']}"
    assert "False-Link Protection" in res["verdict"]


def test_scenario_7_flags_attrition_drop():
    """
    Verifies that Scenario 7 flags a systemic policy leak point (police-to-prosecution drop-off)
    where FIR and MLC exist but no chargesheet or court records exist.
    """
    res = run_simulation("scenario_7_dropped_case")
    journey = res["journey"]

    assert journey["is_dropped"] is True, "Scenario 7 must be flagged as dropped!"
    assert journey["completed_count"] == 3, f"Expected 3 completed stages, got {journey['completed_count']}"

    # Verify chargesheet and court stages are missing
    stages = {s["stage"]: s["status"] for s in journey["timeline_steps"]}
    assert "4. Chargesheet Filed" in stages
    assert stages["4. Chargesheet Filed"] == "DROPPED / MISSING"
    assert "5. Court Verdict" in stages
    assert stages["5. Court Verdict"] == "DROPPED / MISSING"

    # Verify leak point code is identified
    codes = [b["code"] for b in journey["bottlenecks"]]
    assert "LP-02-POLICE-ATTRITION" in codes
    assert "Dropped" in res["verdict"]


def test_scenario_8_flags_forensic_latency():
    """Verifies that Scenario 8 identifies severe forensic turnaround latency (LP-01)."""
    res = run_simulation("scenario_8_delayed_case")
    journey = res["journey"]

    codes = [b["code"] for b in journey["bottlenecks"]]
    assert "LP-01-FORENSIC-DELAY" in codes
    assert "Forensic Latency" in res["verdict"]


def test_simulator_zero_data_side_effects():
    """
    Verifies that executing simulator runs does NOT alter real files in data/,
    does NOT add rows to data/review_queue.csv, and does NOT log personal values.
    """
    rq_path = os.path.join("data", "review_queue.csv")
    rq_len_before = len(pd.read_csv(rq_path)) if os.path.exists(rq_path) else 0

    # Run all 8 scenarios
    for sc_key in SCENARIOS:
        run_simulation(sc_key)

    # Check review queue row count unchanged
    if os.path.exists(rq_path):
        rq_len_after = len(pd.read_csv(rq_path))
        assert rq_len_before == rq_len_after, "Simulator must never write to data/review_queue.csv"

    # Inspect app audit log: personal values (names, addresses) must never be logged
    audit_path = os.path.join("data", "app_audit_log.json")
    if os.path.exists(audit_path):
        with open(audit_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
        sim_logs = [l for l in logs if l.get("event_type") == "CASE_SIMULATOR_RUN"]
        for sl in sim_logs:
            details_str = json.dumps(sl.get("details", {})).lower()
            # Personal names from scenarios must not be in the audit details
            for personal_name in ["sunita sharma", "meena kumari", "rekha yadav", "ramesh sharma"]:
                assert personal_name not in details_str, f"Found personal value '{personal_name}' in simulator audit log!"


def test_custom_edited_scenario_execution():
    """Verifies that custom user-edited records run cleanly through the simulator."""
    custom_records = [
        {
            "agency": "Police FIR",
            "agency_code": "police_fir",
            "record_id": "FIR-CUSTOM-001",
            "name": "Simulated Survivor Alpha",
            "guardian": "Simulated Guardian Alpha",
            "district": "D5_Mayaganj",
            "locality": "Kalyan Ward",
            "incident_date": "2024-05-01",
            "record_date": "2024-05-02",
            "offense": "Sec 354 IPC"
        },
        {
            "agency": "Hospital MLC",
            "agency_code": "hospital_mlc",
            "record_id": "MLC-CUSTOM-001",
            "name": "Simulated Survivor Alpha",
            "guardian": "Simulated Guardian Alpha",
            "district": "D5_Mayaganj",
            "locality": "Kalyan Ward",
            "incident_date": "2024-05-01",
            "record_date": "2024-05-02",
            "details": "Exam conducted"
        }
    ]

    res = run_simulation(custom_records, custom_title="Custom Test Run")
    assert res is not None
    assert res["title"] == "Custom Test Run"
    assert res["summary"]["total_records"] == 2
    assert res["summary"]["auto_accept_pairs"] == 1
    assert res["verdict_type"] == "success"
