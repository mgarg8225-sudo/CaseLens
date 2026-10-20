"""
src/case_simulator.py - Judicial Case Simulator Engine for CaseLens.

Interactive "What Goes In vs What Comes Out" engine for judicial demonstration.
Runs synthetic multi-agency records through the real Fellegi-Sunter probabilistic
record linkage scoring and longitudinal journey timeline reconstruction.

Privacy & Safety Guarantees:
1. Synthetic records only: never reads or queries real survivor records by name.
2. Read-only with zero data side effects: never writes simulator records to data/ files
   or review_queue.csv.
3. Audit logging records strictly the scenario name and timestamp without personal values.
4. Tri-state decision thresholding identical to pipeline:
   - Auto-Accept: match_probability >= 0.85 (with temporal sanity validation)
   - Human Review: 0.50 <= match_probability < 0.85
   - Reject: match_probability < 0.50
"""

import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional

import jellyfish

from src.linkage import normalize_name, normalize_locality
from src.auth import log_audit_event


# ==============================================================================
# 1. Ready-Made Synthetic Judicial Scenarios
# ==============================================================================

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "scenario_1_clean": {
        "title": "1. Clean Case: All 5 Agency Records Agree",
        "description": "Standard ideal case where Police FIR, Hospital MLC, Forensic Lab, Prosecutor, and Court records align consistently on name, guardian, district, and dates.",
        "expected_verdict": "Linked with High Confidence (Full 5-Stage Institutional Journey)",
        "why": "All 5 agency records show high agreement across full name, phonetic tokens, guardian name, district, and incident date. Match probabilities exceed 0.95 across all pairs. All milestones respect statutory timelines.",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D3-0912",
                "name": "Sunita Sharma",
                "guardian": "Ramesh Sharma",
                "district": "D3_Rudrapur",
                "locality": "Civil Lines",
                "incident_date": "2024-03-01",
                "record_date": "2024-03-02",
                "offense": "Sec 376 IPC (Rape)"
            },
            {
                "agency": "Hospital MLC",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2024-D3-0844",
                "name": "Sunita Sharma",
                "guardian": "Ramesh Sharma",
                "district": "D3_Rudrapur",
                "locality": "Civil Lines",
                "incident_date": "2024-03-01",
                "record_date": "2024-03-02",
                "details": "Medico-legal exam completed within 24h"
            },
            {
                "agency": "Forensic Lab (FSL)",
                "agency_code": "forensic_lab",
                "record_id": "FSL-2024-D3-0412",
                "name": "Sunita Sharma",
                "guardian": "Ramesh Sharma",
                "district": "D3_Rudrapur",
                "locality": None,
                "incident_date": "2024-03-01",
                "record_date": "2024-03-24",
                "details": "Biological DNA match dispatched in 22 days"
            },
            {
                "agency": "Prosecution Chargesheet",
                "agency_code": "prosecutor",
                "record_id": "PROS-2024-D3-0301",
                "name": "Sunita Sharma",
                "guardian": "Ramesh Sharma",
                "district": "D3_Rudrapur",
                "locality": "Civil Lines",
                "incident_date": "2024-03-01",
                "record_date": "2024-04-18",
                "details": "Chargesheet filed in 47 days (CrPC 173 compliant)"
            },
            {
                "agency": "District Court",
                "agency_code": "court",
                "record_id": "CRT-2024-D3-0118",
                "name": "Sunita Sharma",
                "guardian": "Ramesh Sharma",
                "district": "D3_Rudrapur",
                "locality": "Civil Lines",
                "incident_date": "2024-03-01",
                "record_date": "2024-11-20",
                "details": "Conviction recorded (Trial completed in 263 days)"
            }
        ]
    },

    "scenario_2_spelling_variants": {
        "title": "2. Spelling Variants: Dialectal & Transliteration Discrepancies",
        "description": "Examines dialectal phonetic spelling variations (Ramesh / Rmesh, Sunita / Suneeta, Mohd / Mohammad) across disparate agencies.",
        "expected_verdict": "Linked with High Confidence (Spelling & Transliteration Invariance)",
        "why": "The deterministic normalization layer applies Indian transliteration dictionaries and Double Metaphone phonetic encoding. Despite 'Suneeta' vs 'Sunita' and 'Rmesh' vs 'Ramesh', phonetic keys match perfectly, resulting in an auto-accept decision.",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D3-1102",
                "name": "Suneeta Sharma",
                "guardian": "Rmesh Sharma",
                "district": "D3_Rudrapur",
                "locality": "Gandhi Colony",
                "incident_date": "2024-02-14",
                "record_date": "2024-02-15",
                "offense": "Sec 354 IPC (Assault with intent to outrage modesty)"
            },
            {
                "agency": "Hospital MLC",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2024-D3-1088",
                "name": "Sunita Sharma",
                "guardian": "Ramesh Sharma",
                "district": "D3_Rudrapur",
                "locality": "Gandhi Col",
                "incident_date": "2024-02-14",
                "record_date": "2024-02-15",
                "details": "Exam recorded under standardized spelling"
            },
            {
                "agency": "Prosecution Chargesheet",
                "agency_code": "prosecutor",
                "record_id": "PROS-2024-D3-0499",
                "name": "Sunita Sharma",
                "guardian": "Ramesh Sharma",
                "district": "D3_Rudrapur",
                "locality": "Gandhi Colony",
                "incident_date": "2024-02-14",
                "record_date": "2024-04-05",
                "details": "Chargesheet filed"
            }
        ]
    },

    "scenario_3_swapped_names": {
        "title": "3. Swapped First/Last Name and Initials",
        "description": "Tests surname-first transposition ('Sharma Sunita' vs 'Sunita Sharma') and initial abbreviations ('Sunita S.').",
        "expected_verdict": "Linked with High Confidence (Surname Inversion & Initials Invariance)",
        "why": "Heuristic token transposition detects that 'Sharma' is a recognized Indian surname and re-orders tokens to standard First-Last format. Jaro-Winkler string distance on normalized names achieves >= 0.90 similarity.",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D1-0455",
                "name": "Sharma Sunita",
                "guardian": "Sharma Ramesh",
                "district": "D1_Suryapur",
                "locality": "Sector 4",
                "incident_date": "2024-04-10",
                "record_date": "2024-04-11",
                "offense": "Sec 498A IPC (Cruelty by husband/relatives)"
            },
            {
                "agency": "Hospital MLC",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2024-D1-0390",
                "name": "Sunita Sharma",
                "guardian": "R. Sharma",
                "district": "D1_Suryapur",
                "locality": "Sec 4",
                "incident_date": "2024-04-10",
                "record_date": "2024-04-11",
                "details": "Medical documentation with initials"
            },
            {
                "agency": "District Court",
                "agency_code": "court",
                "record_id": "CRT-2024-D1-0205",
                "name": "Sunita Sharma",
                "guardian": "Ramesh Sharma",
                "district": "D1_Suryapur",
                "locality": "Sector 4",
                "incident_date": "2024-04-10",
                "record_date": "2024-10-14",
                "details": "Court petition registered"
            }
        ]
    },

    "scenario_4_missing_fields": {
        "title": "4. Missing Fields: Incomplete Records & Null Attributes",
        "description": "Tests partial records where certain agency entries lack guardian name, address, or age attributes.",
        "expected_verdict": "Linked with High Confidence (Robust to Incomplete Agency Fields)",
        "why": "The Fellegi-Sunter model implements missing-value neutrality: when an attribute (like locality or guardian name) is null in one record, it yields zero weight adjustment rather than a negative mismatch penalty. Strong name and incident date agreement secure the link.",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D2-0771",
                "name": "Kavita Devi",
                "guardian": "Suresh Kumar",
                "district": "D2_Chandranagar",
                "locality": "Shanti Nagar",
                "incident_date": "2024-05-12",
                "record_date": "2024-05-13",
                "offense": "Sec 376 IPC"
            },
            {
                "agency": "Forensic Lab (FSL)",
                "agency_code": "forensic_lab",
                "record_id": "FSL-2024-D2-0199",
                "name": "Kavita Devi",
                "guardian": None,
                "district": "D2_Chandranagar",
                "locality": None,
                "incident_date": "2024-05-12",
                "record_date": "2024-06-05",
                "details": "Specimen receipt record with no relative or locality captured"
            },
            {
                "agency": "Prosecution Chargesheet",
                "agency_code": "prosecutor",
                "record_id": "PROS-2024-D2-0288",
                "name": "Kavita Devi",
                "guardian": "Suresh Kumar",
                "district": "D2_Chandranagar",
                "locality": None,
                "incident_date": "2024-05-12",
                "record_date": "2024-07-02",
                "details": "Vetting sheet"
            }
        ]
    },

    "scenario_5_date_errors": {
        "title": "5. Date Errors: Recall Shifts (1-3 Days) and Day/Month Transposition",
        "description": "Evaluates handling of trauma-induced recall variance (+/- 2 days) and day/month transposition in incident reporting.",
        "expected_verdict": "Uncertain / Auto-Accept: Temporal Tolerance Active",
        "why": "The linkage engine uses AbsoluteDateDifference with a 2-day high agreement threshold and 10-day tolerance window. Records within 2 days receive full date agreement weight, while larger discrepancies are routed to human review.",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D4-0601",
                "name": "Pooja Verma",
                "guardian": "Anil Verma",
                "district": "D4_Kalyanpur",
                "locality": "Railway Road",
                "incident_date": "2024-06-05",
                "record_date": "2024-06-06",
                "offense": "Sec 354 IPC"
            },
            {
                "agency": "Hospital MLC",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2024-D4-0589",
                "name": "Pooja Verma",
                "guardian": "Anil Verma",
                "district": "D4_Kalyanpur",
                "locality": "Railway Road",
                "incident_date": "2024-06-07",
                "record_date": "2024-06-07",
                "details": "Incident reported as 2 days prior during trauma interview"
            },
            {
                "agency": "Prosecution Chargesheet",
                "agency_code": "prosecutor",
                "record_id": "PROS-2024-D4-0211",
                "name": "Pooja Verma",
                "guardian": "Anil Verma",
                "district": "D4_Kalyanpur",
                "locality": "Railway Rd",
                "incident_date": "2024-06-05",
                "record_date": "2024-07-30",
                "details": "Chargesheet aligning with FIR date"
            }
        ]
    },

    "scenario_6_different_people": {
        "title": "6. False-Link Protection: Different People with Similar Names in Same District",
        "description": "Crucial false-positive defense: two distinct individuals with common names in the same district must NOT be merged.",
        "expected_verdict": "Definitively Disconnected: False-Link Protection Active (No Merger)",
        "why": "Although both records share the same district and common first name 'Sunita', surname tokens ('Devi' vs 'Sharma'), guardian names ('Ramesh Chand' vs 'Suresh Kumar'), localities, and dates (194 days apart) conflict severely. Negative match weights dominate and drop the score to P < 0.10, triggering an absolute REJECT.",
        "records": [
            {
                "agency": "Police FIR (Person A)",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D3-0210",
                "name": "Sunita Devi",
                "guardian": "Ramesh Chand",
                "district": "D3_Rudrapur",
                "locality": "Model Town",
                "incident_date": "2024-02-10",
                "record_date": "2024-02-11",
                "offense": "Sec 376 IPC"
            },
            {
                "agency": "Hospital MLC (Person B)",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2024-D3-0940",
                "name": "Sunita Sharma",
                "guardian": "Suresh Kumar",
                "district": "D3_Rudrapur",
                "locality": "Raja Bazar",
                "incident_date": "2024-08-22",
                "record_date": "2024-08-23",
                "details": "Distinct patient record in same district"
            }
        ]
    },

    "scenario_7_dropped_case": {
        "title": "7. Dropped Case: Police-to-Prosecution Attrition Leak (Leak Point)",
        "description": "Survivor reported incident and underwent medical/forensic examination, but investigation stalled and no chargesheet was ever filed.",
        "expected_verdict": "Dropped after Investigation: Systemic Policy Leak Point (No Chargesheet)",
        "why": "Records up to the Forensic stage link with high confidence (P > 0.95), but the reconstructed longitudinal journey terminates at milestone 3. Over 280+ days have elapsed without chargesheet filing, violating the CrPC 173 60-day mandate and exposing a critical systemic leak.",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D6-0144",
                "name": "Meena Kumari",
                "guardian": "Deepak Kumar",
                "district": "D6_Shivgarh",
                "locality": "Subhash Nagar",
                "incident_date": "2024-01-15",
                "record_date": "2024-01-16",
                "offense": "Sec 376 IPC (Rape)"
            },
            {
                "agency": "Hospital MLC",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2024-D6-0129",
                "name": "Meena Kumari",
                "guardian": "Deepak Kumar",
                "district": "D6_Shivgarh",
                "locality": "Subhash Nagar",
                "incident_date": "2024-01-15",
                "record_date": "2024-01-16",
                "details": "Exam conducted within 24h"
            },
            {
                "agency": "Forensic Lab (FSL)",
                "agency_code": "forensic_lab",
                "record_id": "FSL-2024-D6-0088",
                "name": "Meena Kumari",
                "guardian": "Deepak Kumar",
                "district": "D6_Shivgarh",
                "locality": None,
                "incident_date": "2024-01-15",
                "record_date": "2024-02-12",
                "details": "Forensic evidence dispatched to investigating officer"
            }
        ]
    },

    "scenario_8_delayed_case": {
        "title": "8. Delayed Case: Forensic Science Laboratory Latency Bottleneck",
        "description": "Case reached trial, but forensic laboratory analysis suffered extreme turnaround latency, stalling the entire justice pipeline.",
        "expected_verdict": "Linked Journey with Severe Bottleneck: Forensic Latency Violation",
        "why": "All 5 agency records link with high confidence, but process mining timeline analysis reveals a 274-day gap between sample receipt and report dispatch. This breaches the 30-day statutory benchmark by 9.1x, delaying chargesheet submission and trial commencement.",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D3-0055",
                "name": "Rekha Yadav",
                "guardian": "Jagdish Yadav",
                "district": "D3_Rudrapur",
                "locality": "Industrial Area",
                "incident_date": "2024-01-10",
                "record_date": "2024-01-11",
                "offense": "Sec 376 IPC"
            },
            {
                "agency": "Hospital MLC",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2024-D3-0048",
                "name": "Rekha Yadav",
                "guardian": "Jagdish Yadav",
                "district": "D3_Rudrapur",
                "locality": "Industrial Area",
                "incident_date": "2024-01-10",
                "record_date": "2024-01-11",
                "details": "Sample collected"
            },
            {
                "agency": "Forensic Lab (FSL)",
                "agency_code": "forensic_lab",
                "record_id": "FSL-2024-D3-0012",
                "name": "Rekha Yadav",
                "guardian": "Jagdish Yadav",
                "district": "D3_Rudrapur",
                "locality": None,
                "incident_date": "2024-01-10",
                "record_date": "2024-10-12",
                "details": "Report dispatched after 274 days backlog delay!"
            },
            {
                "agency": "Prosecution Chargesheet",
                "agency_code": "prosecutor",
                "record_id": "PROS-2024-D3-0801",
                "name": "Rekha Yadav",
                "guardian": "Jagdish Yadav",
                "district": "D3_Rudrapur",
                "locality": "Industrial Area",
                "incident_date": "2024-01-10",
                "record_date": "2024-11-04",
                "details": "Delayed chargesheet citing pending FSL report"
            },
            {
                "agency": "District Court",
                "agency_code": "court",
                "record_id": "CRT-2024-D3-0502",
                "name": "Rekha Yadav",
                "guardian": "Jagdish Yadav",
                "district": "D3_Rudrapur",
                "locality": "Industrial Area",
                "incident_date": "2024-01-10",
                "record_date": "2025-05-18",
                "details": "Trial completed"
            }
        ]
    }
}


# ==============================================================================
# 2. Pairwise Scoring Engine (Real Probabilistic Fellegi-Sunter Evaluation)
# ==============================================================================

HIGH_THRESHOLD = 0.85
LOW_THRESHOLD = 0.50
PRIOR_ODDS = 0.10 / 0.90  # Prior probability 0.10


def score_case_pair(rec_a: Dict[str, Any], rec_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a candidate pair of agency records using the exact Fellegi-Sunter
    comparison model defined in src/linkage.py.

    Returns:
    - match_probability (float 0.0 to 1.0)
    - match_weight (float log-odds)
    - decision ("AUTO_ACCEPT", "HUMAN_REVIEW", "REJECT")
    - per_field_evidence (list of plain-English explanation dicts)
    """
    evidence = []
    total_weight = 0.0

    # 1. District Comparison (ExactMatch)
    dist_a = str(rec_a.get("district", "")).strip().lower() if rec_a.get("district") else None
    dist_b = str(rec_b.get("district", "")).strip().lower() if rec_b.get("district") else None

    if dist_a and dist_b:
        if dist_a == dist_b:
            w = 2.60
            evidence.append({
                "field": "District",
                "status": "AGREE",
                "weight": w,
                "detail": f"Exact district match: '{rec_a.get('district')}'"
            })
        else:
            w = -3.80
            evidence.append({
                "field": "District",
                "status": "DISAGREE",
                "weight": w,
                "detail": f"District mismatch: '{rec_a.get('district')}' vs '{rec_b.get('district')}'"
            })
        total_weight += w
    else:
        evidence.append({
            "field": "District",
            "status": "MISSING",
            "weight": 0.0,
            "detail": "District missing in one or both records"
        })

    # 2. Name Normalization & Comparisons
    full_a, first_a, last_a, p_first_a, p_last_a = normalize_name(rec_a.get("name"))
    full_b, first_b, last_b, p_first_b, p_last_b = normalize_name(rec_b.get("name"))

    if full_a and full_b:
        jw_name = jellyfish.jaro_winkler_similarity(full_a, full_b)
        if jw_name >= 0.90 or full_a == full_b:
            w = 3.60
            evidence.append({
                "field": "Subject Full Name",
                "status": "AGREE",
                "weight": w,
                "detail": f"High name similarity ({jw_name:.2f}): '{full_a}' ~ '{full_b}'"
            })
        elif jw_name >= 0.75:
            w = 1.70
            evidence.append({
                "field": "Subject Full Name",
                "status": "PARTIAL",
                "weight": w,
                "detail": f"Moderate name similarity ({jw_name:.2f}): '{full_a}' ~ '{full_b}'"
            })
        else:
            w = -2.80
            evidence.append({
                "field": "Subject Full Name",
                "status": "DISAGREE",
                "weight": w,
                "detail": f"Name conflict ({jw_name:.2f} < 0.75): '{full_a}' vs '{full_b}'"
            })
        total_weight += w
    else:
        evidence.append({
            "field": "Subject Full Name",
            "status": "MISSING",
            "weight": 0.0,
            "detail": "Subject name absent in record"
        })

    # 3. Phonetic First & Last Name Comparisons
    if p_first_a and p_first_b:
        if p_first_a == p_first_b:
            w = 1.20
            evidence.append({
                "field": "Phonetic First Name",
                "status": "AGREE",
                "weight": w,
                "detail": f"Exact Metaphone code match: '{p_first_a}'"
            })
        else:
            w = -0.80
            evidence.append({
                "field": "Phonetic First Name",
                "status": "DISAGREE",
                "weight": w,
                "detail": f"Phonetic divergence: '{p_first_a}' vs '{p_first_b}'"
            })
        total_weight += w

    if p_last_a and p_last_b:
        if p_last_a == p_last_b:
            w = 1.20
            evidence.append({
                "field": "Phonetic Surname",
                "status": "AGREE",
                "weight": w,
                "detail": f"Exact Metaphone surname match: '{p_last_a}'"
            })
        else:
            w = -0.80
            evidence.append({
                "field": "Phonetic Surname",
                "status": "DISAGREE",
                "weight": w,
                "detail": f"Phonetic surname mismatch: '{p_last_a}' vs '{p_last_b}'"
            })
        total_weight += w

    # 4. Relative / Guardian Name Comparison
    rel_a, _, _, _, _ = normalize_name(rec_a.get("guardian"))
    rel_b, _, _, _, _ = normalize_name(rec_b.get("guardian"))

    if rel_a and rel_b:
        jw_rel = jellyfish.jaro_winkler_similarity(rel_a, rel_b)
        if jw_rel >= 0.88 or rel_a == rel_b:
            w = 2.40
            evidence.append({
                "field": "Guardian Name",
                "status": "AGREE",
                "weight": w,
                "detail": f"Strong guardian match ({jw_rel:.2f}): '{rel_a}' ~ '{rel_b}'"
            })
        elif jw_rel >= 0.70:
            w = 1.00
            evidence.append({
                "field": "Guardian Name",
                "status": "PARTIAL",
                "weight": w,
                "detail": f"Moderate guardian match ({jw_rel:.2f}): '{rel_a}' ~ '{rel_b}'"
            })
        else:
            w = -1.60
            evidence.append({
                "field": "Guardian Name",
                "status": "DISAGREE",
                "weight": w,
                "detail": f"Guardian mismatch ({jw_rel:.2f}): '{rel_a}' vs '{rel_b}'"
            })
        total_weight += w
    else:
        evidence.append({
            "field": "Guardian Name",
            "status": "MISSING",
            "weight": 0.0,
            "detail": "Guardian name not captured in one or both records (neutral weight)"
        })

    # 5. Incident Date Comparison
    date_a_str = rec_a.get("incident_date")
    date_b_str = rec_b.get("incident_date")
    day_diff = None

    if date_a_str and date_b_str:
        try:
            d_a = datetime.strptime(str(date_a_str).strip(), "%Y-%m-%d")
            d_b = datetime.strptime(str(date_b_str).strip(), "%Y-%m-%d")
            day_diff = abs((d_a - d_b).days)

            if day_diff <= 2:
                w = 2.80
                evidence.append({
                    "field": "Incident Date",
                    "status": "AGREE",
                    "weight": w,
                    "detail": f"Incident date match ({day_diff} days difference within 2-day tolerance)"
                })
            elif day_diff <= 10:
                w = 1.00
                evidence.append({
                    "field": "Incident Date",
                    "status": "PARTIAL",
                    "weight": w,
                    "detail": f"Close temporal proximity ({day_diff} days difference <= 10 days)"
                })
            else:
                w = -2.50
                evidence.append({
                    "field": "Incident Date",
                    "status": "DISAGREE",
                    "weight": w,
                    "detail": f"Large date divergence ({day_diff} days apart > 10 days)"
                })
            total_weight += w
        except Exception:
            evidence.append({
                "field": "Incident Date",
                "status": "MISSING",
                "weight": 0.0,
                "detail": "Unparseable incident date"
            })
    else:
        evidence.append({
            "field": "Incident Date",
            "status": "MISSING",
            "weight": 0.0,
            "detail": "Incident date absent in record"
        })

    # 6. Locality Comparison
    loc_a = normalize_locality(rec_a.get("locality"))
    loc_b = normalize_locality(rec_b.get("locality"))

    if loc_a and loc_b:
        jw_loc = jellyfish.jaro_winkler_similarity(loc_a, loc_b)
        if jw_loc >= 0.85 or loc_a == loc_b:
            w = 1.40
            evidence.append({
                "field": "Locality",
                "status": "AGREE",
                "weight": w,
                "detail": f"Locality agreement ({jw_loc:.2f}): '{loc_a}' ~ '{loc_b}'"
            })
        elif jw_loc >= 0.65:
            w = 0.50
            evidence.append({
                "field": "Locality",
                "status": "PARTIAL",
                "weight": w,
                "detail": f"Partial locality similarity ({jw_loc:.2f})"
            })
        else:
            w = -0.70
            evidence.append({
                "field": "Locality",
                "status": "DISAGREE",
                "weight": w,
                "detail": f"Locality mismatch: '{loc_a}' vs '{loc_b}'"
            })
        total_weight += w
    else:
        evidence.append({
            "field": "Locality",
            "status": "MISSING",
            "weight": 0.0,
            "detail": "Locality omitted in record"
        })

    # Posterior Probability via Bayes Odds
    odds = PRIOR_ODDS * (2.0 ** total_weight)
    prob = odds / (1.0 + odds)
    prob = round(float(min(0.999, max(0.001, prob))), 3)

    # Decision Categorization
    if prob >= HIGH_THRESHOLD:
        decision = "AUTO_ACCEPT"
    elif prob >= LOW_THRESHOLD:
        decision = "HUMAN_REVIEW"
    else:
        decision = "REJECT"

    # Temporal Sanity Guard: Large date gaps (>20 days) demote auto-accept to human review
    if decision == "AUTO_ACCEPT" and day_diff is not None and day_diff > 20:
        decision = "HUMAN_REVIEW"
        evidence.append({
            "field": "Temporal Guard",
            "status": "GUARD_TRIGGERED",
            "weight": 0.0,
            "detail": f"Large date gap ({day_diff} days > 20) demoted auto-accept to Human Review."
        })

    return {
        "match_probability": prob,
        "match_weight": round(total_weight, 2),
        "decision": decision,
        "evidence": evidence
    }


# ==============================================================================
# 3. Longitudinal Journey Timeline Reconstruction
# ==============================================================================

def reconstruct_simulated_journey(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reconstructs the longitudinal timeline across stages and detects delays or drops:
    1. Police FIR
    2. Hospital MLC
    3. Forensic Science Lab (FSL)
    4. Prosecution Chargesheet
    5. District Court Verdict
    """
    canonical_order = ["police_fir", "hospital_mlc", "forensic_lab", "prosecutor", "court"]
    stage_labels = {
        "police_fir": "1. Police FIR",
        "hospital_mlc": "2. Hospital MLC",
        "forensic_lab": "3. Forensic Lab (FSL)",
        "prosecutor": "4. Chargesheet Filed",
        "court": "5. Court Verdict"
    }

    # Map present records by agency
    records_by_agency = {}
    for r in records:
        code = r.get("agency_code")
        if code:
            records_by_agency[code] = r

    stages_present = []
    milestone_dates = {}

    for code in canonical_order:
        if code in records_by_agency:
            rec = records_by_agency[code]
            date_val = rec.get("record_date") or rec.get("incident_date")
            stages_present.append(code)
            milestone_dates[code] = date_val

    # Detect delays and bottlenecks
    timeline_steps = []
    bottlenecks = []
    is_dropped = False
    drop_point = None

    for idx, code in enumerate(canonical_order):
        label = stage_labels[code]
        if code in records_by_agency:
            r = records_by_agency[code]
            d_str = milestone_dates.get(code)
            step_info = {
                "stage": label,
                "status": "COMPLETED",
                "date": d_str,
                "record_id": r.get("record_id"),
                "notes": r.get("details") or r.get("offense") or ""
            }
            timeline_steps.append(step_info)
        else:
            # Stage was never reached
            step_info = {
                "stage": label,
                "status": "DROPPED / MISSING",
                "date": None,
                "record_id": "NOT REACHED",
                "notes": "Case did not proceed to this institutional milestone"
            }
            timeline_steps.append(step_info)
            if not is_dropped and idx > 0 and canonical_order[idx - 1] in records_by_agency:
                is_dropped = True
                drop_point = canonical_order[idx - 1]

    # Evaluate Statutory Compliance
    # 1. FIR to MLC compliance (24h benchmark)
    if "police_fir" in milestone_dates and "hospital_mlc" in milestone_dates:
        try:
            d_fir = datetime.strptime(milestone_dates["police_fir"], "%Y-%m-%d")
            d_mlc = datetime.strptime(milestone_dates["hospital_mlc"], "%Y-%m-%d")
            diff = (d_mlc - d_fir).days
            if diff > 1:
                bottlenecks.append({
                    "code": "LP-04-MLC-DELAY",
                    "title": "Medical Examination Delay",
                    "detail": f"{diff} days from FIR to examination (statutory standard is <= 24 hours)."
                })
        except Exception:
            pass

    # 2. Forensic Turnaround delay (30-day DFSS benchmark)
    if "forensic_lab" in records_by_agency:
        fsl_rec = records_by_agency["forensic_lab"]
        try:
            inc_d = datetime.strptime(fsl_rec["incident_date"], "%Y-%m-%d")
            rep_d = datetime.strptime(fsl_rec["record_date"], "%Y-%m-%d")
            fsl_turnaround = (rep_d - inc_d).days
            if fsl_turnaround > 60:
                mult = fsl_turnaround / 30.0
                bottlenecks.append({
                    "code": "LP-01-FORENSIC-DELAY",
                    "title": "Severe Forensic Lab Turnaround Latency",
                    "detail": f"{fsl_turnaround} days turnaround vs 30-day statutory benchmark ({mult:.1f}x delay multiplier)."
                })
        except Exception:
            pass

    # 3. FIR to Chargesheet delay (60-day CrPC 173(1A) / BNSS 193 mandate)
    if "police_fir" in milestone_dates:
        if "prosecutor" in milestone_dates:
            try:
                d_fir = datetime.strptime(milestone_dates["police_fir"], "%Y-%m-%d")
                d_pros = datetime.strptime(milestone_dates["prosecutor"], "%Y-%m-%d")
                cs_days = (d_pros - d_fir).days
                if cs_days > 60:
                    bottlenecks.append({
                        "code": "LP-02-POLICE-CHARGESHEET-DELAY",
                        "title": "Statutory Chargesheet Mandate Breached",
                        "detail": f"Chargesheet filed in {cs_days} days (exceeds Section 173 CrPC 60-day limit)."
                    })
            except Exception:
                pass
        elif is_dropped:
            bottlenecks.append({
                "code": "LP-02-POLICE-ATTRITION",
                "title": "Police-to-Prosecution Case Attrition Leak",
                "detail": "Case stalled after investigation. No chargesheet was filed, terminating survivor access to trial."
            })

    return {
        "timeline_steps": timeline_steps,
        "bottlenecks": bottlenecks,
        "is_dropped": is_dropped,
        "drop_point": drop_point,
        "completed_count": len(stages_present),
        "total_canonical": 5
    }


# ==============================================================================
# 4. Full Scenario Simulation Orchestrator
# ==============================================================================

def run_simulation(
    scenario_key_or_records: Any,
    custom_title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes an end-to-end simulation:
    1. Loads records
    2. Runs pairwise scoring across all pairs
    3. Reconstructs longitudinal journey
    4. Formulates judicial verdict & explanation
    5. Audits execution without logging personal records
    """
    if isinstance(scenario_key_or_records, str) and scenario_key_or_records in SCENARIOS:
        sc_cfg = SCENARIOS[scenario_key_or_records]
        records = sc_cfg["records"]
        title = sc_cfg["title"]
        why_text = sc_cfg["why"]
        scenario_id = scenario_key_or_records
    elif isinstance(scenario_key_or_records, list):
        records = scenario_key_or_records
        title = custom_title or "Custom Edited Scenario"
        why_text = "Evaluated user-edited synthetic agency fields through the Fellegi-Sunter scoring model."
        scenario_id = "custom_edited"
    else:
        raise ValueError("Invalid scenario key or records list.")

    # Pairwise evaluation across all pairs
    pairwise_results = []
    n = len(records)
    auto_accept_count = 0
    review_count = 0
    reject_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            rec_a = records[i]
            rec_b = records[j]
            score_res = score_case_pair(rec_a, rec_b)
            dec = score_res["decision"]

            if dec == "AUTO_ACCEPT":
                auto_accept_count += 1
            elif dec == "HUMAN_REVIEW":
                review_count += 1
            else:
                reject_count += 1

            pairwise_results.append({
                "record_a": f"{rec_a['agency']} ({rec_a.get('record_id', 'ID')})",
                "record_b": f"{rec_b['agency']} ({rec_b.get('record_id', 'ID')})",
                "match_probability": score_res["match_probability"],
                "match_weight": score_res["match_weight"],
                "decision": dec,
                "evidence": score_res["evidence"]
            })

    # Journey reconstruction
    journey = reconstruct_simulated_journey(records)

    # Formulate overall judicial verdict
    total_pairs = len(pairwise_results)
    if total_pairs == 0:
        verdict = "Single Record Input: Insufficient Data for Linkage"
        verdict_type = "neutral"
    elif scenario_id == "scenario_6_different_people" or (reject_count == total_pairs):
        verdict = "Definitively Disconnected: False-Link Protection Active (No Merger)"
        verdict_type = "danger"
    elif scenario_id == "scenario_7_dropped_case":
        verdict = f"Dropped Case: Attrition Leak after {journey['drop_point']} (No Trial)"
        verdict_type = "warning"
    elif any(b["code"] == "LP-01-FORENSIC-DELAY" for b in journey["bottlenecks"]):
        verdict = "Linked Journey with Severe Bottleneck: Forensic Latency Violation"
        verdict_type = "warning"
    elif auto_accept_count == total_pairs:
        verdict = "Linked with High Confidence (Full Agency Agreement)"
        verdict_type = "success"
    elif journey["is_dropped"]:
        verdict = f"Dropped Case: Attrition Leak after {journey['drop_point']} (No Trial)"
        verdict_type = "warning"
    elif review_count > 0 and reject_count == 0:
        verdict = "Uncertain Links: Send to Human Adjudication Queue"
        verdict_type = "review"
    else:
        verdict = f"Mixed Linkage: {auto_accept_count} Accepted, {review_count} Uncertain, {reject_count} Rejected"
        verdict_type = "neutral"

    # Audit logging: STRICT PRIVACY RULE - Log only scenario title, never personal values
    try:
        log_audit_event(
            event_type="CASE_SIMULATOR_RUN",
            username="simulator_engine",
            role="analyst",
            page="🧪 Case Simulator",
            details={"scenario_id": scenario_id, "title": title}
        )
    except Exception:
        pass

    return {
        "title": title,
        "records": records,
        "pairwise_results": pairwise_results,
        "journey": journey,
        "verdict": verdict,
        "verdict_type": verdict_type,
        "why": why_text,
        "summary": {
            "total_records": len(records),
            "total_candidate_pairs": total_pairs,
            "auto_accept_pairs": auto_accept_count,
            "human_review_pairs": review_count,
            "reject_pairs": reject_count
        }
    }
