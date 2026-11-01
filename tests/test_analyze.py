"""
tests/test_analyze.py - Comprehensive Unit & Integration Tests for Analyze Your Data Feature.

Covers:
1. Template contents: validates zip file contains the 5 CSVs matching exact source headers.
2. Missing-column error: verifies 400 Bad Request listing missing columns.
3. Size limits: verifies 400 Bad Request for files > 5 MB and row counts > 2000 cases.
4. Sample run: runs /api/v1/analyze/sample and verifies KPIs, funnel, bottlenecks, linkage stats.
5. Misspelled name matching: validates phonetic and transliteration tolerance in single case.
6. Single case missing sections & invalid dates: verifies 400 Bad Request on < 2 sections or non-chronological dates.
7. Role restriction (RBAC): verifies Viewers receive 403 Forbidden on all /analyze/... endpoints.
8. Server-side name masking: verifies survivor names are masked (e.g. 'S***a S***a') with zero PII leakage.
9. Immutability guarantee: asserts that all data/*.json files remain byte-identical before and after runs.
"""

import io
import os
import glob
import json
import hashlib
import zipfile
import pytest
import pandas as pd
from starlette.testclient import TestClient

from src.api import create_app, DATA_STORE
from src.upload_analysis import REQUIRED_COLUMNS, LINKAGE_METHOD_NAME


@pytest.fixture(scope="module")
def api_client():
    """Initializes the FastAPI TestClient on primary data repository."""
    app = create_app(data_dir="data")
    DATA_STORE.budget_tracker.reset()
    client = TestClient(app)
    yield client


ANALYST_HEADERS = {"X-API-Key": "analyst-key-leaky-2026"}
ADMIN_HEADERS = {"X-API-Key": "admin-key-leaky-2026"}
VIEWER_HEADERS = {"X-API-Key": "viewer-key-leaky-2026"}


# ==============================================================================
# 1. Template Generation Tests
# ==============================================================================

def test_template_contents(api_client):
    """Verifies that GET /analyze/template returns a zip of header-only CSVs matching real schemas."""
    resp = api_client.get("/api/v1/analyze/template", headers=ANALYST_HEADERS)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"

    zip_bytes = io.BytesIO(resp.content)
    with zipfile.ZipFile(zip_bytes, "r") as zf:
        file_list = zf.namelist()
        expected_files = ["police_fir.csv", "hospital_mlc.csv", "forensic_lab.csv", "prosecutor.csv", "court.csv"]
        for exp in expected_files:
            assert exp in file_list, f"Missing {exp} in template zip archive"

            # Verify header columns match exact REQUIRED_COLUMNS
            csv_content = zf.read(exp).decode("utf-8").strip()
            headers = [h.strip() for h in csv_content.split(",")]
            agency_key = exp.replace(".csv", "")
            assert headers == REQUIRED_COLUMNS[agency_key], f"Headers mismatch in {exp}"


# ==============================================================================
# 2. File Validation Tests (Missing Columns & Size Limits)
# ==============================================================================

def test_missing_column_error(api_client):
    """Verifies that uploading CSVs with missing required columns returns 400 listing missing ones."""
    # Create invalid police_fir with missing 'complainant_name' and 'district'
    bad_fir = "fir_number,father_or_husband,age,locality,incident_date,fir_date,offense_category,police_station\nFIR-01,Ramesh,25,Area,2024-01-01,2024-01-02,rape,PS1\n"

    # Valid empty content for others
    mlc_csv = ",".join(REQUIRED_COLUMNS["hospital_mlc"]) + "\n"
    fsl_csv = ",".join(REQUIRED_COLUMNS["forensic_lab"]) + "\n"
    pros_csv = ",".join(REQUIRED_COLUMNS["prosecutor"]) + "\n"
    court_csv = ",".join(REQUIRED_COLUMNS["court"]) + "\n"

    files = {
        "police_fir": ("police_fir.csv", bad_fir.encode("utf-8"), "text/csv"),
        "hospital_mlc": ("hospital_mlc.csv", mlc_csv.encode("utf-8"), "text/csv"),
        "forensic_lab": ("forensic_lab.csv", fsl_csv.encode("utf-8"), "text/csv"),
        "prosecutor": ("prosecutor.csv", pros_csv.encode("utf-8"), "text/csv"),
        "court": ("court.csv", court_csv.encode("utf-8"), "text/csv")
    }

    resp = api_client.post("/api/v1/analyze/upload", files=files, headers=ANALYST_HEADERS)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "complainant_name" in detail
    assert "district" in detail
    assert "police_fir" in detail


def test_size_limit_bytes_error(api_client):
    """Verifies that uploading a CSV exceeding 5 MB returns 400 Bad Request."""
    # Generate 5.2 MB CSV payload
    large_line = "A" * 1000 + "\n"
    large_csv = (",".join(REQUIRED_COLUMNS["police_fir"]) + "\n" + (large_line * 5300)).encode("utf-8")
    assert len(large_csv) > 5 * 1024 * 1024

    mlc_csv = (",".join(REQUIRED_COLUMNS["hospital_mlc"]) + "\n").encode("utf-8")
    fsl_csv = (",".join(REQUIRED_COLUMNS["forensic_lab"]) + "\n").encode("utf-8")
    pros_csv = (",".join(REQUIRED_COLUMNS["prosecutor"]) + "\n").encode("utf-8")
    court_csv = (",".join(REQUIRED_COLUMNS["court"]) + "\n").encode("utf-8")

    files = {
        "police_fir": ("police_fir.csv", large_csv, "text/csv"),
        "hospital_mlc": ("hospital_mlc.csv", mlc_csv, "text/csv"),
        "forensic_lab": ("forensic_lab.csv", fsl_csv, "text/csv"),
        "prosecutor": ("prosecutor.csv", pros_csv, "text/csv"),
        "court": ("court.csv", court_csv, "text/csv")
    }

    resp = api_client.post("/api/v1/analyze/upload", files=files, headers=ANALYST_HEADERS)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "5 MB" in detail or "limit" in detail.lower()


def test_size_limit_row_count_error(api_client):
    """Verifies that uploading a CSV exceeding 2000 cases returns 400 Bad Request."""
    header = ",".join(REQUIRED_COLUMNS["police_fir"]) + "\n"
    row = "FIR-001,Sunita,Ramesh,25,Colony,D1,2024-01-01,2024-01-02,rape,PS1\n"
    csv_2005 = (header + (row * 2005)).encode("utf-8")

    mlc_csv = (",".join(REQUIRED_COLUMNS["hospital_mlc"]) + "\n").encode("utf-8")
    fsl_csv = (",".join(REQUIRED_COLUMNS["forensic_lab"]) + "\n").encode("utf-8")
    pros_csv = (",".join(REQUIRED_COLUMNS["prosecutor"]) + "\n").encode("utf-8")
    court_csv = (",".join(REQUIRED_COLUMNS["court"]) + "\n").encode("utf-8")

    files = {
        "police_fir": ("police_fir.csv", csv_2005, "text/csv"),
        "hospital_mlc": ("hospital_mlc.csv", mlc_csv, "text/csv"),
        "forensic_lab": ("forensic_lab.csv", fsl_csv, "text/csv"),
        "prosecutor": ("prosecutor.csv", pros_csv, "text/csv"),
        "court": ("court.csv", court_csv, "text/csv")
    }

    resp = api_client.post("/api/v1/analyze/upload", files=files, headers=ANALYST_HEADERS)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "2000" in detail


# ==============================================================================
# 3. Sample Benchmark Dataset Run Tests
# ==============================================================================

def test_sample_run(api_client):
    """Verifies running analysis on sample files produces expected KPIs, funnel, and linkage info."""
    resp = api_client.post("/api/v1/analyze/sample?limit=50", headers=ANALYST_HEADERS)
    assert resp.status_code == 200
    data = resp.json()

    # Summary KPIs
    assert "summary_kpis" in data
    kpis = data["summary_kpis"]
    assert "total_cases" in kpis
    assert kpis["total_cases"] > 0
    assert "fir_rate_pct" in kpis
    assert "verdict_rate_pct" in kpis
    assert "one_line_insight" in kpis

    # Funnel Stages
    assert "funnel_stages" in data
    assert len(data["funnel_stages"]) == 5
    stage_ids = [s["stage_id"] for s in data["funnel_stages"]]
    assert "FIR_REGISTERED" in stage_ids
    assert "COURT_VERDICT" in stage_ids

    # Bottlenecks
    assert "bottlenecks" in data
    assert "transitions" in data["bottlenecks"]

    # Linkage Stats & True Name Verification (NEVER label as Splink)
    assert "linkage_stats" in data
    l_stats = data["linkage_stats"]
    assert "linkage_method" in l_stats
    assert l_stats["linkage_method"] == LINKAGE_METHOD_NAME
    assert "splink" not in l_stats["linkage_method"].lower()

    # Privacy Info
    assert "privacy_info" in data
    assert "masked_fields" in data["privacy_info"]
    assert "suppressed_groups" in data["privacy_info"]


# ==============================================================================
# 4. Single Case Analysis Tests (Spelling Invariance, Validation, & Bottlenecks)
# ==============================================================================

def test_single_case_misspelled_name_matching(api_client):
    """Verifies that small spelling and transliteration differences ('Sunita' vs 'Suneeta') still match."""
    case_payload = {
        "police_fir": {
            "fir_number": "FIR-2024-TEST-001",
            "complainant_name": "Sunita Sharma",
            "father_or_husband": "Ramesh Sharma",
            "age": 28,
            "locality": "Civil Lines",
            "district": "D3_Rudrapur",
            "incident_date": "2024-03-01",
            "fir_date": "2024-03-02"
        },
        "hospital_mlc": {
            "mlc_number": "MLC-2024-TEST-001",
            "patient_name": "Suneeta Sharma",  # Transliteration variant
            "guardian_name": "Ramesh Sharma",
            "district": "D3_Rudrapur",
            "incident_date": "2024-03-01",
            "examination_date": "2024-03-02"
        },
        "prosecutor": {
            "prosecution_file_no": "PP-2024-TEST-001",
            "victim_name": "Sunita Sharma",
            "parent_spouse_name": "Sharma Ramesh",  # Surname transposition
            "district": "D3_Rudrapur",
            "incident_date": "2024-03-01",
            "chargesheet_date": "2024-05-15"
        }
    }

    resp = api_client.post("/api/v1/analyze/case", json=case_payload, headers=ANALYST_HEADERS)
    assert resp.status_code == 200
    data = resp.json()

    assert data["overall_linkage_label"] == "same case"
    assert data["masked_survivor_name"] == "S****a S****a"
    assert "Sunita" not in data["masked_survivor_name"]

    # Pairwise scores
    pairs = data["pairwise_match_scores"]
    assert len(pairs) >= 3
    for p in pairs:
        assert p["label"] == "same case"
        assert p["match_probability"] >= 0.85

    # Timeline & stall point
    timeline = data["timeline"]
    assert timeline["all_milestones_reached"] is False
    assert timeline["stalled_at"] == "4. Prosecution Chargesheet"
    assert timeline["missing_record_from"] == "5. Court Verdict"


def test_single_case_missing_sections_error(api_client):
    """Verifies that providing fewer than 2 sections returns 400 Bad Request."""
    payload_one_sec = {
        "police_fir": {
            "fir_number": "FIR-001",
            "complainant_name": "Anita Verma",
            "district": "D1_Suryapur"
        }
    }
    resp = api_client.post("/api/v1/analyze/case", json=payload_one_sec, headers=ANALYST_HEADERS)
    assert resp.status_code == 400
    assert "At least 2" in resp.json()["detail"]


def test_single_case_invalid_dates_error(api_client):
    """Verifies that illogical or chronologically reversed dates return 400 Bad Request."""
    # Verdict date preceding FIR registration date
    bad_dates_payload = {
        "police_fir": {
            "fir_number": "FIR-001",
            "complainant_name": "Pooja Singh",
            "district": "D2_Chandranagar",
            "incident_date": "2024-05-01",
            "fir_date": "2024-05-02"
        },
        "court": {
            "cnr_number": "CNR-001",
            "petitioner_victim": "Pooja Singh",
            "district": "D2_Chandranagar",
            "date_of_incident": "2024-05-01",
            "verdict_date": "2023-01-01"  # Illogical: Verdict year earlier than FIR!
        }
    }
    resp = api_client.post("/api/v1/analyze/case", json=bad_dates_payload, headers=ANALYST_HEADERS)
    assert resp.status_code == 400
    assert "precede" in resp.json()["detail"].lower() or "violation" in resp.json()["detail"].lower()


# ==============================================================================
# 5. Role Restriction (RBAC) Tests
# ==============================================================================

def test_role_restriction_viewer_gets_403(api_client):
    """Verifies that Viewers receive HTTP 403 Forbidden on all /analyze/... endpoints."""
    # 1. Template
    r_tmpl = api_client.get("/api/v1/analyze/template", headers=VIEWER_HEADERS)
    assert r_tmpl.status_code == 403
    assert "Forbidden" in r_tmpl.json()["detail"]

    # 2. Sample
    r_smpl = api_client.post("/api/v1/analyze/sample", headers=VIEWER_HEADERS)
    assert r_smpl.status_code == 403
    assert "Forbidden" in r_smpl.json()["detail"]

    # 3. Upload
    dummy_files = {"police_fir": ("fir.csv", b"fir_number\n", "text/csv")}
    r_upld = api_client.post("/api/v1/analyze/upload", files=dummy_files, headers=VIEWER_HEADERS)
    assert r_upld.status_code == 403
    assert "Forbidden" in r_upld.json()["detail"]

    # 4. Case
    r_case = api_client.post("/api/v1/analyze/case", json={"police_fir": {}}, headers=VIEWER_HEADERS)
    assert r_case.status_code == 403
    assert "Forbidden" in r_case.json()["detail"]


# ==============================================================================
# 6. Server-Side Name Masking Test
# ==============================================================================

def test_server_side_name_masking(api_client):
    """Verifies that survivor name is always masked server-side in case analysis."""
    payload = {
        "police_fir": {
            "fir_number": "FIR-2024-D1-01",
            "complainant_name": "Anita Kumari",
            "district": "D1_Suryapur",
            "incident_date": "2024-01-10",
            "fir_date": "2024-01-11"
        },
        "hospital_mlc": {
            "mlc_number": "MLC-2024-D1-01",
            "patient_name": "Anita Kumari",
            "district": "D1_Suryapur",
            "incident_date": "2024-01-10",
            "examination_date": "2024-01-11"
        }
    }
    resp = api_client.post("/api/v1/analyze/case", json=payload, headers=ANALYST_HEADERS)
    assert resp.status_code == 200
    res_str = resp.text
    # Raw name must NOT be present as a value in the response
    assert '"Anita Kumari"' not in res_str
    # Must be masked
    assert "A***a K****i" in res_str


# ==============================================================================
# 7. Audit Log Tests: Counts Only, Never Data
# ==============================================================================

def test_audit_log_records_counts_only_never_data(api_client):
    """Verifies that audit logging records only caller and counts (rows/cases), never data microdata."""
    case_payload = {
        "police_fir": {
            "fir_number": "FIR-AUDIT-01",
            "complainant_name": "Confidential Survivor",
            "district": "D3_Rudrapur",
            "incident_date": "2024-03-01",
            "fir_date": "2024-03-02"
        },
        "hospital_mlc": {
            "mlc_number": "MLC-AUDIT-01",
            "patient_name": "Confidential Survivor",
            "district": "D3_Rudrapur",
            "incident_date": "2024-03-01",
            "examination_date": "2024-03-02"
        }
    }
    resp = api_client.post("/api/v1/analyze/case", json=case_payload, headers=ANALYST_HEADERS)
    assert resp.status_code == 200

    logs = DATA_STORE.audit_logger.get_logs(limit=10)
    case_log = next((l for l in logs if l.get("endpoint") == "/api/v1/analyze/case"), None)
    assert case_log is not None
    assert case_log["role"] == "analyst"
    # Ensure details only has counts, never personal names or records
    details = case_log.get("details", {})
    assert "sections_count" in details
    assert "Confidential Survivor" not in json.dumps(case_log)


# ==============================================================================
# 8. Immutability Guarantee Test: data/*.json Files Byte-Identical
# ==============================================================================

def test_data_json_files_byte_identical_before_and_after(api_client):
    """
    Guarantees zero side effects on disk: all pipeline data/*.json files must be
    100% byte-for-byte identical before and after executing all analyze endpoints.
    """
    # All analytical and pipeline JSON files in data/
    json_files = [f for f in glob.glob("data/*.json") if not f.endswith("audit_log.json") and "active_analysis_registry" not in f]
    assert len(json_files) >= 10, f"Expected at least 10 pipeline JSON files, found {len(json_files)}"

    def compute_hashes():
        hashes = {}
        for fpath in json_files:
            with open(fpath, "rb") as f:
                hashes[fpath] = hashlib.sha256(f.read()).hexdigest()
        return hashes

    # 1. Snapshot hashes before run
    hashes_before = compute_hashes()

    # 2. Run /analyze/template
    api_client.get("/api/v1/analyze/template", headers=ANALYST_HEADERS)

    # 3. Run /analyze/sample
    api_client.post("/api/v1/analyze/sample?limit=25", headers=ANALYST_HEADERS)

    # 4. Run /analyze/case
    api_client.post("/api/v1/analyze/case", json={
        "police_fir": {
            "fir_number": "FIR-IMMUT-01",
            "complainant_name": "Sunita Sharma",
            "district": "D3_Rudrapur",
            "incident_date": "2024-02-01",
            "fir_date": "2024-02-02"
        },
        "hospital_mlc": {
            "mlc_number": "MLC-IMMUT-01",
            "patient_name": "Sunita Sharma",
            "district": "D3_Rudrapur",
            "incident_date": "2024-02-01",
            "examination_date": "2024-02-02"
        }
    }, headers=ANALYST_HEADERS)

    # 5. Snapshot hashes after run
    hashes_after = compute_hashes()

    # 6. Verify byte-identical match on every single file
    for fpath in json_files:
        assert hashes_before[fpath] == hashes_after[fpath], (
            f"FILE CORRUPTION DETECTED: {fpath} was modified during analyze execution!"
        )


# ==============================================================================
# 9. Header Aliases, Whitespace/Case Insensitivity, & Multi-File Error Reporting
# ==============================================================================

def test_csv_validation_with_aliases(api_client):
    """Verifies that uploading CSVs with supported aliases (e.g. name, father_name, station) succeeds."""
    # police_fir with 'name' -> 'complainant_name', 'father_name' -> 'father_or_husband', 'police_station_name' -> 'police_station'
    fir_csv = (
        "fir_number,name,father_name,age,locality,district,incident_date,fir_date,offense_category,police_station_name\n"
        "FIR-ALIAS-01,Sunita Sharma,Ramesh Sharma,28,Civil Lines,D3_Rudrapur,2024-01-10,2024-01-12,rape,PS_Rudrapur_Central\n"
    ).encode("utf-8")

    # hospital_mlc with 'victim_name' -> 'patient_name', 'exam_date' -> 'examination_date'
    mlc_csv = (
        "mlc_number,victim_name,guardian_name,dob,residential_locality,district,incident_date,exam_date,injury_type,hospital_name\n"
        "MLC-ALIAS-01,Sunita Sharma,Ramesh Sharma,1996-01-01,Civil Lines,D3_Rudrapur,2024-01-10,2024-01-11,Blunt Trauma,District_Hospital\n"
    ).encode("utf-8")

    fsl_csv = (",".join(REQUIRED_COLUMNS["forensic_lab"]) + "\n").encode("utf-8")
    pros_csv = (",".join(REQUIRED_COLUMNS["prosecutor"]) + "\n").encode("utf-8")
    court_csv = (",".join(REQUIRED_COLUMNS["court"]) + "\n").encode("utf-8")

    files = {
        "police_fir": ("police_fir.csv", fir_csv, "text/csv"),
        "hospital_mlc": ("hospital_mlc.csv", mlc_csv, "text/csv"),
        "forensic_lab": ("forensic_lab.csv", fsl_csv, "text/csv"),
        "prosecutor": ("prosecutor.csv", pros_csv, "text/csv"),
        "court": ("court.csv", court_csv, "text/csv")
    }

    resp = api_client.post("/api/v1/analyze/upload", files=files, headers=ANALYST_HEADERS)
    assert resp.status_code == 200, f"Upload with aliases failed: {resp.text}"
    data = resp.json()
    assert data["summary_kpis"]["total_cases"] >= 1


def test_csv_validation_whitespace_and_case_insensitive(api_client):
    """Verifies that headers with leading/trailing whitespaces and different casing are recognized."""
    # police_fir with mixed case, hyphens, and whitespace padding
    fir_csv = (
        " FIR_NUMBER , Complainant Name , Father Or Husband , AGE , Locality , District , incident-date , FIR Date , Offense Category , Police Station \n"
        "FIR-WS-01,Pooja Verma,Suresh Verma,24,Station Road,D1_Alampur,2024-02-01,2024-02-03,rape,PS_Alampur_West\n"
    ).encode("utf-8")

    mlc_csv = (",".join(REQUIRED_COLUMNS["hospital_mlc"]) + "\n").encode("utf-8")
    fsl_csv = (",".join(REQUIRED_COLUMNS["forensic_lab"]) + "\n").encode("utf-8")
    pros_csv = (",".join(REQUIRED_COLUMNS["prosecutor"]) + "\n").encode("utf-8")
    court_csv = (",".join(REQUIRED_COLUMNS["court"]) + "\n").encode("utf-8")

    files = {
        "police_fir": ("police_fir.csv", fir_csv, "text/csv"),
        "hospital_mlc": ("hospital_mlc.csv", mlc_csv, "text/csv"),
        "forensic_lab": ("forensic_lab.csv", fsl_csv, "text/csv"),
        "prosecutor": ("prosecutor.csv", pros_csv, "text/csv"),
        "court": ("court.csv", court_csv, "text/csv")
    }

    resp = api_client.post("/api/v1/analyze/upload", files=files, headers=ANALYST_HEADERS)
    assert resp.status_code == 200, f"Upload with whitespace/case variations failed: {resp.text}"


def test_multi_file_missing_columns_error_message(api_client):
    """Verifies that when multiple uploaded CSVs are missing columns, all missing columns per file are listed."""
    bad_fir = "fir_number,age,locality,incident_date,fir_date,offense_category,police_station\nFIR-01,25,Area,2024-01-01,2024-01-02,rape,PS1\n".encode("utf-8")
    bad_mlc = "mlc_number,dob,residential_locality,district,incident_date,injury_type,hospital_name\nMLC-01,1995-01-01,Area,D3_Rudrapur,2024-01-01,Trauma,Hosp\n".encode("utf-8")
    fsl_csv = (",".join(REQUIRED_COLUMNS["forensic_lab"]) + "\n").encode("utf-8")
    pros_csv = (",".join(REQUIRED_COLUMNS["prosecutor"]) + "\n").encode("utf-8")
    court_csv = (",".join(REQUIRED_COLUMNS["court"]) + "\n").encode("utf-8")

    files = {
        "police_fir": ("police_fir.csv", bad_fir, "text/csv"),
        "hospital_mlc": ("hospital_mlc.csv", bad_mlc, "text/csv"),
        "forensic_lab": ("forensic_lab.csv", fsl_csv, "text/csv"),
        "prosecutor": ("prosecutor.csv", pros_csv, "text/csv"),
        "court": ("court.csv", court_csv, "text/csv")
    }

    resp = api_client.post("/api/v1/analyze/upload", files=files, headers=ANALYST_HEADERS)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    # Check that both files' missing columns are included in detail
    assert "police_fir.csv" in detail
    assert "complainant_name" in detail
    assert "hospital_mlc.csv" in detail
    assert "patient_name" in detail
    assert "examination_date" in detail


def test_successful_upload_synthetic_csvs(api_client):
    """Verifies a full successful upload of valid 5-agency synthetic CSV records."""
    fir_rows = (
        ",".join(REQUIRED_COLUMNS["police_fir"]) + "\n"
        "FIR-SYN-01,Meena Devi,Ram Lal,32,Market Road,D3_Rudrapur,2024-03-01,2024-03-03,domestic_violence,PS_Rudrapur_Central\n"
        "FIR-SYN-02,Kavita Singh,Rajesh Singh,27,Sector 4,D2_Chandranagar,2024-03-05,2024-03-06,rape,PS_Chandranagar_East\n"
    ).encode("utf-8")

    mlc_rows = (
        ",".join(REQUIRED_COLUMNS["hospital_mlc"]) + "\n"
        "MLC-SYN-01,Meena Devi,Ram Lal,1992-05-10,Market Road,D3_Rudrapur,2024-03-01,2024-03-02,Blunt Trauma,District_Civil_Hospital_Rudrapur\n"
        "MLC-SYN-02,Kavita Singh,Rajesh Singh,1997-08-20,Sector 4,D2_Chandranagar,2024-03-05,2024-03-06,Physical Trace,District_Civil_Hospital_Chandranagar\n"
    ).encode("utf-8")

    fsl_rows = (
        ",".join(REQUIRED_COLUMNS["forensic_lab"]) + "\n"
        "FSL-SYN-01,Meena Devi,Ram Lal,32,D3_Rudrapur,2024-03-08,2024-04-15,Biological Serology,Regional_FSL_Rudrapur\n"
    ).encode("utf-8")

    pros_rows = (
        ",".join(REQUIRED_COLUMNS["prosecutor"]) + "\n"
        "PP-SYN-01,Meena Devi,Ram Lal,32,Market Road,D3_Rudrapur,2024-03-01,2024-05-20,domestic_violence,PP_Office_Rudrapur\n"
    ).encode("utf-8")

    court_rows = (
        ",".join(REQUIRED_COLUMNS["court"]) + "\n"
        "CNR-SYN-01,Meena Devi,Ram Lal,,Market Road,D3_Rudrapur,2024-03-01,2024-11-10,Conviction,District_Sessions_Court_Rudrapur\n"
    ).encode("utf-8")

    files = {
        "police_fir": ("police_fir.csv", fir_rows, "text/csv"),
        "hospital_mlc": ("hospital_mlc.csv", mlc_rows, "text/csv"),
        "forensic_lab": ("forensic_lab.csv", fsl_rows, "text/csv"),
        "prosecutor": ("prosecutor.csv", pros_rows, "text/csv"),
        "court": ("court.csv", court_rows, "text/csv")
    }

    resp = api_client.post("/api/v1/analyze/upload", files=files, headers=ANALYST_HEADERS)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["summary_kpis"]["total_cases"] >= 1
    assert len(res_data["funnel_stages"]) == 5
    assert "transitions" in res_data["bottlenecks"]
    assert res_data["linkage_stats"]["linkage_method"] == LINKAGE_METHOD_NAME


def test_multi_case_upload_cross_agency_processing_and_linking(api_client):
    """
    Verifies that multi-case uploads across all 5 CSVs are actually processed and linked across agencies:
    - 3 distinct cases with multi-agency records.
    - Fellegi-Sunter scoring links matching agency records together into case entities.
    - Transition latency percentiles (P25, P75, P90, P95, std, IQR), delay buckets, and statutory compliance are derived.
    - Dashboard bundle properly formats inter_stage_transitions, comparative before/after, and honest linkage metrics.
    """
    from src.upload_analysis import build_dashboard_bundle_from_analysis

    fir_rows = (
        ",".join(REQUIRED_COLUMNS["police_fir"]) + "\n"
        "FIR-M1,Anjali Sharma,Mohan Sharma,24,Civil Lines,D3_Rudrapur,2024-01-10,2024-01-11,rape,PS_Rudrapur\n"
        "FIR-M2,Rekha Devi,Suresh Devi,30,Station Road,D3_Rudrapur,2024-02-01,2024-02-02,domestic_violence,PS_Rudrapur\n"
        "FIR-M3,Pooja Patel,Dinesh Patel,22,Gandhi Chowk,D2_Chandranagar,2024-03-01,2024-03-02,assault,PS_Chandranagar\n"
    ).encode("utf-8")

    mlc_rows = (
        ",".join(REQUIRED_COLUMNS["hospital_mlc"]) + "\n"
        "MLC-M1,Anjali Sharma,Mohan Sharma,2000-01-01,Civil Lines,D3_Rudrapur,2024-01-10,2024-01-11,Severe Trauma,Civil_Hosp_Rudrapur\n"
        "MLC-M2,Rekha Devi,Suresh Devi,1994-01-01,Station Road,D3_Rudrapur,2024-02-01,2024-02-02,Physical Injury,Civil_Hosp_Rudrapur\n"
        "MLC-M3,Pooja Patel,Dinesh Patel,2002-01-01,Gandhi Chowk,D2_Chandranagar,2024-03-01,2024-03-03,Bruises,Civil_Hosp_Chandranagar\n"
    ).encode("utf-8")

    fsl_rows = (
        ",".join(REQUIRED_COLUMNS["forensic_lab"]) + "\n"
        "FSL-M1,Anjali Sharma,Mohan Sharma,24,D3_Rudrapur,2024-01-15,2024-02-20,Biological DNA,FSL_Division_Rudrapur\n"
        "FSL-M2,Rekha Devi,Suresh Devi,30,D3_Rudrapur,2024-02-05,2024-03-10,Toxicology,FSL_Division_Rudrapur\n"
    ).encode("utf-8")

    pros_rows = (
        ",".join(REQUIRED_COLUMNS["prosecutor"]) + "\n"
        "PP-M1,Anjali Sharma,Mohan Sharma,24,Civil Lines,D3_Rudrapur,2024-01-10,2024-03-25,rape,PP_Rudrapur\n"
    ).encode("utf-8")

    court_rows = (
        ",".join(REQUIRED_COLUMNS["court"]) + "\n"
        "CNR-M1,Anjali Sharma,Mohan Sharma,2000-01-01,Civil Lines,D3_Rudrapur,2024-01-10,2024-10-15,Conviction,Court_Rudrapur\n"
    ).encode("utf-8")

    files = {
        "police_fir": ("police_fir.csv", fir_rows, "text/csv"),
        "hospital_mlc": ("hospital_mlc.csv", mlc_rows, "text/csv"),
        "forensic_lab": ("forensic_lab.csv", fsl_rows, "text/csv"),
        "prosecutor": ("prosecutor.csv", pros_rows, "text/csv"),
        "court": ("court.csv", court_rows, "text/csv")
    }

    resp = api_client.post("/api/v1/analyze/upload", files=files, headers=ANALYST_HEADERS)
    assert resp.status_code == 200
    res = resp.json()

    # 1. Cross-agency linkage verification
    stats = res["linkage_stats"]
    assert stats["total_pairs_evaluated"] > 0
    assert stats["pairs_linked"] >= 3, f"Expected cross-agency links, got {stats['pairs_linked']}"
    assert res["summary_kpis"]["total_cases"] == 3

    # 2. Funnel metrics verification
    stages = {s["stage_id"]: s for s in res["funnel_stages"]}
    assert stages["FIR_REGISTERED"]["raw_count"] == 3
    assert stages["MEDICAL_EXAM"]["raw_count"] == 3
    assert stages["FORENSIC_REPORT"]["raw_count"] == 2
    assert stages["CHARGESHEET_FILED"]["raw_count"] == 1
    assert stages["COURT_VERDICT"]["raw_count"] == 1

    # 3. Transitions & Non-parametric percentiles verification
    transitions = res["bottlenecks"]["transitions"]
    assert len(transitions) >= 1
    for pair, entry in transitions.items():
        assert "median_days" in entry
        assert "mean_days" in entry
        assert "std_days" in entry
        assert "p25_days" in entry
        assert "p75_days" in entry
        assert "p90_days" in entry
        assert "p95_days" in entry
        assert "iqr_days" in entry
        assert "delay_buckets" in entry
        assert "statutory_compliance" in entry

    # 4. Bundle construction verification
    bundle = build_dashboard_bundle_from_analysis(res)
    assert bundle["summary"]["pipeline_health"]["total_reconstructed_cases"] == 3
    assert len(bundle["bottlenecks"]["inter_stage_transitions"]) >= 1
    assert bundle["bottlenecks"]["inter_stage_transitions"][0]["transition"] in transitions

    # Linkage precision/recall are None when ungrounded (honest reporting)
    assert bundle["evaluation"]["pairwise_metrics"]["metrics_at_default_threshold"]["recall"] is None
    assert bundle["evaluation"]["pairwise_metrics"]["metrics_at_default_threshold"]["precision"] is None

    # Before / After comparative breakdown exists
    assert "comparative_table" in bundle["before_after"]
    assert len(bundle["before_after"]["comparative_table"]) >= 3


