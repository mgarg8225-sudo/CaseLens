"""
tests/test_policy_context.py - Unit tests for TinyFish Policy Context Integration.
==================================================================================
Tests:
1. Query sanitization blocks record IDs, synthetic district tokens, survivor PII, and micro-cohorts.
2. Query sanitization allows generic public policy queries.
3. API mocked success (Search + Fetch).
4. API mocked timeout and retry logic (max 2 retries).
5. API mocked empty search results handling.
6. Offline cache fallback mechanism.
7. External API audit logging (ensuring zero PII or case-level leak).
8. Web Agent statutory benchmark fact retrieval and caching.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import MagicMock

from src.policy_context import (
    sanitize_query,
    PrivacyViolationError,
    log_external_call,
    get_policy_context,
    get_government_benchmark_fact,
    load_cache,
    save_cache,
    normalize_leak_point_key,
    LEAK_POINTS_CATALOG
)


# =====================================================================
# 1. Privacy Guardrail & Query Sanitization Tests
# =====================================================================

class TestQuerySanitization:

    @pytest.mark.parametrize("bad_query", [
        "Investigate delay for survivor case ENT-00012 in system",
        "FIR-2023-8891 chargesheet status and police investigation",
        "Forensic toxicology backlog for specimen FSL-4412",
        "Medical examination report HOSP-00912 delay",
        "Court hearing record CRT-10928 pendency",
        "Prosecutor file PROS-3301 sanction",
        "Case 550e8400-e29b-41d4-a716-446655440000 tracking",
        "Hash token a1b2c3d4e5f60718293a4b5c6d7e8f90 tracking",
    ])
    def test_sanitize_query_blocks_record_and_case_ids(self, bad_query):
        """Proves that any query containing case IDs or entity hashes is strictly blocked."""
        with pytest.raises(PrivacyViolationError) as exc_info:
            sanitize_query(bad_query)
        assert "Privacy Guardrail Blocked" in str(exc_info.value)
        assert "record identifier" in str(exc_info.value).lower()

    @pytest.mark.parametrize("bad_query", [
        "D3_Rudrapur forensic lab turnaround delay",
        "Police chargesheet drop-off in D6_Shivgarh district",
        "Court trial pendency in Anandpur court",
        "Suryapur district FIR attrition rate",
        "Forensic testing delays in Rudrapur FSL",
        "Kalyanpur trial duration POCSO",
        "Mayaganj chargesheet compliance",
        "Devkalan judicial backlog",
        "Case leakage analysis for district D1",
        "Drop-off in d7 court trials",
    ])
    def test_sanitize_query_blocks_synthetic_districts(self, bad_query):
        """Proves that queries containing synthetic district names or district codes are blocked."""
        with pytest.raises(PrivacyViolationError) as exc_info:
            sanitize_query(bad_query)
        assert "Privacy Guardrail Blocked" in str(exc_info.value)
        assert "district identifier" in str(exc_info.value).lower()

    @pytest.mark.parametrize("bad_query", [
        "Survivor Pooja statement under section 164",
        "Victim Sunita medical examination delay",
        "Accused Rahul trial proceedings",
        "Survivor contact phone 9876543210 police follow-up",
        "Aadhaar card 1234 5678 9012 compensation verification",
        "Incident reported on 2024-03-15 forensic sample",
        "PIN code 110001 shelter home details",
    ])
    def test_sanitize_query_blocks_pii_and_names(self, bad_query):
        """Proves that survivor/accused names, phone numbers, aadhaar, or exact dates are blocked."""
        with pytest.raises(PrivacyViolationError) as exc_info:
            sanitize_query(bad_query)
        assert "Privacy Guardrail Blocked" in str(exc_info.value)

    @pytest.mark.parametrize("bad_query", [
        "Query for 12 cases in district failing chargesheet",
        "Investigate cohort with age=14 terminal stage drop off",
    ])
    def test_sanitize_query_blocks_micro_cohorts(self, bad_query):
        """Proves that micro-cohort or record attribute filters are blocked."""
        with pytest.raises(PrivacyViolationError) as exc_info:
            sanitize_query(bad_query)
        assert "Privacy Guardrail Blocked" in str(exc_info.value)

    @pytest.mark.parametrize("valid_query", [
        "India forensic science laboratory turnaround delay backlog policy guidelines POCSO CrPC",
        "India police FIR to chargesheet drop off rate attrition investigation statutory timeline 60 days",
        "India court trial delay pendency POCSO fast track special courts disposal timeline",
        "India medico legal care sexual assault survivor 24 hours statutory guideline Section 164A",
        "India survivor victim compensation scheme NALSA disbursal delay interim relief",
    ])
    def test_sanitize_query_allows_generic_policy_queries(self, valid_query):
        """Proves that clean, generic statutory and policy research queries pass sanitization."""
        cleaned = sanitize_query(valid_query)
        assert cleaned == " ".join(valid_query.strip().split())


# =====================================================================
# 2. Mocked API Success, Timeout & Retry Tests
# =====================================================================

class TestMockedTinyFishAPI:

    @pytest.fixture
    def temp_cache_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.remove(path)

    @pytest.fixture
    def temp_audit_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_api_mocked_success(self, temp_cache_file, temp_audit_file):
        """Verifies successful Search + Fetch execution with mock client."""
        mock_client = MagicMock()

        # Mock Search Result
        mock_search_item1 = MagicMock()
        mock_search_item1.url = "https://pib.gov.in/PressReleasePage.aspx?PRID=12345"
        mock_search_item1.title = "Fast Track Special Courts Mandate"
        mock_search_item1.snippet = "Courts must expedite trials under POCSO Act."

        mock_search_item2 = MagicMock()
        mock_search_item2.url = "https://lawcommissionofindia.nic.in/reports/report.pdf"
        mock_search_item2.title = "Law Commission Report on Judicial Delays"
        mock_search_item2.snippet = "Recommendations to resolve trial backlogs."

        mock_search_resp = MagicMock()
        mock_search_resp.results = [mock_search_item1, mock_search_item2]
        mock_client.search.query.return_value = mock_search_resp

        # Mock Fetch Result
        mock_fetch_item1 = MagicMock()
        mock_fetch_item1.url = "https://pib.gov.in/PressReleasePage.aspx?PRID=12345"
        mock_fetch_item1.text = "The Ministry of Law and Justice has directed all Fast Track Special Courts to adhere to strict 1-year disposal norms."

        mock_fetch_resp = MagicMock()
        mock_fetch_resp.results = [mock_fetch_item1]
        mock_client.fetch.get_contents.return_value = mock_fetch_resp

        result = get_policy_context(
            leak_point_label="court trial delay",
            force_refresh=True,
            cache_path=temp_cache_file,
            client_instance=mock_client
        )

        assert result["leak_code"] == "LP-03-COURT-DELAY"
        assert result["leak_point_label"] == "Court Trial Adjudication Pendency"
        assert len(result["sources"]) == 2

        # Check required fields for each source
        for source in result["sources"]:
            assert "title" in source and len(source["title"]) > 0
            assert "source_domain" in source and len(source["source_domain"]) > 0
            assert "url" in source and source["url"].startswith("http")
            assert "summary" in source and len(source["summary"]) > 0
            assert "retrieved_at" in source

        # Check cache was updated
        cache_data = load_cache(temp_cache_file)
        assert "court_trial_delay" in cache_data

    def test_api_mocked_timeout_and_retries(self, temp_cache_file):
        """Verifies retry mechanism: retries up to 2 times on timeout and then succeeds."""
        mock_client = MagicMock()

        mock_search_item = MagicMock()
        mock_search_item.url = "https://nalsa.gov.in/scheme"
        mock_search_item.title = "NALSA Scheme"
        mock_search_item.snippet = "Interim relief scheme."
        mock_search_resp = MagicMock()
        mock_search_resp.results = [mock_search_item]

        # Fails on attempt 1 & 2 with TimeoutError, succeeds on attempt 3
        mock_client.search.query.side_effect = [
            TimeoutError("Request timed out"),
            TimeoutError("Gateway timeout"),
            mock_search_resp
        ]

        result = get_policy_context(
            leak_point_label="victim compensation delay",
            force_refresh=True,
            cache_path=temp_cache_file,
            client_instance=mock_client
        )

        # 3 calls total (initial + 2 retries)
        assert mock_client.search.query.call_count == 3
        assert result["leak_code"] == "LP-05-VICTIM-COMPENSATION-DELAY"
        assert len(result["sources"]) >= 1

    def test_api_mocked_empty_results(self, temp_cache_file):
        """Verifies graceful handling when Search returns 0 results."""
        mock_client = MagicMock()
        mock_search_resp = MagicMock()
        mock_search_resp.results = []
        mock_client.search.query.return_value = mock_search_resp

        result = get_policy_context(
            leak_point_label="forensic report delay",
            force_refresh=True,
            cache_path=temp_cache_file,
            client_instance=mock_client
        )

        # Handles gracefully and falls back to statutory default benchmark
        assert result["leak_code"] == "LP-01-FORENSIC-DELAY"
        assert len(result["sources"]) >= 1
        assert "ijlmh.com" in [s["source_domain"] for s in result["sources"]]

    def test_cache_fallback_offline(self, temp_cache_file):
        """Proves that when offline (client is None or fails), cached data is returned."""
        cached_entry = {
            "leak_point_label": "Forensic Lab Turnaround Delay",
            "leak_code": "LP-01-FORENSIC-DELAY",
            "finding": "Cached finding: 137d forensic turnaround.",
            "suggested_intervention": "Cached intervention.",
            "query_used": "India forensic science laboratory turnaround delay backlog policy guidelines POCSO CrPC",
            "retrieved_at": "2026-10-17T09:00:00+00:00",
            "sources": [
                {
                    "title": "Cached Forensic Study",
                    "source_domain": "offline.gov.in",
                    "url": "https://offline.gov.in/forensic",
                    "summary": "Offline cached summary.",
                    "retrieved_at": "2026-10-17T09:00:00+00:00"
                }
            ]
        }
        save_cache({"forensic_report_delay": cached_entry}, temp_cache_file)

        # Call with no client (simulating offline mode)
        result = get_policy_context(
            leak_point_label="forensic report delay",
            force_refresh=False,
            cache_path=temp_cache_file,
            client_instance=None
        )

        assert result["finding"] == "Cached finding: 137d forensic turnaround."
        assert result["sources"][0]["source_domain"] == "offline.gov.in"

    def test_audit_logging_preserves_zero_pii(self, temp_audit_file):
        """Verifies that audit logging records ONLY timestamp, endpoint, query, and status."""
        query = "India police FIR to chargesheet drop off rate attrition investigation statutory timeline 60 days"
        log_external_call(query=query, status="SUCCESS", endpoint="search", audit_log_path=temp_audit_file)

        assert os.path.exists(temp_audit_file)
        with open(temp_audit_file, "r", encoding="utf-8") as f:
            entries = json.load(f)

        assert len(entries) == 1
        entry = entries[0]
        assert set(entry.keys()) == {"timestamp", "endpoint", "query", "status"}
        assert entry["query"] == query
        assert entry["status"] == "SUCCESS"
        assert entry["endpoint"] == "search"

        # Verify no sensitive keywords or IDs in log
        raw_log_text = json.dumps(entry)
        assert "ENT-" not in raw_log_text
        assert "Rudrapur" not in raw_log_text
        assert "Shivgarh" not in raw_log_text
        assert "api_key" not in raw_log_text

    def test_web_agent_benchmark_fact(self, temp_cache_file):
        """Verifies Web Agent benchmark fact retrieval and caching."""
        mock_client = MagicMock()
        mock_agent_run = MagicMock()
        mock_agent_run.result = {"result": 2012}
        mock_client.agent.run.return_value = mock_agent_run

        fact = get_government_benchmark_fact(
            force_refresh=True,
            cache_path=temp_cache_file,
            client_instance=mock_client
        )

        assert "POCSO" in fact["title"]
        assert fact["source_domain"] == "wikipedia.org"
        assert "one year" in fact["fact"].lower()
        assert fact["statutory_reference"] == "Section 35(2), Protection of Children from Sexual Offences (POCSO) Act, 2012"
        assert fact["enactment_year"] == 2012

        # Verify caching
        cache_data = load_cache(temp_cache_file)
        assert "government_benchmark_fact" in cache_data
