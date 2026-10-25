"""
tests/test_global_filters.py - Comprehensive Verification of Global Filters,
Baseline Invariance, Consistency, Date Edge Cases, Suppression & Caching.
"""

import os
import pytest
from starlette.testclient import TestClient
from src.api import create_app, DATA_STORE
from src.privacy import get_configured_total_budget


@pytest.fixture(scope="module")
def api_client():
    """Initializes FastAPI TestClient with fresh cache and budget."""
    app = create_app(data_dir="data")
    DATA_STORE.budget_tracker.reset()
    DATA_STORE.audit_logger.clear()
    DATA_STORE.filter_cache.clear()
    client = TestClient(app)
    yield client


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "analyst-key-leaky-2026"}


@pytest.fixture
def admin_headers():
    return {"X-API-Key": "admin-key-leaky-2026"}


# ==============================================================================
# 1. Baseline Invariance: "All" reproduces current baseline numbers exactly
# ==============================================================================

def test_summary_all_equals_baseline(api_client, auth_headers):
    """Verifies that empty filters and 'All' filters reproduce the baseline exactly."""
    res_default = api_client.get("/api/v1/summary", headers=auth_headers)
    assert res_default.status_code == 200
    data_default = res_default.json()["data"]
    assert data_default["pipeline_health"]["total_reconstructed_cases"] == 5125

    res_all = api_client.get(
        "/api/v1/summary?district=All%20Districts&stage=All%20Stages&from_date=2022-12-31&to_date=2026-11-27",
        headers=auth_headers
    )
    assert res_all.status_code == 200
    data_all = res_all.json()["data"]
    assert data_all["pipeline_health"]["total_reconstructed_cases"] == 5125
    assert data_all["pipeline_health"]["cases_initiating_fir"] == data_default["pipeline_health"]["cases_initiating_fir"]
    assert data_all["pipeline_health"]["cases_reaching_verdict"] == data_default["pipeline_health"]["cases_reaching_verdict"]


def test_funnel_all_equals_baseline(api_client, auth_headers):
    """Verifies that Funnel with 'All' reproduces exact baseline stages."""
    res_default = api_client.get("/api/v1/funnel", headers=auth_headers)
    assert res_default.status_code == 200
    base_stages = res_default.json()["funnel"]["funnel_stages"]

    res_all = api_client.get("/api/v1/funnel?district=All&stage=All", headers=auth_headers)
    assert res_all.status_code == 200
    all_stages = res_all.json()["funnel"]["funnel_stages"]

    assert len(base_stages) == len(all_stages)
    for b_stage, a_stage in zip(base_stages, all_stages):
        assert b_stage["stage_name"] == a_stage["stage_name"]
        assert b_stage["noisy_case_count"] == a_stage["noisy_case_count"]


def test_bottlenecks_all_equals_baseline(api_client, auth_headers):
    """Verifies that Bottlenecks with 'All' reproduces baseline median and P95 latency."""
    res_default = api_client.get("/api/v1/bottlenecks", headers=auth_headers)
    assert res_default.status_code == 200
    base_data = res_default.json()["transitions"]

    res_all = api_client.get("/api/v1/bottlenecks?district=All%20Districts", headers=auth_headers)
    assert res_all.status_code == 200
    all_data = res_all.json()["transitions"]

    for trans in ["CHARGESHEET_FILED -> COURT_VERDICT", "FIR_REGISTERED -> CHARGESHEET_FILED"]:
        assert base_data[trans]["median_days"] == all_data[trans]["median_days"]
        assert base_data[trans]["p95_days"] == all_data[trans]["p95_days"]


def test_before_after_all_equals_baseline(api_client, auth_headers):
    """Verifies that Before/After endpoint returns 5,125 cases and baseline table for 'All'."""
    res = api_client.get("/api/v1/before-after", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["filtered_case_count"] == 5125
    assert data["filtered_event_count"] == 14384
    assert len(data["comparative_table"]) == 5


# ==============================================================================
# 2. Filters change at least the case counts
# ==============================================================================

def test_district_filter_changes_case_counts(api_client, auth_headers):
    """Verifies that filtering by a specific district narrows the case cohort below 5,125."""
    res_d3 = api_client.get("/api/v1/summary?district=D3_Rudrapur", headers=auth_headers)
    assert res_d3.status_code == 200
    data_d3 = res_d3.json()["data"]
    cases_d3 = data_d3["pipeline_health"]["total_reconstructed_cases"]
    assert 100 < cases_d3 < 5125

    res_d6 = api_client.get("/api/v1/summary?district=D6_Shivgarh", headers=auth_headers)
    assert res_d6.status_code == 200
    data_d6 = res_d6.json()["data"]
    cases_d6 = data_d6["pipeline_health"]["total_reconstructed_cases"]
    assert 100 < cases_d6 < 5125
    assert cases_d3 != cases_d6


def test_stage_filter_changes_funnel_and_counts(api_client, auth_headers):
    """Verifies that filtering by stage (e.g. COURT_VERDICT) returns a smaller cohort."""
    res_court = api_client.get("/api/v1/funnel?stage=COURT_VERDICT", headers=auth_headers)
    assert res_court.status_code == 200
    data_court = res_court.json()["funnel"]
    # Reaching trial verdict is exactly 1,397 cases out of 5,125 baseline
    assert data_court["filtered_case_count"] == 1397
    assert data_court["too_few_cases"] is False


# ==============================================================================
# 3. Consistency: Sum of per-district counts equals the total (5,125)
# ==============================================================================

def test_sum_of_per_district_counts_equals_total(api_client, auth_headers):
    """
    Verifies that the sum of reconstructed cases across all 8 districts
    equals the total unfiltered baseline of 5,125 cases exactly.
    """
    districts = [
        "D1_Suryapur",
        "D2_Chandranagar",
        "D3_Rudrapur",
        "D4_Kalyanpur",
        "D5_Mayaganj",
        "D6_Shivgarh",
        "D7_Anandpur",
        "D8_Devkalan"
    ]
    district_counts = []
    for dist in districts:
        res = api_client.get(f"/api/v1/summary?district={dist}", headers=auth_headers)
        assert res.status_code == 200
        cases = res.json()["data"]["pipeline_health"]["total_reconstructed_cases"]
        district_counts.append(cases)

    total_sum = sum(district_counts)
    assert total_sum == 5125, f"Expected 5,125 cases across all 8 districts, got {total_sum} ({district_counts})"


# ==============================================================================
# 4. Small-Cell Suppression & Too Few Cases Handling (< 5)
# ==============================================================================

def test_too_few_cases_safe_display(api_client, auth_headers):
    """
    When a filter produces < 5 cases, the API flags too_few_cases: True
    and returns friendly message / suppressed cells.
    """
    # 2023-01-01 to 2023-01-02 in D8_Devkalan matches fewer than 5 cases
    res = api_client.get(
        "/api/v1/summary?district=D8_Devkalan&from_date=2023-01-01&to_date=2023-01-02",
        headers=auth_headers
    )
    assert res.status_code == 200
    data = res.json()["data"]
    assert data.get("too_few_cases") is True
    assert "Too few cases to display safely" in data.get("message", "")
    assert data["pipeline_health"]["total_reconstructed_cases"] == "<5"


def test_funnel_small_cell_suppression(api_client, auth_headers):
    """Checks that small stages in a restricted cohort show '<5' and are marked suppressed."""
    res = api_client.get(
        "/api/v1/funnel?district=D8_Devkalan&from_date=2023-01-01&to_date=2023-01-02",
        headers=auth_headers
    )
    assert res.status_code == 200
    funnel = res.json()["funnel"]
    assert funnel.get("too_few_cases") is True
    for stage in funnel["funnel_stages"]:
        assert stage["is_k_anonymity_suppressed"] is True
        assert stage["noisy_case_count"] == "<5"


# ==============================================================================
# 5. Date Edge Cases
# ==============================================================================

def test_date_edge_cases_single_day(api_client, auth_headers):
    """Single-day filter window does not crash."""
    res = api_client.get(
        "/api/v1/summary?from_date=2024-06-15&to_date=2024-06-15",
        headers=auth_headers
    )
    assert res.status_code == 200
    assert "pipeline_health" in res.json()["data"]


def test_date_edge_cases_future_dates(api_client, auth_headers):
    """Future out-of-bounds dates safely return too_few_cases without errors."""
    res = api_client.get(
        "/api/v1/summary?from_date=2030-01-01&to_date=2030-12-31",
        headers=auth_headers
    )
    assert res.status_code == 200
    data = res.json()["data"]
    assert data.get("too_few_cases") is True


def test_date_edge_cases_inverted_dates(api_client, auth_headers):
    """Inverted dates (from > to) do not crash the service."""
    res = api_client.get(
        "/api/v1/summary?from_date=2025-01-01&to_date=2023-01-01",
        headers=auth_headers
    )
    assert res.status_code == 200
    data = res.json()["data"]
    assert data.get("too_few_cases") is True


# ==============================================================================
# 6. Privacy Budget Caching: Repeated identical filters do NOT spend extra budget
# ==============================================================================

def test_filter_cache_zero_privacy_budget_deduction(api_client, auth_headers):
    """
    Verifies that repeating the exact same filter query serves from cache
    and costs epsilon = 0.0 without extra privacy budget deduction.
    """
    # First request: computes and caches
    res1 = api_client.get(
        "/api/v1/funnel?district=D3_Rudrapur&from_date=2023-06-01&to_date=2024-06-01",
        headers=auth_headers
    )
    assert res1.status_code == 200
    rem_after_first = DATA_STORE.budget_tracker.remaining_epsilon

    # Second identical request: MUST be served from cache
    res2 = api_client.get(
        "/api/v1/funnel?district=D3_Rudrapur&from_date=2023-06-01&to_date=2024-06-01",
        headers=auth_headers
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["privacy_protection"]["cached"] is True

    # Remaining epsilon MUST NOT have decreased further
    rem_after_second = DATA_STORE.budget_tracker.remaining_epsilon
    assert rem_after_second == rem_after_first, (
        f"Cache hit spent privacy budget: {rem_after_first} -> {rem_after_second}"
    )


# ==============================================================================
# 7. Demo Safety: Configurable Budget, Admin Reset, Exhaustion & Live Diagnostics
# ==============================================================================

def test_configurable_budget_env_var(monkeypatch):
    """Verifies that total demo budget is configurable via environment variables."""
    # Test fallback default is 10.0
    monkeypatch.delenv("DP_TOTAL_BUDGET", raising=False)
    monkeypatch.delenv("TOTAL_EPSILON_BUDGET", raising=False)
    assert get_configured_total_budget() == 10.0

    # Test DP_TOTAL_BUDGET env override
    monkeypatch.setenv("DP_TOTAL_BUDGET", "25.0")
    assert get_configured_total_budget() == 25.0

    # Test TOTAL_EPSILON_BUDGET alternative
    monkeypatch.delenv("DP_TOTAL_BUDGET", raising=False)
    monkeypatch.setenv("TOTAL_EPSILON_BUDGET", "15.5")
    assert get_configured_total_budget() == 15.5


def test_admin_reset_privacy_budget_endpoint(api_client, admin_headers, auth_headers):
    """
    Verifies that POST /api/v1/privacy/budget/reset allows admins to reset the budget,
    clears filter cache, and denies access to non-admins.
    """
    # Non-admin attempt should be rejected (403)
    res_forbidden = api_client.post(
        "/api/v1/privacy/budget/reset",
        json={"total_epsilon": 20.0},
        headers=auth_headers
    )
    assert res_forbidden.status_code == 403

    # Admin reset with custom total_epsilon
    res_reset = api_client.post(
        "/api/v1/privacy/budget/reset",
        json={"total_epsilon": 20.0, "total_delta": 1e-4},
        headers=admin_headers
    )
    assert res_reset.status_code == 200
    body = res_reset.json()
    assert body["new_total_epsilon"] == 20.0
    assert body["remaining_epsilon"] == 20.0
    assert DATA_STORE.budget_tracker.remaining_epsilon == 20.0
    assert len(DATA_STORE.filter_cache) == 0

    # Admin reset with empty body defaults to configured budget
    res_reset_default = api_client.post(
        "/api/v1/privacy/budget/reset",
        json={},
        headers=admin_headers
    )
    assert res_reset_default.status_code == 200
    assert res_reset_default.json()["new_total_epsilon"] >= 10.0


def test_cached_repeated_dynamic_aggregates_spend_zero_budget(api_client, auth_headers):
    """
    Verifies that repeated dynamic aggregate queries (POST /api/v1/query/aggregate)
    hit the cache and spend 0.0 epsilon.
    """
    # Reset budget to clean state
    DATA_STORE.budget_tracker.reset(total_epsilon=10.0)
    DATA_STORE.filter_cache.clear()

    req_payload = {
        "dimension": "district",
        "metric": "count",
        "filters": {"district": "D1_Suryapur"},
        "epsilon": 0.25,
        "k_threshold": 5
    }

    # First query: uncached, allocates 0.25 epsilon
    res1 = api_client.post("/api/v1/query/aggregate", json=req_payload, headers=auth_headers)
    assert res1.status_code == 200
    body1 = res1.json()
    assert body1["epsilon_consumed"] == 0.25
    assert body1["cached"] is False
    rem_after_first = DATA_STORE.budget_tracker.remaining_epsilon
    assert rem_after_first == pytest.approx(9.75, abs=1e-4)

    # Second identical query: cached, spends 0.0 epsilon
    res2 = api_client.post("/api/v1/query/aggregate", json=req_payload, headers=auth_headers)
    assert res2.status_code == 200
    body2 = res2.json()
    assert body2["epsilon_consumed"] == 0.0
    assert body2["cached"] is True
    assert DATA_STORE.budget_tracker.remaining_epsilon == rem_after_first


def test_cached_repeated_categories_spend_zero_budget(api_client, auth_headers):
    """Verifies that GET /api/v1/categories repeated queries serve from cache with 0.0 epsilon."""
    DATA_STORE.filter_cache.clear()
    DATA_STORE.budget_tracker.reset(total_epsilon=10.0)

    # First call: uncached
    res1 = api_client.get("/api/v1/categories", headers=auth_headers)
    assert res1.status_code == 200
    rem_after_first = DATA_STORE.budget_tracker.remaining_epsilon
    assert rem_after_first == pytest.approx(9.90, abs=1e-4)

    # Second call: cached, 0.0 epsilon
    res2 = api_client.get("/api/v1/categories", headers=auth_headers)
    assert res2.status_code == 200
    assert res2.json()["privacy_metadata"]["cached"] is True
    assert DATA_STORE.budget_tracker.remaining_epsilon == rem_after_first


def test_budget_exhaustion_allows_cached_queries_and_blocks_fresh(api_client, admin_headers, auth_headers):
    """
    Verifies Requirement 4:
    When the privacy budget is exhausted (0.0 eps remaining):
    - Cached queries continue to be served with 200 OK.
    - New uncached queries are blocked with HTTP 429 Too Many Requests.
    - Admin can reset the budget to restore query access.
    """
    DATA_STORE.filter_cache.clear()

    # Pre-load cache with summary baseline (eps = 0.05) with total_epsilon = 0.05
    api_client.post(
        "/api/v1/privacy/budget/reset",
        json={"total_epsilon": 0.05},
        headers=admin_headers
    )
    assert DATA_STORE.budget_tracker.remaining_epsilon == pytest.approx(0.05, abs=1e-4)

    # First request: caches baseline summary and spends exactly 0.05
    res_cached = api_client.get("/api/v1/summary", headers=auth_headers)
    assert res_cached.status_code == 200
    assert DATA_STORE.budget_tracker.remaining_epsilon == pytest.approx(0.0, abs=1e-4)

    # Budget is now exhausted (0.00 remaining)!
    # 1. Calling the ALREADY CACHED query must still succeed with 200 OK
    res_cached_again = api_client.get("/api/v1/summary", headers=auth_headers)
    assert res_cached_again.status_code == 200
    assert res_cached_again.json()["privacy_metadata"]["cached"] is True

    # 2. Calling a NEW UNCACHED filtered query must return 429
    res_fresh = api_client.get("/api/v1/summary?district=D3_Rudrapur", headers=auth_headers)
    assert res_fresh.status_code == 429
    assert "Privacy budget exhausted" in res_fresh.json()["detail"]

    # 3. Admin resets budget
    res_reset = api_client.post(
        "/api/v1/privacy/budget/reset",
        json={"total_epsilon": 10.0},
        headers=admin_headers
    )
    assert res_reset.status_code == 200

    # 4. Now the fresh query succeeds
    res_fresh_retry = api_client.get("/api/v1/summary?district=D3_Rudrapur", headers=auth_headers)
    assert res_fresh_retry.status_code == 200


def test_privacy_diagnostics_reflects_live_budget(api_client, admin_headers):
    """Verifies that GET /api/v1/privacy/diagnostics dynamically reflects live budget tracker."""
    api_client.post(
        "/api/v1/privacy/budget/reset",
        json={"total_epsilon": 12.5},
        headers=admin_headers
    )

    res = api_client.get("/api/v1/privacy/diagnostics", headers=admin_headers)
    assert res.status_code == 200
    body = res.json()
    budget_sum = body["privacy_budget_summary"]
    assert budget_sum["total_epsilon_allocated"] == 12.5
    assert budget_sum["remaining_epsilon"] == 12.5
    assert budget_sum["spent_epsilon"] == 0.0

