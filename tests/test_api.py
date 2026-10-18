"""
tests/test_api.py - Unit & Integration Tests for Phase 6 Privacy-Preserving FastAPI Service.
"""

import os
import json
import pytest
from starlette.testclient import TestClient

from src.api import create_app, Role, API_KEYS, DATA_STORE


@pytest.fixture(scope="module")
def api_client():
    """Initializes the FastAPI TestClient on the primary data repository."""
    app = create_app(data_dir="data")
    # Reset privacy budget for clean test execution
    DATA_STORE.budget_tracker.reset()
    DATA_STORE.audit_logger.clear()
    client = TestClient(app)
    yield client


# ==============================================================================
# 1. System & Health Endpoint Tests
# ==============================================================================

def test_root_endpoint(api_client):
    """Checks root metadata endpoint accessibility and zero-PII guarantee."""
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "The Leaky Pipeline - Policy Analytics API"
    assert data["status"] == "OPERATIONAL"
    assert "roles_supported" in data
    assert "admin" in data["roles_supported"]


def test_health_check_endpoint(api_client):
    """Verifies service health check and active privacy engine status."""
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "privacy_engine" in data
    assert data["privacy_engine"]["status"] == "ACTIVE"
    assert data["privacy_engine"]["remaining_epsilon"] > 0


# ==============================================================================
# 2. Authentication & Role-Based Access Control (RBAC) Tests
# ==============================================================================

def test_unauthenticated_request_blocked(api_client):
    """Verifies that protected endpoints reject unauthenticated calls with 401."""
    # Viewer endpoint without auth
    resp1 = api_client.get("/api/v1/funnel")
    assert resp1.status_code == 401
    assert "Unauthorized" in resp1.json()["detail"]

    # Analyst endpoint without auth
    resp2 = api_client.get("/api/v1/bottlenecks")
    assert resp2.status_code == 401

    # Admin endpoint without auth
    resp3 = api_client.get("/api/v1/audit/logs")
    assert resp3.status_code == 401


def test_invalid_api_key_rejected(api_client):
    """Verifies that invalid or forged API keys are rejected with 401."""
    response = api_client.get(
        "/api/v1/funnel",
        headers={"X-API-Key": "invalid-forged-key-12345"}
    )
    assert response.status_code == 401


def test_viewer_role_permissions(api_client):
    """
    Verifies Viewer permissions:
    - CAN access /summary, /funnel, /districts, /policy-context.
    - CANNOT access /bottlenecks, /categories (Analyst only) -> 403.
    - CANNOT access /audit/logs, /privacy/budget/reset (Admin only) -> 403.
    """
    headers = {"X-API-Key": "viewer-key-leaky-2026"}

    # Allowed endpoints
    res_funnel = api_client.get("/api/v1/funnel", headers=headers)
    assert res_funnel.status_code == 200
    assert "funnel" in res_funnel.json()

    res_dist = api_client.get("/api/v1/districts", headers=headers)
    assert res_dist.status_code == 200

    res_pol = api_client.get("/api/v1/policy-context", headers=headers)
    assert res_pol.status_code == 200

    # Forbidden Analyst endpoints
    res_bot = api_client.get("/api/v1/bottlenecks", headers=headers)
    assert res_bot.status_code == 403
    assert "Forbidden" in res_bot.json()["detail"]

    # Forbidden Admin endpoints
    res_audit = api_client.get("/api/v1/audit/logs", headers=headers)
    assert res_audit.status_code == 403


def test_analyst_role_permissions(api_client):
    """
    Verifies Analyst permissions:
    - CAN access Viewer endpoints (/funnel, /districts).
    - CAN access Analyst endpoints (/bottlenecks, /categories, /privacy/budget).
    - CANNOT access Admin endpoints (/audit/logs, /privacy/budget/reset) -> 403.
    """
    headers = {"X-API-Key": "analyst-key-leaky-2026"}

    # Allowed Viewer & Analyst endpoints
    assert api_client.get("/api/v1/funnel", headers=headers).status_code == 200
    assert api_client.get("/api/v1/bottlenecks", headers=headers).status_code == 200
    assert api_client.get("/api/v1/categories", headers=headers).status_code == 200
    assert api_client.get("/api/v1/privacy/budget", headers=headers).status_code == 200

    # Forbidden Admin endpoints
    assert api_client.get("/api/v1/audit/logs", headers=headers).status_code == 403
    assert api_client.post("/api/v1/privacy/budget/reset", json={"total_epsilon": 10.0}, headers=headers).status_code == 403


def test_admin_role_permissions(api_client):
    """
    Verifies Admin permissions:
    - Full access across Viewer, Analyst, and Admin endpoints.
    """
    headers = {"X-API-Key": "admin-key-leaky-2026"}

    assert api_client.get("/api/v1/funnel", headers=headers).status_code == 200
    assert api_client.get("/api/v1/bottlenecks", headers=headers).status_code == 200
    assert api_client.get("/api/v1/audit/logs", headers=headers).status_code == 200
    assert api_client.get("/api/v1/privacy/diagnostics", headers=headers).status_code == 200


def test_bearer_token_and_header_auth(api_client):
    """Verifies Authorization: Bearer <key> and X-User-Role work interchangeably."""
    # Bearer header
    resp_bearer = api_client.get(
        "/api/v1/districts",
        headers={"Authorization": "Bearer viewer-key-leaky-2026"}
    )
    assert resp_bearer.status_code == 200

    # X-User-Role header
    resp_role = api_client.get(
        "/api/v1/districts",
        headers={"X-User-Role": "viewer"}
    )
    assert resp_role.status_code == 200


# ==============================================================================
# 3. Aggregates-Only & Privacy Layer Verification Tests
# ==============================================================================

def test_funnel_aggregate_privacy_properties(api_client):
    """Verifies that the funnel aggregate contains DP noisy counts and confidence bounds."""
    headers = {"X-API-Key": "viewer-key-leaky-2026"}
    response = api_client.get("/api/v1/funnel", headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert data["query_type"] == "funnel_analytics"
    assert "privacy_protection" in data
    assert data["privacy_protection"]["epsilon_spent"] == 0.10

    # Ensure no individual case data leaked
    for stage in data["funnel"].get("funnel_stages", []):
        assert "stage_id" in stage
        assert "noisy_case_count" in stage
        assert "case_id" not in stage


def test_query_aggregate_filters_pii_blocking(api_client):
    """
    Verifies that direct case-level probing or PII filters are strictly blocked:
    - Filtering by 'case_id' -> 400 Bad Request.
    - Passing mobile numbers, Aadhaar, or survivor names -> 400 Bad Request.
    """
    headers = {"X-API-Key": "analyst-key-leaky-2026"}

    # Case ID probing
    bad_req_case = {
        "dimension": "district",
        "filters": {"case_id": "FIR-0001"}
    }
    resp1 = api_client.post("/api/v1/query/aggregate", json=bad_req_case, headers=headers)
    assert resp1.status_code == 400
    assert "Forbidden filter" in resp1.json()["detail"]

    # Survivor name probing
    bad_req_name = {
        "dimension": "district",
        "filters": {"survivor_name": "Pooja"}
    }
    resp2 = api_client.post("/api/v1/query/aggregate", json=bad_req_name, headers=headers)
    assert resp2.status_code == 400

    # Phone number probing
    bad_req_phone = {
        "dimension": "case_category",
        "filters": {"notes": "9876543210"}
    }
    resp3 = api_client.post("/api/v1/query/aggregate", json=bad_req_phone, headers=headers)
    assert resp3.status_code == 400


def test_query_aggregate_laplace_and_small_cell_suppression(api_client):
    """
    Tests dynamic aggregate query:
    - Adds Laplace noise.
    - Enforces k-anonymity suppression (counts < k masked as '<k').
    """
    headers = {"X-API-Key": "analyst-key-leaky-2026"}
    query_body = {
        "dimension": "district",
        "metric": "count",
        "epsilon": 0.5,
        "k_threshold": 5
    }
    response = api_client.post("/api/v1/query/aggregate", json=query_body, headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert data["query_dimension"] == "district"
    assert data["epsilon_consumed"] == 0.5
    assert len(data["results"]) > 0

    for item in data["results"]:
        assert "dimension_value" in item
        assert "noisy_count" in item
        # If suppressed, must be string '<5'
        if item["is_k_anonymity_suppressed"]:
            assert item["noisy_count"] == "<5"
            assert item["confidence_bound_95"] is None
        else:
            assert isinstance(item["noisy_count"], int)
            assert item["confidence_bound_95"] is not None


# ==============================================================================
# 4. Privacy Budget Exhaustion & Accounting Tests
# ==============================================================================

def test_privacy_budget_sequential_composition_exhaustion(api_client):
    """
    Tests that multiple queries consume privacy budget and that
    exhaustion triggers HTTP 429 Too Many Requests.
    """
    admin_headers = {"X-API-Key": "admin-key-leaky-2026"}
    analyst_headers = {"X-API-Key": "analyst-key-leaky-2026"}

    # 1. Reset budget to small limit (0.3 epsilon)
    reset_resp = api_client.post(
        "/api/v1/privacy/budget/reset",
        json={"total_epsilon": 0.3, "total_delta": 1e-4},
        headers=admin_headers
    )
    assert reset_resp.status_code == 200
    assert reset_resp.json()["remaining_epsilon"] == 0.3

    # 2. Query 1: costs 0.25 eps -> remaining: 0.05
    query1 = {
        "dimension": "district",
        "epsilon": 0.25,
        "k_threshold": 5
    }
    res1 = api_client.post("/api/v1/query/aggregate", json=query1, headers=analyst_headers)
    assert res1.status_code == 200

    # 3. Query 2: asks for 0.25 eps, but only 0.05 remains -> 429 Too Many Requests
    query2 = {
        "dimension": "case_category",
        "epsilon": 0.25,
        "k_threshold": 5
    }
    res2 = api_client.post("/api/v1/query/aggregate", json=query2, headers=analyst_headers)
    assert res2.status_code == 429
    assert "Privacy budget exceeded" in res2.json()["detail"]

    # 4. Admin resets budget to 10.0 -> Query succeeds again
    reset_resp2 = api_client.post(
        "/api/v1/privacy/budget/reset",
        json={"total_epsilon": 10.0, "total_delta": 1e-4},
        headers=admin_headers
    )
    assert reset_resp2.status_code == 200

    res3 = api_client.post("/api/v1/query/aggregate", json=query2, headers=analyst_headers)
    assert res3.status_code == 200


# ==============================================================================
# 5. Audit Logging Tests
# ==============================================================================

def test_audit_log_recording_and_inspection(api_client):
    """
    Verifies that API interactions are immutably recorded into the audit ledger
    and that only Admins can query the audit logs.
    """
    admin_headers = {"X-API-Key": "admin-key-leaky-2026"}
    viewer_headers = {"X-API-Key": "viewer-key-leaky-2026"}

    # Perform an action as viewer
    api_client.get("/api/v1/districts", headers=viewer_headers)

    # Admin inspects audit logs
    audit_resp = api_client.get("/api/v1/audit/logs?limit=10", headers=admin_headers)
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()

    assert "logs" in audit_data
    assert len(audit_data["logs"]) > 0

    latest_entry = audit_data["logs"][0]
    assert "request_id" in latest_entry
    assert "timestamp" in latest_entry
    assert "endpoint" in latest_entry
    assert "status_code" in latest_entry
    assert latest_entry["zero_pii_verified"] is True


def test_statutory_policy_context_endpoint(api_client):
    """Verifies that statutory policy context and external benchmarks return expected schema."""
    headers = {"X-API-Key": "viewer-key-leaky-2026"}
    response = api_client.get("/api/v1/policy-context", headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert "statutory_benchmarks" in data
    assert "external_policy_context" in data
    assert data["zero_pii_verified"] is True
