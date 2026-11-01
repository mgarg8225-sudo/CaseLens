"""
tests/test_viewer_dataset_access.py - Comprehensive Unit & Integration Tests for CaseLens Viewer Dataset Access.

Verifies:
1. Cross-Session Access: An Analyst creates an analysis (sample/upload), registering it on the backend,
   and an authorized Viewer in a distinct session can access the approved aggregate results via GET /api/v1/datasets/active.
2. Available Datasets: Viewers can list approved registered datasets (GET /api/v1/datasets) and fetch specific ones (GET /api/v1/datasets/{id}).
3. Server-Side RBAC Enforcement: Viewers cannot activate or clear backend datasets (POST/DELETE /api/v1/datasets/active return 403 Forbidden).
4. Zero-PII Guarantee: Viewer dataset responses never expose raw survivor names or unmasked personal identifiers.
5. Clear Empty State: When no dataset is active or after clearing, GET /api/v1/datasets/active returns status 'empty',
   distinct from 401/403 authorization errors.
6. Dynamic Filter Recomputation: When filters change (district, date range, milestone stage),
   build_dashboard_bundle_from_analysis recomputes bundle metrics and funnel counts dynamically for Viewer dashboards.
"""

import pytest
from starlette.testclient import TestClient

from src.api import create_app, DATA_STORE, Role
from src.upload_analysis import build_dashboard_bundle_from_analysis
from app.dashboard import compute_dashboard_kpis


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
# 1. Empty State Tests
# ==============================================================================

def test_empty_state_when_no_dataset_analyzed(api_client):
    """Verifies that GET /api/v1/datasets/active returns clear 'empty' status without error."""
    # Ensure active dataset is cleared
    api_client.delete("/api/v1/datasets/active", headers=ANALYST_HEADERS)

    resp = api_client.get("/api/v1/datasets/active", headers=VIEWER_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "empty"
    assert "No dataset analyzed yet" in data["message"]
    assert data["dataset_id"] is None
    assert data["analysis_result"] is None


# ==============================================================================
# 2. Cross-Session Sharing & Viewer Access Tests
# ==============================================================================

def test_analyst_analyzes_sample_and_viewer_accesses_it(api_client):
    """Verifies that when Analyst runs sample analysis, an authorized Viewer can retrieve approved aggregate results."""
    # 1. Analyst triggers benchmark sample analysis
    analyst_run = api_client.post("/api/v1/analyze/sample?limit=50", headers=ANALYST_HEADERS)
    assert analyst_run.status_code == 200
    analyst_res = analyst_run.json()
    assert "summary_kpis" in analyst_res
    sample_total = analyst_res["summary_kpis"]["total_cases"]

    # 2. Viewer in separate session retrieves the active dataset
    viewer_resp = api_client.get("/api/v1/datasets/active", headers=VIEWER_HEADERS)
    assert viewer_resp.status_code == 200
    viewer_data = viewer_resp.json()
    assert viewer_data["status"] == "success"
    assert viewer_data["dataset_id"] == "sample"
    assert "Benchmark Sample Dataset" in viewer_data["dataset_label"]
    assert viewer_data["analysis_result"] is not None
    assert viewer_data["analysis_result"]["summary_kpis"]["total_cases"] == sample_total

    # 3. Viewer lists available datasets
    list_resp = api_client.get("/api/v1/datasets", headers=VIEWER_HEADERS)
    assert list_resp.status_code == 200
    datasets_list = list_resp.json().get("datasets", [])
    assert any(d["dataset_id"] == "sample" for d in datasets_list)

    # 4. Viewer fetches specific dataset by ID
    get_id_resp = api_client.get("/api/v1/datasets/sample", headers=VIEWER_HEADERS)
    assert get_id_resp.status_code == 200
    assert get_id_resp.json()["dataset_id"] == "sample"


# ==============================================================================
# 3. Server-Side Role Enforcement (RBAC) Tests
# ==============================================================================

def test_viewer_cannot_mutate_or_delete_backend_dataset(api_client):
    """Verifies that Viewers receive 403 Forbidden on POST and DELETE /api/v1/datasets/active."""
    # Viewer tries to switch active dataset
    switch_resp = api_client.post(
        "/api/v1/datasets/active",
        json={"dataset_id": "sample"},
        headers=VIEWER_HEADERS
    )
    assert switch_resp.status_code == 403
    assert "Forbidden" in switch_resp.json()["detail"]

    # Viewer tries to clear active dataset
    clear_resp = api_client.delete(
        "/api/v1/datasets/active",
        headers=VIEWER_HEADERS
    )
    assert clear_resp.status_code == 403
    assert "Forbidden" in clear_resp.json()["detail"]


def test_analyst_and_admin_can_switch_and_clear_dataset(api_client):
    """Verifies that Analyst and Admin can switch and clear active dataset on backend."""
    # Analyst switches
    switch_resp = api_client.post(
        "/api/v1/datasets/active",
        json={"dataset_id": "sample"},
        headers=ANALYST_HEADERS
    )
    assert switch_resp.status_code == 200
    assert switch_resp.json()["status"] == "success"

    # Admin clears
    clear_resp = api_client.delete(
        "/api/v1/datasets/active",
        headers=ADMIN_HEADERS
    )
    assert clear_resp.status_code == 200
    assert clear_resp.json()["status"] == "success"

    # Confirm empty state afterwards
    v_resp = api_client.get("/api/v1/datasets/active", headers=VIEWER_HEADERS)
    assert v_resp.json()["status"] == "empty"


# ==============================================================================
# 4. Zero-PII Compliance in Viewer Endpoints
# ==============================================================================

def test_viewer_endpoints_zero_pii_guarantee(api_client):
    """Verifies that no unmasked survivor names or PII tokens appear in dataset responses."""
    # Re-run sample analysis
    api_client.post("/api/v1/analyze/sample?limit=25", headers=ANALYST_HEADERS)

    viewer_resp = api_client.get("/api/v1/datasets/active", headers=VIEWER_HEADERS)
    assert viewer_resp.status_code == 200
    content_str = viewer_resp.text

    # Raw survivor name candidates must not be present as unmasked strings
    for sensitive_name in ["Sunita Sharma", "Anita Kumari", "Pooja Singh"]:
        assert sensitive_name not in content_str, f"PII Leakage: Found {sensitive_name} in Viewer response!"


# ==============================================================================
# 5. Dynamic Filter Recomputation Tests
# ==============================================================================

def test_dashboard_bundle_filter_recomputation(api_client):
    """Verifies that build_dashboard_bundle_from_analysis recomputes correctly on district filters."""
    # Get active dataset analysis
    resp = api_client.get("/api/v1/datasets/active", headers=VIEWER_HEADERS)
    assert resp.status_code == 200
    analysis = resp.json()["analysis_result"]
    assert analysis is not None

    # Unfiltered bundle
    bundle_all = build_dashboard_bundle_from_analysis(analysis)
    kpis_all = compute_dashboard_kpis(bundle_all)
    total_cases = int(kpis_all["reconstructed_cases_display"].replace(",", ""))
    assert total_cases > 0

    # Filtered bundle by specific district
    districts = [d["district"] for d in analysis.get("district_leakage", []) if not d.get("is_suppressed")]
    if districts:
        target_dist = districts[0]
        bundle_dist = build_dashboard_bundle_from_analysis(analysis, district=target_dist)
        kpis_dist = compute_dashboard_kpis(bundle_dist)
        dist_cases = int(kpis_dist["reconstructed_cases_display"].replace(",", ""))
        # Filtered district count should be less than or equal to total cases
        assert dist_cases <= total_cases
