"""
tests/test_auth_privacy_isolation.py - Regression & Isolation Test Suite for Auth and Privacy Separation.

Requirements Covered:
1. Generic HTTP 429 (reverse proxy, edge rate limits) is classified as rate limiting.
2. Authentication HTTP 429 (/api/v1/auth/login, /api/v1/auth/logout) is strictly classified as rate limiting, never privacy exhaustion.
3. Genuine differential privacy exhaustion (analytics endpoints with DP indicators) is correctly classified as DP budget limit.
4. Generic HTTP 429 does not set the privacy_budget_exhausted session state flag.
5. Existing differential privacy exhaustion behavior remains intact on protected analytical endpoints.
6. Invalid credentials remain rejected (HTTP 401/403) and do not touch the privacy budget.
7. Authentication login does not depend on or consume privacy budget (0 epsilon spent), even when budget is 0.
"""

import os
import json
import pytest
from starlette.testclient import TestClient

import app.api_client as api_client
from src.api import create_app, DATA_STORE
from src.auth import get_db, init_auth_db, create_user_record, close_db
import src.seed_users


class MockResponse:
    """Mock requests.Response object for unit testing response status handlers."""
    def __init__(self, status_code: int = 200, json_data=None, text: str = "", reason: str = "", url: str = ""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.reason = reason or ("OK" if status_code == 200 else "Error")
        self.url = url

    def json(self):
        if self._json_data is not None:
            return self._json_data
        raise ValueError("No JSON in response")


class MockStreamlitModule:
    """Mock Streamlit object with dict-like session_state for isolated testing."""
    def __init__(self):
        self.session_state = {}

    def warning(self, msg):
        pass

    def error(self, msg):
        pass

    def info(self, msg):
        pass

    def success(self, msg):
        pass


@pytest.fixture
def mock_st_env(monkeypatch):
    """Sets up a clean isolated mock Streamlit environment on api_client."""
    mock_st = MockStreamlitModule()
    recorded_messages = []

    def mock_show_ui(level, msg):
        recorded_messages.append({"level": level, "message": msg})

    monkeypatch.setattr(api_client, "st", mock_st)
    monkeypatch.setattr(api_client, "_HAS_STREAMLIT", True)
    monkeypatch.setattr(api_client, "_show_ui_message", mock_show_ui)

    return {
        "st": mock_st,
        "session_state": mock_st.session_state,
        "messages": recorded_messages
    }


@pytest.fixture(scope="module")
def analytical_app_env(tmp_path_factory):
    """Provides an isolated DuckDB database and TestClient for analytical DP exhaustion without user seeding."""
    mp = pytest.MonkeyPatch()
    db_file = None
    client = None
    initial_total = DATA_STORE.budget_tracker.total_epsilon
    initial_spent = DATA_STORE.budget_tracker.spent_epsilon
    orig_cache = dict(DATA_STORE.filter_cache)

    try:
        temp_dir = tmp_path_factory.mktemp("analytics_isolation_test")
        db_file = os.path.join(temp_dir, "test_analytics_isolated.duckdb")

        mp.setenv("JWT_SECRET_KEY", "super-secure-test-jwt-secret-key-32-chars-min!")
        mp.setenv("LEAKY_DB_PATH", db_file)

        # Prevent redundant institutional demo-user and judge-user seeding during lifespan startup
        mp.setattr(src.seed_users, "init_demo_users", lambda *a, **k: {})
        mp.setattr(src.seed_users, "seed_judge_users", lambda *a, **k: {})

        conn = get_db(db_file)
        init_auth_db(conn)

        app = create_app()
        DATA_STORE.budget_tracker.reset()
        DATA_STORE.filter_cache.clear()

        client = TestClient(app)
        yield {
            "client": client,
            "db_path": db_file
        }
    finally:
        if client is not None:
            client.close()
        if db_file is not None:
            try:
                close_db(db_file)
            except Exception:
                pass
        DATA_STORE.budget_tracker.total_epsilon = initial_total
        DATA_STORE.budget_tracker.spent_epsilon = initial_spent
        DATA_STORE.filter_cache.clear()
        DATA_STORE.filter_cache.update(orig_cache)
        mp.undo()


@pytest.fixture(scope="module")
def auth_app_env(tmp_path_factory):
    """Provides a separate isolated DuckDB database and TestClient with a single viewer account."""
    mp = pytest.MonkeyPatch()
    db_file = None
    initial_total = DATA_STORE.budget_tracker.total_epsilon
    initial_spent = DATA_STORE.budget_tracker.spent_epsilon

    try:
        temp_dir = tmp_path_factory.mktemp("auth_isolation_test")
        db_file = os.path.join(temp_dir, "test_auth_isolated.duckdb")

        mp.setenv("JWT_SECRET_KEY", "super-secure-test-jwt-secret-key-32-chars-min!")
        mp.setenv("LEAKY_DB_PATH", db_file)

        # Prevent redundant institutional demo-user and judge-user seeding during lifespan startup
        mp.setattr(src.seed_users, "init_demo_users", lambda *a, **k: {})
        mp.setattr(src.seed_users, "seed_judge_users", lambda *a, **k: {})

        # Initialize DuckDB schema and provision strictly the single viewer account needed
        conn = get_db(db_file)
        init_auth_db(conn)
        create_user_record("viewer", "ViewerPassword123!", "viewer", db_path=db_file)

        app = create_app()

        with TestClient(app) as client:
            yield {
                "client": client,
                "db_path": db_file
            }
    finally:
        if db_file is not None:
            try:
                close_db(db_file)
            except Exception:
                pass
        DATA_STORE.budget_tracker.total_epsilon = initial_total
        DATA_STORE.budget_tracker.spent_epsilon = initial_spent
        mp.undo()


# ==============================================================================
# Phase 3 Tests: Frontend _handle_response_status Classification
# ==============================================================================

def test_generic_429_classified_as_rate_limiting(mock_st_env):
    """Test 1: Generic 429 (e.g. Render/Cloudflare 'Too Many Requests') is classified as rate limiting."""
    resp = MockResponse(
        status_code=429,
        text="Too Many Requests",
        reason="Too Many Requests",
        url="https://caselens-dashboard.onrender.com/api/v1/summary"
    )

    result = api_client._handle_response_status(resp)
    assert result is False

    # Check notification
    messages = mock_st_env["messages"]
    assert len(messages) == 1
    assert messages[0]["level"] == "warning"
    assert "Rate Limit Exceeded" in messages[0]["message"]
    assert "Privacy Budget Limit Reached" not in messages[0]["message"]

    # Check session state flag is not set
    assert mock_st_env["session_state"].get("privacy_budget_exhausted") is not True


def test_auth_429_classified_as_rate_limiting(mock_st_env):
    """Test 2: A 429 from login or logout must never be classified as differential privacy exhaustion."""
    for auth_url in [
        "https://caselens-dashboard.onrender.com/api/v1/auth/login",
        "https://caselens-dashboard.onrender.com/api/v1/auth/logout"
    ]:
        # Case 2a: Standard auth gateway rate limit
        mock_st_env["messages"].clear()
        mock_st_env["session_state"].clear()

        resp = MockResponse(
            status_code=429,
            json_data={"detail": "Too many requests to authentication gateway. Please wait 60 seconds."},
            reason="Too Many Requests",
            url=auth_url
        )

        result = api_client._handle_response_status(resp)
        assert result is False

        messages = mock_st_env["messages"]
        assert len(messages) == 1
        assert "Rate Limit Exceeded" in messages[0]["message"]
        assert "Privacy Budget Limit Reached" not in messages[0]["message"]
        assert mock_st_env["session_state"].get("privacy_budget_exhausted") is not True

        # Case 2b: Auth 429 whose detail explicitly contains both "privacy budget" and "epsilon"
        mock_st_env["messages"].clear()
        mock_st_env["session_state"].clear()

        resp_dp = MockResponse(
            status_code=429,
            json_data={"detail": "Authentication rate limit reached: privacy budget and epsilon tokens exhausted on auth service."},
            reason="Too Many Requests",
            url=auth_url
        )

        result_dp = api_client._handle_response_status(resp_dp)
        assert result_dp is False

        messages_dp = mock_st_env["messages"]
        assert len(messages_dp) == 1
        assert "Rate Limit Exceeded" in messages_dp[0]["message"]
        assert "Privacy Budget Limit Reached" not in messages_dp[0]["message"]
        assert mock_st_env["session_state"].get("privacy_budget_exhausted") is not True
        assert "budget_exhausted_detail" not in mock_st_env["session_state"]


def test_genuine_dp_exhaustion_classified_correctly(mock_st_env):
    """Test 3: Genuine differential privacy exhaustion on analytical endpoints is classified correctly."""
    resp = MockResponse(
        status_code=429,
        json_data={"detail": "Privacy budget exhausted: remaining 0.000 epsilon < requested 0.100."},
        reason="Too Many Requests",
        url="https://caselens-dashboard.onrender.com/api/v1/funnel"
    )

    result = api_client._handle_response_status(resp)
    assert result is False

    messages = mock_st_env["messages"]
    assert len(messages) == 1
    assert "Privacy Budget Limit Reached" in messages[0]["message"]
    assert "Rate Limit Exceeded" not in messages[0]["message"]

    # Session state flag must be set
    assert mock_st_env["session_state"].get("privacy_budget_exhausted") is True
    assert "remaining 0.000 epsilon" in mock_st_env["session_state"].get("budget_exhausted_detail", "")


def test_generic_429_does_not_set_privacy_exhaustion_flag(mock_st_env):
    """Test 4: Multiple variations of generic 429 do not pollute the privacy_budget_exhausted session flag."""
    test_cases = [
        MockResponse(status_code=429, text="Too Many Requests", reason="Too Many Requests"),
        MockResponse(status_code=429, json_data={"detail": "Too Many Requests"}),
        MockResponse(status_code=429, json_data={"message": "Rate limit exceeded. Try again in 30s."}),
        MockResponse(status_code=429, json_data={"error": "Too Many Requests"}),
    ]

    mock_st_env["session_state"]["privacy_budget_exhausted"] = False

    for resp in test_cases:
        mock_st_env["messages"].clear()
        api_client._handle_response_status(resp)
        assert mock_st_env["session_state"].get("privacy_budget_exhausted") is False
        assert any("Rate Limit Exceeded" in m["message"] for m in mock_st_env["messages"])


# ==============================================================================
# Phase 3 Tests: Backend Privacy & Authentication Isolation
# ==============================================================================

def test_existing_privacy_exhaustion_behavior_intact(analytical_app_env):
    """Test 5: Protected analytical queries still return 429 when the privacy budget is exhausted."""
    client = analytical_app_env["client"]
    analyst_headers = {"X-API-Key": "analyst-key-leaky-2026"}

    # Temporarily exhaust privacy budget in data store
    orig_spent = DATA_STORE.budget_tracker.spent_epsilon
    orig_total = DATA_STORE.budget_tracker.total_epsilon

    try:
        DATA_STORE.budget_tracker.spent_epsilon = DATA_STORE.budget_tracker.total_epsilon
        assert DATA_STORE.budget_tracker.remaining_epsilon == 0.0

        # Query that demands budget must be halted with HTTP 429
        query_payload = {
            "dimension": "district",
            "epsilon": 0.25,
            "k_threshold": 5
        }
        res = client.post("/api/v1/query/aggregate", json=query_payload, headers=analyst_headers)
        assert res.status_code == 429
        assert "Privacy budget" in res.json().get("detail", "")
    finally:
        # Restore budget
        DATA_STORE.budget_tracker.spent_epsilon = orig_spent
        DATA_STORE.budget_tracker.total_epsilon = orig_total


def test_invalid_credentials_rejected(auth_app_env):
    """Test 6: Invalid credentials return HTTP 401 Unauthorized, never 429, and do not alter privacy budget."""
    client = auth_app_env["client"]
    initial_spent = DATA_STORE.budget_tracker.spent_epsilon

    res = client.post("/api/v1/auth/login", json={
        "username": "viewer",
        "password": "WrongPassword123!"
    })

    assert res.status_code == 401
    assert "Invalid username or password" in res.json().get("detail", "")
    assert DATA_STORE.budget_tracker.spent_epsilon == initial_spent


def test_login_does_not_consume_epsilon(auth_app_env):
    """Test 7: Authentication login succeeds even when privacy budget is exhausted, and spends 0.0 epsilon."""
    client = auth_app_env["client"]

    # Temporarily exhaust privacy budget
    orig_spent = DATA_STORE.budget_tracker.spent_epsilon
    orig_total = DATA_STORE.budget_tracker.total_epsilon

    try:
        DATA_STORE.budget_tracker.spent_epsilon = DATA_STORE.budget_tracker.total_epsilon
        assert DATA_STORE.budget_tracker.remaining_epsilon == 0.0

        # Login must succeed despite 0 remaining epsilon
        login_res = client.post("/api/v1/auth/login", json={
            "username": "viewer",
            "password": "ViewerPassword123!"
        })
        assert login_res.status_code == 200
        token = login_res.json().get("access_token")
        assert token is not None

        # Verify epsilon spent remains unchanged (0.0 epsilon consumed)
        assert DATA_STORE.budget_tracker.spent_epsilon == DATA_STORE.budget_tracker.total_epsilon

    finally:
        DATA_STORE.budget_tracker.spent_epsilon = orig_spent
        DATA_STORE.budget_tracker.total_epsilon = orig_total
