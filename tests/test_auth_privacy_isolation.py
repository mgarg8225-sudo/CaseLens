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


# ==============================================================================
# 8. Audit Log Endpoint Security & Anti-Spoofing Tests (Phase 2)
# ==============================================================================

def test_audit_log_endpoint_unauthenticated_rejected(auth_app_env):
    """
    Test 8a: Unauthenticated audit log submission must be rejected with HTTP 401.
    Verifies that authentication failures create ZERO audit records in the ledger.
    """
    client = auth_app_env["client"]

    initial_probe_events = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("event_type") == "UNAUTHENTICATED_PROBE_ATTEMPT"
    ])

    # 1. Missing Authorization header
    res_no_auth = client.post("/api/v1/audit/log", json={
        "event_type": "UNAUTHENTICATED_PROBE_ATTEMPT",
        "page": "📊 Executive Overview",
        "user_id": "attacker",
        "role": "admin"
    })
    assert res_no_auth.status_code == 401

    # 2. Invalid / forged Bearer token
    res_invalid_jwt = client.post(
        "/api/v1/audit/log",
        json={"event_type": "UNAUTHENTICATED_PROBE_ATTEMPT", "page": "Dashboard"},
        headers={"Authorization": "Bearer forged.invalid.token"}
    )
    assert res_invalid_jwt.status_code == 401

    # 3. Invalid API key
    res_invalid_key = client.post(
        "/api/v1/audit/log",
        json={"event_type": "UNAUTHENTICATED_PROBE_ATTEMPT", "page": "Dashboard"},
        headers={"X-API-Key": "invalid-random-key-9999"}
    )
    assert res_invalid_key.status_code == 401

    # 4. Critical requirement: Authentication failures must create ZERO audit entries from the endpoint
    final_probe_events = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("event_type") == "UNAUTHENTICATED_PROBE_ATTEMPT"
    ])
    assert final_probe_events == initial_probe_events == 0


def test_audit_log_endpoint_anti_spoofing_and_jwt_identity_fidelity(auth_app_env):
    """
    Test 8b: Authenticated audit submission derives actor identity strictly from verified JWT.
    Verifies:
    1. Forged request-body user_id cannot impersonate another user.
    2. Forged request-body role cannot grant elevated privileges in audit logs.
    3. Credentials and tokens in details are safely redacted.
    """
    client = auth_app_env["client"]

    # Log in as legitimate viewer to obtain genuine signed JWT
    login_res = client.post("/api/v1/auth/login", json={
        "username": "viewer",
        "password": "ViewerPassword123!"
    })
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    user_data = login_res.json()["user"]
    verified_user_id = user_data["id"]
    headers = {"Authorization": f"Bearer {token}"}

    # Submit audit event attempting to spoof actor identity as "admin"
    res = client.post(
        "/api/v1/audit/log",
        json={
            "event_type": "MALICIOUS_IMPERSONATION_ATTEMPT",
            "page": "Admin Panel",
            "user_id": "target_admin_victim_99",
            "role": "admin",
            "details": {
                "action": "privilege_escalation",
                "auth_token": "secret_session_bearer_token",
                "api_key": "leaked_master_admin_api_key",
                "normal_param": "benchmark_run"
            }
        },
        headers=headers
    )
    assert res.status_code == 200
    assert res.json()["status"] == "recorded"
    assert res.json()["event_type"] == "MALICIOUS_IMPERSONATION_ATTEMPT"

    # Inspect recorded log in DATA_STORE.audit_logger matching the custom event
    matching = [e for e in DATA_STORE.audit_logger.get_logs(limit=20) if e.get("event_type") == "MALICIOUS_IMPERSONATION_ATTEMPT"]
    assert len(matching) >= 1
    recent_entry = matching[0]

    # Verify actor identity matches verified JWT, NOT the forged body fields
    assert recent_entry["user_id"] == verified_user_id
    assert recent_entry["user_id"] != "target_admin_victim_99"
    assert recent_entry["role"] == "viewer"
    assert recent_entry["role"] != "admin"

    # Verify spoofing attempt was detected and marked
    assert recent_entry["details"]["client_supplied_user_id"] == "[IGNORED_MISMATCH]"
    assert recent_entry["details"]["client_supplied_role"] == "[IGNORED_MISMATCH]"

    # Verify credentials in details were redacted
    assert recent_entry["details"]["auth_token"] == "[REDACTED]"
    assert recent_entry["details"]["api_key"] == "[REDACTED]"
    assert recent_entry["details"]["normal_param"] == "benchmark_run"


def test_audit_log_endpoint_authorized_legitimate_event(auth_app_env):
    """
    Test 8c: Authorized user can submit legitimate audit event without spoofed fields.
    """
    client = auth_app_env["client"]

    # Log in as legitimate viewer
    login_res = client.post("/api/v1/auth/login", json={
        "username": "viewer",
        "password": "ViewerPassword123!"
    })
    token = login_res.json()["access_token"]
    verified_user_id = login_res.json()["user"]["id"]
    headers = {"Authorization": f"Bearer {token}"}

    res = client.post(
        "/api/v1/audit/log",
        json={
            "event_type": "PAGE_NAVIGATION",
            "page": "Case Explorer",
            "details": {"action": "tab_switch", "target_tab": "Timeline"}
        },
        headers=headers
    )
    assert res.status_code == 200
    assert res.json() == {"status": "recorded", "event_type": "PAGE_NAVIGATION"}

    matching = [e for e in DATA_STORE.audit_logger.get_logs(limit=20) if e.get("event_type") == "PAGE_NAVIGATION"]
    assert len(matching) >= 1
    entry = matching[0]
    assert entry["user_id"] == verified_user_id
    assert entry["role"] == "viewer"
    assert entry["event_type"] == "PAGE_NAVIGATION"
    assert entry["page"] == "Case Explorer"
    assert entry["details"]["action"] == "tab_switch"


def test_audit_log_endpoint_recursive_sanitization_nested_and_lists(auth_app_env):
    """
    Test 8d: Sensitive fields are redacted recursively, including nested dictionaries and lists,
    while harmless details remain intact.
    """
    client = auth_app_env["client"]

    login_res = client.post("/api/v1/auth/login", json={
        "username": "viewer",
        "password": "ViewerPassword123!"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    nested_payload = {
        "event_type": "DATA_EXPORT",
        "page": "Reports",
        "details": {
            "session": {
                "auth_token": "secret-nested-token-999",
                "ordinary_session_id": "sess-xyz-123"
            },
            "credentials_list": [
                {"access_key": "raw-access-key-val", "label": "primary"},
                {"user_password": "super-secret-password-123", "role": "viewer"}
            ],
            "metrics": {
                "rows_exported": 50,
                "filters_applied": ["district_A", "2026"]
            }
        }
    }

    res = client.post("/api/v1/audit/log", json=nested_payload, headers=headers)
    assert res.status_code == 200

    matching = [e for e in DATA_STORE.audit_logger.get_logs(limit=20) if e.get("event_type") == "DATA_EXPORT"]
    assert len(matching) >= 1
    entry = matching[0]
    details = entry["details"]

    # Nested dictionary redaction
    assert details["session"]["auth_token"] == "[REDACTED]"
    assert details["session"]["ordinary_session_id"] == "sess-xyz-123"

    # List of dictionaries redaction
    assert details["credentials_list"][0]["access_key"] == "[REDACTED]"
    assert details["credentials_list"][0]["label"] == "primary"
    assert details["credentials_list"][1]["user_password"] == "[REDACTED]"
    assert details["credentials_list"][1]["role"] == "viewer"

    # Harmless details preservation
    assert details["metrics"]["rows_exported"] == 50
    assert details["metrics"]["filters_applied"] == ["district_A", "2026"]


def test_audit_log_endpoint_institutional_api_key_compatible(auth_app_env):
    """
    Test 8e: Institutional API-key authentication is compatible with POST /api/v1/audit/log.
    """
    client = auth_app_env["client"]

    # Use institutional analyst API key
    analyst_headers = {"X-API-Key": "analyst-key-leaky-2026"}

    res = client.post(
        "/api/v1/audit/log",
        json={
            "event_type": "ANALYST_QUERY",
            "page": "Funnel Analysis",
            "user_id": "spoofed_admin",
            "role": "admin",
            "details": {"query": "district_leakage"}
        },
        headers=analyst_headers
    )
    assert res.status_code == 200

    matching = [e for e in DATA_STORE.audit_logger.get_logs(limit=20) if e.get("event_type") == "ANALYST_QUERY"]
    assert len(matching) >= 1
    entry = matching[0]
    assert entry["user_id"] == "key_analyst"
    assert entry["role"] == "analyst"
    assert entry["details"]["client_supplied_user_id"] == "[IGNORED_MISMATCH]"
    assert entry["details"]["client_supplied_role"] == "[IGNORED_MISMATCH]"
    assert entry["details"]["query"] == "district_leakage"


def test_audit_log_endpoint_key_normalization_variations(auth_app_env):
    """
    Test 8f (Phase 2.1): Normalized sensitive key variations are securely redacted.
    Covers:
    - apiKey, api_key, apikey, api-key, API_KEY
    - accessKey, access_key, Access-Key
    - accessToken, access_token
    - clientSecret, client_secret
    - password, passwd, user_password
    - authorization, credential
    """
    client = auth_app_env["client"]
    login_res = client.post("/api/v1/auth/login", json={
        "username": "viewer",
        "password": "ViewerPassword123!"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    variations_payload = {
        "event_type": "KEY_VARIATIONS_CHECK",
        "page": "Security Test",
        "details": {
            "apiKey": "raw-val-1",
            "api_key": "raw-val-2",
            "apikey": "raw-val-3",
            "api-key": "raw-val-4",
            "API_KEY": "raw-val-5",
            "accessKey": "raw-val-6",
            "access_key": "raw-val-7",
            "Access-Key": "raw-val-8",
            "accessToken": "raw-val-9",
            "access_token": "raw-val-10",
            "clientSecret": "raw-val-11",
            "client_secret": "raw-val-12",
            "password": "raw-val-13",
            "passwd": "raw-val-14",
            "user_password": "raw-val-15",
            "authorization": "Bearer raw-val-16",
            "credential": "raw-val-17",
            "harmless_metric": 42,
            "harmless_label": "benchmark"
        }
    }

    res = client.post("/api/v1/audit/log", json=variations_payload, headers=headers)
    assert res.status_code == 200

    matching = [e for e in DATA_STORE.audit_logger.get_logs(limit=20) if e.get("event_type") == "KEY_VARIATIONS_CHECK"]
    assert len(matching) >= 1
    details = matching[0]["details"]

    sensitive_keys = [
        "apiKey", "api_key", "apikey", "api-key", "API_KEY",
        "accessKey", "access_key", "Access-Key",
        "accessToken", "access_token",
        "clientSecret", "client_secret",
        "password", "passwd", "user_password",
        "authorization", "credential"
    ]
    for sk in sensitive_keys:
        assert details[sk] == "[REDACTED]", f"Expected {sk} to be [REDACTED], got {details[sk]}"

    assert details["harmless_metric"] == 42
    assert details["harmless_label"] == "benchmark"


def test_audit_log_endpoint_nested_identity_protection(auth_app_env):
    """
    Test 8g (Phase 2.1): Client-supplied identity and role values inside arbitrary details
    cannot override, impersonate, or be mistaken for verified audit actor identity.
    """
    client = auth_app_env["client"]
    login_res = client.post("/api/v1/auth/login", json={
        "username": "viewer",
        "password": "ViewerPassword123!"
    })
    token = login_res.json()["access_token"]
    verified_user_id = login_res.json()["user"]["id"]
    headers = {"Authorization": f"Bearer {token}"}

    identity_spoof_payload = {
        "event_type": "IDENTITY_PROTECTION_CHECK",
        "page": "Admin Gateway",
        "details": {
            "user_id": "impersonated_admin_user",
            "actor_id": "root_system_admin",
            "email": "victim_admin@gov.in",
            "role": "admin",
            "query_target": "district_leakage",
            "sub_context": {
                "user_id": "nested_attacker",
                "email": "nested_victim@gov.in",
                "action": "deep_inspect"
            }
        }
    }

    res = client.post("/api/v1/audit/log", json=identity_spoof_payload, headers=headers)
    assert res.status_code == 200

    matching = [e for e in DATA_STORE.audit_logger.get_logs(limit=20) if e.get("event_type") == "IDENTITY_PROTECTION_CHECK"]
    assert len(matching) >= 1
    entry = matching[0]

    # Authoritative top-level actor identity MUST be the verified JWT caller
    assert entry["user_id"] == verified_user_id
    assert entry["user_id"] != "impersonated_admin_user"
    assert entry["role"] == "viewer"
    assert entry["role"] != "admin"

    # Nested identity fields in details must be sanitized to prevent mistaking them for verified actors
    details = entry["details"]
    assert details["user_id"] == "[UNVERIFIED_CLIENT_IDENTITY]"
    assert details["actor_id"] == "[UNVERIFIED_CLIENT_IDENTITY]"
    assert details["email"] == "[UNVERIFIED_CLIENT_IDENTITY]"
    assert details["role"] == "[UNVERIFIED_CLIENT_ROLE]"
    assert details["query_target"] == "district_leakage"

    sub_ctx = details["sub_context"]
    assert sub_ctx["user_id"] == "[UNVERIFIED_CLIENT_IDENTITY]"
    assert sub_ctx["email"] == "[UNVERIFIED_CLIENT_IDENTITY]"
    assert sub_ctx["action"] == "deep_inspect"


def test_audit_log_endpoint_bounds_payload_size_limit(auth_app_env):
    """
    Test 8h (Phase 2.1): Submissions exceeding maximum serialized details size (16 KB)
    are rejected with HTTP 400 Bad Request and create ZERO audit log entries.
    """
    client = auth_app_env["client"]
    login_res = client.post("/api/v1/auth/login", json={
        "username": "viewer",
        "password": "ViewerPassword123!"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    initial_audit_events = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("endpoint", "").startswith("/audit/event")
    ])
    initial_matching = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("event_type") == "OVERSIZED_SIZE_CHECK"
    ])

    # Generate a payload whose details serialize to > 16 KB (e.g. ~18 KB)
    oversized_details = {"large_array": ["x" * 100 for _ in range(180)]}
    res = client.post(
        "/api/v1/audit/log",
        json={"event_type": "OVERSIZED_SIZE_CHECK", "details": oversized_details},
        headers=headers
    )
    assert res.status_code == 400
    assert "16 KB" in res.json().get("detail", "") or "serialized size" in res.json().get("detail", "")

    final_audit_events = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("endpoint", "").startswith("/audit/event")
    ])
    final_matching = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("event_type") == "OVERSIZED_SIZE_CHECK"
    ])
    assert final_audit_events == initial_audit_events
    assert final_matching == initial_matching == 0


def test_audit_log_endpoint_bounds_nesting_depth_limit(auth_app_env):
    """
    Test 8i (Phase 2.1): Submissions exceeding maximum nesting depth (8)
    are rejected with HTTP 400 Bad Request and create ZERO audit log entries.
    """
    client = auth_app_env["client"]
    login_res = client.post("/api/v1/auth/login", json={
        "username": "viewer",
        "password": "ViewerPassword123!"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    initial_audit_events = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("endpoint", "").startswith("/audit/event")
    ])
    initial_matching = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("event_type") == "OVERSIZED_DEPTH_CHECK"
    ])

    # Depth 9
    deep_details = {}
    curr = deep_details
    for i in range(8):
        curr["level"] = {}
        curr = curr["level"]
    curr["val"] = "too_deep"

    res = client.post(
        "/api/v1/audit/log",
        json={"event_type": "OVERSIZED_DEPTH_CHECK", "details": deep_details},
        headers=headers
    )
    assert res.status_code == 400
    assert "nesting depth" in res.json().get("detail", "")

    final_audit_events = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("endpoint", "").startswith("/audit/event")
    ])
    final_matching = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("event_type") == "OVERSIZED_DEPTH_CHECK"
    ])
    assert final_audit_events == initial_audit_events
    assert final_matching == initial_matching == 0


def test_audit_log_endpoint_bounds_total_entries_limit(auth_app_env):
    """
    Test 8j (Phase 2.1): Submissions exceeding maximum entries limit (200)
    are rejected with HTTP 400 Bad Request and create ZERO audit log entries.
    """
    client = auth_app_env["client"]
    login_res = client.post("/api/v1/auth/login", json={
        "username": "viewer",
        "password": "ViewerPassword123!"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    initial_audit_events = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("endpoint", "").startswith("/audit/event")
    ])
    initial_matching = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("event_type") == "OVERSIZED_ENTRIES_CHECK"
    ])

    # 201 entries
    many_entries = {f"key_{i}": f"val_{i}" for i in range(201)}

    res = client.post(
        "/api/v1/audit/log",
        json={"event_type": "OVERSIZED_ENTRIES_CHECK", "details": many_entries},
        headers=headers
    )
    assert res.status_code == 400
    assert "total entries limit" in res.json().get("detail", "")

    final_audit_events = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("endpoint", "").startswith("/audit/event")
    ])
    final_matching = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("event_type") == "OVERSIZED_ENTRIES_CHECK"
    ])
    assert final_audit_events == initial_audit_events
    assert final_matching == initial_matching == 0


def test_audit_log_endpoint_bounds_string_length_limit(auth_app_env):
    """
    Test 8k (Phase 2.1): Submissions containing strings exceeding 2,000 characters
    are rejected with HTTP 400 Bad Request and create ZERO audit log entries.
    """
    client = auth_app_env["client"]
    login_res = client.post("/api/v1/auth/login", json={
        "username": "viewer",
        "password": "ViewerPassword123!"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    initial_audit_events = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("endpoint", "").startswith("/audit/event")
    ])
    initial_matching = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("event_type") == "LONG_STRING_CHECK"
    ])

    long_string_details = {"normal_key": "x" * 2001}

    res = client.post(
        "/api/v1/audit/log",
        json={"event_type": "LONG_STRING_CHECK", "details": long_string_details},
        headers=headers
    )
    assert res.status_code == 400
    assert "maximum length" in res.json().get("detail", "")

    final_audit_events = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("endpoint", "").startswith("/audit/event")
    ])
    final_matching = len([
        e for e in DATA_STORE.audit_logger.get_logs(limit=1000)
        if e.get("event_type") == "LONG_STRING_CHECK"
    ])
    assert final_audit_events == initial_audit_events
    assert final_matching == initial_matching == 0
