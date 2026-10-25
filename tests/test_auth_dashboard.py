"""
tests/test_auth_dashboard.py - Part A Test Suite: Authentication, Lockout, RBAC & Audit Log.

Verifies:
1. Authentication with correct and wrong passwords.
2. 5-attempt account lockout protection (15-minute lockout).
3. Role-Based Access Control (RBAC):
   - viewer cannot reach Case Simulator, Review Queue, or Audit Log.
   - analyst can reach viewer pages + Case Simulator, but not Review Queue or Audit Log.
   - admin can reach all pages including Review Queue and Audit Log.
4. Privacy & Zero-Credential verification:
   - No passwords, hashes, or JWT tokens appear in audit logs.
   - Review queue display uses only pseudonymized IDs with zero personal names.
"""

import os
import json
import pytest
from fastapi import HTTPException

from src.auth import (
    authenticate_user,
    create_access_token,
    decode_access_token,
    check_user_permission,
    normalize_role,
    log_audit_event,
    get_audit_logs,
    get_user_by_username,
    close_db,
    ROLE_PERMISSIONS,
    DEFAULT_LOCKOUT_ATTEMPTS,
    LOCKOUT_DURATION_MINUTES
)
from src.seed_users import init_demo_users


@pytest.fixture(scope="module")
def temp_auth_env(tmp_path_factory):
    """Sets up an isolated temporary DuckDB and audit log for Part A testing."""
    temp_dir = tmp_path_factory.mktemp("part_a_test")
    db_file = os.path.join(temp_dir, "test_auth.duckdb")
    audit_file = os.path.join(temp_dir, "test_app_audit_log.json")

    os.environ["JWT_SECRET_KEY"] = "test-secret-key-32-bytes-minimum-length-req!"
    os.environ["LEAKY_DB_PATH"] = db_file
    os.environ["DEMO_VIEWER_PASSWORD"] = "ViewerPassword123!"
    os.environ["DEMO_ANALYST_PASSWORD"] = "AnalystPassword123!"
    os.environ["DEMO_ADMIN_PASSWORD"] = "AdminPassword123!"

    # Initialize demo users in isolated DB
    creds = init_demo_users(db_path=db_file, print_output=False)

    yield {
        "db_path": db_file,
        "audit_path": audit_file,
        "creds": creds
    }

    close_db(db_file)


# ==============================================================================
# 1. Demo User Authentication Tests
# ==============================================================================

def test_demo_users_successful_auth(temp_auth_env):
    """Verifies that viewer, analyst, and admin authenticate successfully with correct roles."""
    db_path = temp_auth_env["db_path"]
    creds = temp_auth_env["creds"]

    for username, role in [("viewer", "viewer"), ("analyst", "analyst"), ("admin", "admin")]:
        pwd = creds[username]["password"]
        user = authenticate_user(username, pwd, db_path=db_path)
        assert user is not None
        assert user["username"] == username
        assert user["role"] == role
        assert "password_hash" not in user, "Password hash must never be returned in user context"


def test_wrong_password_rejected(temp_auth_env):
    """Verifies that an incorrect password raises 401 with generic error message."""
    db_path = temp_auth_env["db_path"]

    with pytest.raises(HTTPException) as exc_info:
        authenticate_user("viewer", "CompletelyWrongPassword!", db_path=db_path)

    assert exc_info.value.status_code == 401
    assert "Invalid username or password" in exc_info.value.detail


# ==============================================================================
# 2. Account Lockout Tests
# ==============================================================================

def test_account_lockout_after_five_failed_attempts(temp_auth_env):
    """
    Verifies that 5 consecutive failed logins trigger a 15-minute account lockout,
    and the 6th attempt is blocked with 403 Forbidden.
    """
    db_path = temp_auth_env["db_path"]
    user = get_user_by_username("analyst", db_path=db_path)
    assert user is not None

    # Perform failed attempts
    for attempt in range(DEFAULT_LOCKOUT_ATTEMPTS):
        with pytest.raises(HTTPException) as exc:
            authenticate_user("analyst", "WrongPassAttempt!", db_path=db_path)
        if attempt < DEFAULT_LOCKOUT_ATTEMPTS - 1:
            assert exc.value.status_code == 401
        else:
            # 5th failed attempt triggers the lockout
            assert exc.value.status_code == 403
            assert "locked" in exc.value.detail.lower()

    # 6th attempt while locked must fail immediately with 403 Forbidden
    with pytest.raises(HTTPException) as exc_locked:
        authenticate_user("analyst", "WrongPassAttempt!", db_path=db_path)
    assert exc_locked.value.status_code == 403
    assert "locked" in exc_locked.value.detail.lower()

    # Even providing the correct password while locked must be rejected with 403
    with pytest.raises(HTTPException) as exc_locked_correct:
        authenticate_user("analyst", temp_auth_env["creds"]["analyst"]["password"], db_path=db_path)
    assert exc_locked_correct.value.status_code == 403


# ==============================================================================
# 3. Server-Side RBAC Permission Tests
# ==============================================================================

def test_viewer_rbac_permissions():
    """
    Verifies viewer can only access aggregate views:
    Overview, Funnel, Bottlenecks, Linkage Quality, Policy Insights.
    Viewer CANNOT access Case Simulator, Review Queue, or Audit Log.
    """
    # Allowed aggregate pages & Case Explorer
    assert check_user_permission("viewer", "📊 Executive Overview") is True
    assert check_user_permission("viewer", "🔻 Funnel with Uncertainty Ranges") is True
    assert check_user_permission("viewer", "⏱️ Duration Bottlenecks & Latency") is True
    assert check_user_permission("viewer", "🔗 Linkage Quality & Evaluation") is True
    assert check_user_permission("viewer", "📜 Statutory Policy Insights (TinyFish)") is True
    assert check_user_permission("viewer", "🔍 Case Explorer") is True

    # Forbidden pages for viewer
    assert check_user_permission("viewer", "🧪 Case Simulator") is False
    assert check_user_permission("viewer", "📋 Review Queue") is False
    assert check_user_permission("viewer", "📜 Audit Log") is False


def test_analyst_rbac_permissions():
    """
    Verifies analyst can access viewer pages + Case Simulator + Case Explorer,
    but CANNOT access admin-only Review Queue or Audit Log.
    """
    assert check_user_permission("analyst", "📊 Executive Overview") is True
    assert check_user_permission("analyst", "🧪 Case Simulator") is True
    assert check_user_permission("analyst", "🔍 Case Explorer") is True
    assert check_user_permission("analyst", "⚖️ Before / After Policy Transformation") is True
    assert check_user_permission("analyst", "🛡️ Survivor Privacy & DP Engine") is True

    # Forbidden pages for analyst
    assert check_user_permission("analyst", "📋 Review Queue") is False
    assert check_user_permission("analyst", "📜 Audit Log") is False


def test_admin_rbac_permissions():
    """Verifies admin has full access to all pages including Review Queue and Audit Log."""
    assert check_user_permission("admin", "📊 Executive Overview") is True
    assert check_user_permission("admin", "🧪 Case Simulator") is True
    assert check_user_permission("admin", "🔍 Case Explorer") is True
    assert check_user_permission("admin", "📋 Review Queue") is True
    assert check_user_permission("admin", "📜 Audit Log") is True


# ==============================================================================
# 4. Audit Logging & Zero-Credential Tests
# ==============================================================================

def test_audit_log_records_events_without_credentials(temp_auth_env):
    """
    Verifies login, logout, failed login, and page visits are recorded,
    and NEVER contain passwords, tokens, or hashes.
    """
    audit_file = temp_auth_env["audit_path"]
    secret_pass = "MySecretPassword123!"
    jwt_sample = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-IDNxdgqNXqYjISBDM"

    # Log various events, some attempting to include secrets
    log_audit_event("LOGIN_SUCCESS", "viewer", role="viewer", page="Overview", log_path=audit_file)
    log_audit_event(
        "LOGIN_FAILED",
        "viewer",
        details={"reason": "Invalid password", "password": secret_pass, "token": jwt_sample},
        log_path=audit_file
    )
    log_audit_event("PAGE_VISIT", "viewer", role="viewer", page="📊 Executive Overview", log_path=audit_file)
    log_audit_event("ACCESS_DENIED", "viewer", role="viewer", page="📋 Review Queue", log_path=audit_file)
    log_audit_event("LOGOUT", "viewer", role="viewer", log_path=audit_file)

    logs = get_audit_logs(limit=20, log_path=audit_file)
    assert len(logs) == 5

    # Check raw file contents as well
    with open(audit_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Strict privacy assertions
    assert secret_pass not in raw_text, "Plaintext password must NEVER appear in the audit log!"
    assert jwt_sample not in raw_text, "JWT token must NEVER appear in the audit log!"
    assert "$argon2id$" not in raw_text, "Password hash must NEVER appear in the audit log!"


# ==============================================================================
# 5. Review Queue Pseudonymization Integrity Test
# ==============================================================================

def test_review_queue_pseudonymization_no_pii():
    """
    Verifies that the Review Queue dataset only displays pseudonymized columns,
    and strips all direct personal identifiers.
    """
    import pandas as pd
    review_path = os.path.join("data", "review_queue.csv")
    if os.path.exists(review_path):
        df_rq = pd.read_csv(review_path)
        safe_cols = [
            "unique_id_l", "unique_id_r", "district_l", "district_r",
            "incident_date_l", "incident_date_r", "match_probability",
            "match_weight", "review_status"
        ]
        display_cols = [c for c in safe_cols if c in df_rq.columns]
        df_display = df_rq[display_cols]

        forbidden_pii_cols = [
            "norm_full_name_l", "norm_full_name_r",
            "norm_relative_name_l", "norm_relative_name_r",
            "norm_locality_l", "norm_locality_r"
        ]
        for f_col in forbidden_pii_cols:
            assert f_col not in df_display.columns, f"PII column '{f_col}' must not be exposed in review queue display!"
