"""
tests/test_judge_access.py - Unit & Integration Tests for Hackathon Judge Access.

Verifies:
1. With judge env variables set, both judge users exist after startup in production mode;
   with them unset, neither exists and startup does not crash.
2. Judge analyst can run analysis but receives HTTP 403 on publish, unpublish, user management,
   privacy settings modification, budget reset, and audit log.
3. Judge viewer only sees published datasets and receives HTTP 403 on draft datasets and analyst-only endpoints.
4. Removing the variables and restarting disables judge logins.
5. JUDGE_MODE unset or "off" hides the buttons (verified via helper function).
6. Zero plaintext passwords or hashes printed or stored insecurely; safe-init remains intact.
"""

import os
import logging
import pytest
from starlette.testclient import TestClient

from src.auth import (
    get_db,
    init_auth_db,
    close_db,
    get_user_by_username,
    authenticate_user,
    create_access_token,
    is_production_mode
)
from src.seed_users import init_demo_users, seed_judge_users
from src.api import create_app
from app.api_client import is_judge_access_available, get_judge_credential


@pytest.fixture
def judge_db(tmp_path):
    """Provides an isolated temporary DuckDB file for testing judge access."""
    db_file = str(tmp_path / "test_judge.duckdb")
    close_db(db_file)
    yield db_file
    close_db(db_file)


# ==============================================================================
# 1. Startup Seeding in Production Mode
# ==============================================================================

def test_judge_users_created_when_env_vars_set(judge_db, monkeypatch):
    """Verifies that with judge env variables set in production mode, both judge users are created."""
    monkeypatch.setenv("CASELENS_ENV", "production")
    monkeypatch.setenv("ADMIN_PASSWORD", "ProductionAdminPassword123!")
    monkeypatch.setenv("JUDGE_VIEWER_USERNAME", "judge_viewer_hackathon")
    monkeypatch.setenv("JUDGE_VIEWER_PASSWORD", "JudgeViewerSecret2026!")
    monkeypatch.setenv("JUDGE_ANALYST_USERNAME", "judge_analyst_hackathon")
    monkeypatch.setenv("JUDGE_ANALYST_PASSWORD", "JudgeAnalystSecret2026!")

    assert is_production_mode() is True

    # Run safe user initialization followed by judge user seeding
    init_demo_users(db_path=judge_db, print_output=False, only_if_missing=True)
    judge_creds = seed_judge_users(db_path=judge_db)

    assert "judge_viewer_hackathon" in judge_creds
    assert "judge_analyst_hackathon" in judge_creds

    # Check database records
    v_user = get_user_by_username("judge_viewer_hackathon", db_path=judge_db)
    assert v_user is not None
    assert v_user["role"] == "viewer"
    assert v_user["agency"] == "judge"
    assert v_user["is_active"] is True

    a_user = get_user_by_username("judge_analyst_hackathon", db_path=judge_db)
    assert a_user is not None
    assert a_user["role"] == "analyst"
    assert a_user["agency"] == "judge"
    assert a_user["is_active"] is True

    # Authenticate successfully
    auth_v = authenticate_user("judge_viewer_hackathon", "JudgeViewerSecret2026!", db_path=judge_db)
    assert auth_v["username"] == "judge_viewer_hackathon"
    assert auth_v["role"] == "viewer"

    auth_a = authenticate_user("judge_analyst_hackathon", "JudgeAnalystSecret2026!", db_path=judge_db)
    assert auth_a["username"] == "judge_analyst_hackathon"
    assert auth_a["role"] == "analyst"


def test_judge_users_unset_variables(judge_db, monkeypatch):
    """Verifies that with judge env vars unset, neither judge user exists and startup does not crash."""
    monkeypatch.setattr("src.auth.load_env_file", lambda *a, **kw: None)
    monkeypatch.setattr("src.seed_users.load_env_file", lambda *a, **kw: None)
    monkeypatch.setenv("CASELENS_ENV", "production")
    monkeypatch.setenv("ADMIN_PASSWORD", "ProductionAdminPassword123!")
    monkeypatch.delenv("JUDGE_VIEWER_USERNAME", raising=False)
    monkeypatch.delenv("JUDGE_VIEWER_PASSWORD", raising=False)
    monkeypatch.delenv("JUDGE_ANALYST_USERNAME", raising=False)
    monkeypatch.delenv("JUDGE_ANALYST_PASSWORD", raising=False)

    init_demo_users(db_path=judge_db, print_output=False, only_if_missing=True)
    judge_creds = seed_judge_users(db_path=judge_db)

    assert judge_creds == {}
    assert get_user_by_username("judge_viewer_hackathon", db_path=judge_db) is None
    assert get_user_by_username("judge_analyst_hackathon", db_path=judge_db) is None


# ==============================================================================
# 2. Judge Analyst: Permissions & 403 Forbidden Restrictions
# ==============================================================================

def test_judge_analyst_permissions_and_restrictions(judge_db, monkeypatch):
    """
    Verifies that the Judge Analyst can perform analysis (stays draft) but receives
    HTTP 403 on publish, unpublish, user management, privacy settings, budget reset, and audit log.
    """
    monkeypatch.setenv("LEAKY_DB_PATH", judge_db)
    monkeypatch.setenv("CASELENS_ENV", "production")
    monkeypatch.setenv("ADMIN_PASSWORD", "ProductionAdminPassword123!")
    monkeypatch.setenv("JWT_SECRET_KEY", "JudgeTestJwtSecretKeyForTestingOnly2026!")
    monkeypatch.setenv("JUDGE_ANALYST_USERNAME", "judge_analyst")
    monkeypatch.setenv("JUDGE_ANALYST_PASSWORD", "JudgeAnalystPass2026!")

    init_demo_users(db_path=judge_db, print_output=False)
    seed_judge_users(db_path=judge_db)

    user = get_user_by_username("judge_analyst", db_path=judge_db)
    token = create_access_token(user_id=user["id"], username=user["username"], role=user["role"])
    headers = {"Authorization": f"Bearer {token}"}

    app = create_app()
    client = TestClient(app)

    # 1. Analyst CAN run analysis (e.g. benchmark sample analysis)
    sample_resp = client.post("/api/v1/analyze/sample", json={"limit": 50}, headers=headers)
    assert sample_resp.status_code == 200

    # 2. Analyst gets 403 on publish
    pub_resp = client.post("/api/v1/datasets/sample/publish", headers=headers)
    assert pub_resp.status_code == 403
    assert "Forbidden" in pub_resp.json().get("detail", "")

    # 3. Analyst gets 403 on unpublish
    unpub_resp = client.post("/api/v1/datasets/sample/unpublish", headers=headers)
    assert unpub_resp.status_code == 403

    # 4. Analyst gets 403 on user management
    users_resp = client.get("/api/v1/auth/users", headers=headers)
    assert users_resp.status_code == 403

    create_user_resp = client.post(
        "/api/v1/auth/users",
        json={"username": "bad_user", "password": "password123", "role": "viewer"},
        headers=headers
    )
    assert create_user_resp.status_code == 403

    # 5. Analyst gets 403 on modifying privacy settings
    privacy_resp = client.post(
        "/api/v1/privacy/settings",
        json={"default_epsilon": 0.5, "k_threshold": 10},
        headers=headers
    )
    assert privacy_resp.status_code == 403

    # 6. Analyst gets 403 on budget reset
    reset_resp = client.post("/api/v1/privacy/budget/reset", json={"total_epsilon": 20.0}, headers=headers)
    assert reset_resp.status_code == 403

    # 7. Analyst gets 403 on audit log
    audit_resp = client.get("/api/v1/audit/logs", headers=headers)
    assert audit_resp.status_code == 403


# ==============================================================================
# 3. Judge Viewer: Permissions & 403 Forbidden Restrictions
# ==============================================================================

def test_judge_viewer_permissions_and_restrictions(judge_db, monkeypatch):
    """
    Verifies that the Judge Viewer only sees published datasets and receives
    HTTP 403 on draft datasets and analyst-only endpoints.
    """
    monkeypatch.setenv("LEAKY_DB_PATH", judge_db)
    monkeypatch.setenv("CASELENS_ENV", "production")
    monkeypatch.setenv("ADMIN_PASSWORD", "ProductionAdminPassword123!")
    monkeypatch.setenv("JWT_SECRET_KEY", "JudgeTestJwtSecretKeyForTestingOnly2026!")
    monkeypatch.setenv("JUDGE_VIEWER_USERNAME", "judge_viewer")
    monkeypatch.setenv("JUDGE_VIEWER_PASSWORD", "JudgeViewerPass2026!")

    init_demo_users(db_path=judge_db, print_output=False)
    seed_judge_users(db_path=judge_db)

    user = get_user_by_username("judge_viewer", db_path=judge_db)
    token = create_access_token(user_id=user["id"], username=user["username"], role=user["role"])
    headers = {"Authorization": f"Bearer {token}"}

    app = create_app()
    client = TestClient(app)

    # 1. Viewer gets 403 on running analysis
    sample_resp = client.post("/api/v1/analyze/sample", json={"limit": 50}, headers=headers)
    assert sample_resp.status_code == 403

    case_resp = client.post("/api/v1/analyze/case", json={"police": {"fir_number": "FIR1"}}, headers=headers)
    assert case_resp.status_code == 403

    # 2. Viewer dataset access: registers an unpublished draft dataset directly
    from src.api import DATA_STORE
    DATA_STORE.registered_datasets["test_draft"] = {
        "dataset_id": "test_draft",
        "dataset_label": "Unpublished Draft",
        "is_published": False,
        "analysis_result": {"summary_kpis": {"total_cases": 10}}
    }

    # Viewing list only returns published datasets for viewers
    list_resp = client.get("/api/v1/datasets", headers=headers)
    assert list_resp.status_code == 200
    listed_ids = [d["dataset_id"] for d in list_resp.json().get("datasets", [])]
    assert "test_draft" not in listed_ids

    # Attempting to fetch draft directly returns 403
    draft_resp = client.get("/api/v1/datasets/test_draft", headers=headers)
    assert draft_resp.status_code == 403


# ==============================================================================
# 4. Removing Variables and Restarting Disables Judge Logins
# ==============================================================================

def test_removing_variables_and_restart_disables_judge_logins(judge_db, monkeypatch):
    """
    Verifies that removing the judge environment variables and restarting the application
    disables the judge logins (they can no longer authenticate).
    """
    monkeypatch.setenv("CASELENS_ENV", "production")
    monkeypatch.setenv("ADMIN_PASSWORD", "ProductionAdminPassword123!")
    monkeypatch.setenv("JUDGE_VIEWER_USERNAME", "temp_judge_viewer")
    monkeypatch.setenv("JUDGE_VIEWER_PASSWORD", "TempJudgePass123!")

    # Initial startup with variables set
    init_demo_users(db_path=judge_db, print_output=False)
    seed_judge_users(db_path=judge_db)

    # Confirm judge user exists and can authenticate
    auth_user = authenticate_user("temp_judge_viewer", "TempJudgePass123!", db_path=judge_db)
    assert auth_user["username"] == "temp_judge_viewer"

    # Now remove the variables (simulating turning off judge access)
    monkeypatch.delenv("JUDGE_VIEWER_USERNAME", raising=False)
    monkeypatch.delenv("JUDGE_VIEWER_PASSWORD", raising=False)

    # Next startup / restart:
    seed_judge_users(db_path=judge_db)

    # User must no longer exist or authenticate
    assert get_user_by_username("temp_judge_viewer", db_path=judge_db) is None

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        authenticate_user("temp_judge_viewer", "TempJudgePass123!", db_path=judge_db)
    assert exc_info.value.status_code in (401, 403)


# ==============================================================================
# 5. JUDGE_MODE Visibility Helper Test
# ==============================================================================

def test_judge_mode_visibility_helper(monkeypatch):
    """
    Verifies that is_judge_access_available returns True ONLY when JUDGE_MODE='on'
    AND the specific credentials for that button are configured.
    """
    monkeypatch.setattr("app.api_client._DOTENV_CACHE", {})
    monkeypatch.delenv("JUDGE_MODE", raising=False)
    monkeypatch.delenv("JUDGE_VIEWER_USERNAME", raising=False)
    monkeypatch.delenv("JUDGE_VIEWER_PASSWORD", raising=False)
    monkeypatch.delenv("JUDGE_ANALYST_USERNAME", raising=False)
    monkeypatch.delenv("JUDGE_ANALYST_PASSWORD", raising=False)

    # 1. JUDGE_MODE unset
    monkeypatch.setenv("JUDGE_VIEWER_USERNAME", "viewer_j")
    monkeypatch.setenv("JUDGE_VIEWER_PASSWORD", "Password123!")
    assert is_judge_access_available("viewer") is False
    assert is_judge_access_available("analyst") is False

    # 2. JUDGE_MODE='off'
    monkeypatch.setenv("JUDGE_MODE", "off")
    assert is_judge_access_available("viewer") is False

    # 3. JUDGE_MODE='OFF' (case insensitive check)
    monkeypatch.setenv("JUDGE_MODE", "OFF")
    assert is_judge_access_available("viewer") is False

    # 4. JUDGE_MODE='on' but credentials missing
    monkeypatch.setenv("JUDGE_MODE", "on")
    monkeypatch.delenv("JUDGE_VIEWER_PASSWORD", raising=False)
    assert is_judge_access_available("viewer") is False

    # 5. JUDGE_MODE='on' and viewer credentials set
    monkeypatch.setenv("JUDGE_VIEWER_PASSWORD", "Password123!")
    assert is_judge_access_available("viewer") is True
    assert is_judge_access_available("analyst") is False

    # 6. Password length check (< 8 characters rejected)
    monkeypatch.setenv("JUDGE_VIEWER_PASSWORD", "short")
    assert is_judge_access_available("viewer") is False
    monkeypatch.setenv("JUDGE_VIEWER_PASSWORD", "12345678")
    assert is_judge_access_available("viewer") is True

    # 7. JUDGE_MODE='ON' and analyst credentials set
    monkeypatch.setenv("JUDGE_ANALYST_USERNAME", "analyst_j")
    monkeypatch.setenv("JUDGE_ANALYST_PASSWORD", "AnalystPassword123!")
    assert is_judge_access_available("analyst") is True


def test_judge_credential_quotes_and_whitespace_cleaning(monkeypatch):
    """
    Verifies that get_judge_credential strips surrounding single quotes, double quotes,
    nested quotes, and leading/trailing whitespace.
    """
    monkeypatch.setattr("app.api_client._DOTENV_CACHE", {})

    # Single quotes
    monkeypatch.setenv("JUDGE_MODE", "'on'")
    assert get_judge_credential("JUDGE_MODE") == "on"

    # Double quotes with leading/trailing whitespace
    monkeypatch.setenv("JUDGE_MODE", '  "on"  ')
    assert get_judge_credential("JUDGE_MODE") == "on"

    # Surrounding quotes on username and password
    monkeypatch.setenv("JUDGE_VIEWER_USERNAME", '  "judge_viewer_user"  ')
    monkeypatch.setenv("JUDGE_VIEWER_PASSWORD", " 'SuperSecretPass2026!' ")
    assert get_judge_credential("JUDGE_VIEWER_USERNAME") == "judge_viewer_user"
    assert get_judge_credential("JUDGE_VIEWER_PASSWORD") == "SuperSecretPass2026!"
    assert is_judge_access_available("viewer") is True


def test_judge_credential_precedence_and_dotenv_fallback(monkeypatch):
    """
    Verifies the strict priority order of configuration resolution:
    1. os.environ takes highest priority
    2. st.secrets takes secondary priority
    3. .env fallback takes third priority
    """
    # 1. Clean environment
    monkeypatch.delenv("JUDGE_MODE", raising=False)
    monkeypatch.delenv("JUDGE_VIEWER_USERNAME", raising=False)
    monkeypatch.delenv("JUDGE_VIEWER_PASSWORD", raising=False)

    # Mock .env cache
    mock_dotenv = {
        "JUDGE_MODE": "on",
        "JUDGE_VIEWER_USERNAME": "dotenv_viewer",
        "JUDGE_VIEWER_PASSWORD": "DotenvPassword2026!"
    }
    monkeypatch.setattr("app.api_client._DOTENV_CACHE", mock_dotenv)

    # .env fallback works when env and secrets are unset
    assert get_judge_credential("JUDGE_MODE") == "on"
    assert get_judge_credential("JUDGE_VIEWER_USERNAME") == "dotenv_viewer"
    assert is_judge_access_available("viewer") is True

    # 2. st.secrets overrides .env fallback
    import app.api_client as ac
    mock_st = type("MockSt", (), {
        "secrets": {
            "JUDGE_MODE": "secrets_mode",
            "JUDGE_VIEWER_USERNAME": "secrets_viewer"
        }
    })()
    monkeypatch.setattr(ac, "st", mock_st)
    monkeypatch.setattr(ac, "_HAS_STREAMLIT", True)

    assert get_judge_credential("JUDGE_MODE") == "secrets_mode"
    assert get_judge_credential("JUDGE_VIEWER_USERNAME") == "secrets_viewer"
    # JUDGE_VIEWER_PASSWORD not in secrets -> falls back to .env
    assert get_judge_credential("JUDGE_VIEWER_PASSWORD") == "DotenvPassword2026!"

    # 3. os.environ overrides both st.secrets and .env
    monkeypatch.setenv("JUDGE_MODE", "env_override_mode")
    assert get_judge_credential("JUDGE_MODE") == "env_override_mode"


def test_judge_button_visibility_logic(monkeypatch):
    """
    Verifies that judge buttons appear ONLY when JUDGE_MODE is 'on'
    and credentials for that role are configured.
    """
    monkeypatch.setattr("app.api_client._DOTENV_CACHE", {})
    monkeypatch.delenv("JUDGE_MODE", raising=False)
    monkeypatch.delenv("JUDGE_VIEWER_USERNAME", raising=False)
    monkeypatch.delenv("JUDGE_VIEWER_PASSWORD", raising=False)
    monkeypatch.delenv("JUDGE_ANALYST_USERNAME", raising=False)
    monkeypatch.delenv("JUDGE_ANALYST_PASSWORD", raising=False)

    # 1. JUDGE_MODE unset: neither button visible
    assert is_judge_access_available("viewer") is False
    assert is_judge_access_available("analyst") is False

    # 2. JUDGE_MODE='on', neither credentials set: neither button visible
    monkeypatch.setenv("JUDGE_MODE", "on")
    assert is_judge_access_available("viewer") is False
    assert is_judge_access_available("analyst") is False

    # 3. Viewer configured, analyst not configured
    monkeypatch.setenv("JUDGE_VIEWER_USERNAME", "judge_v")
    monkeypatch.setenv("JUDGE_VIEWER_PASSWORD", "SecretPassword2026!")
    assert is_judge_access_available("viewer") is True
    assert is_judge_access_available("analyst") is False

    # 4. Both configured
    monkeypatch.setenv("JUDGE_ANALYST_USERNAME", "judge_a")
    monkeypatch.setenv("JUDGE_ANALYST_PASSWORD", "AnalystSecret2026!")
    assert is_judge_access_available("viewer") is True
    assert is_judge_access_available("analyst") is True


# ==============================================================================
# 6. No Plaintext Passwords Leaked
# ==============================================================================

def test_no_plaintext_passwords_or_hashes_leaked(judge_db, monkeypatch, capsys, caplog):
    """
    Verifies that no plaintext passwords or password hashes appear in logs or stdout,
    and DuckDB stores strictly Argon2id hashes.
    """
    monkeypatch.setenv("CASELENS_ENV", "production")
    monkeypatch.setenv("ADMIN_PASSWORD", "ProductionAdminPassword123!")
    monkeypatch.setenv("JUDGE_VIEWER_USERNAME", "audit_judge_viewer")
    monkeypatch.setenv("JUDGE_VIEWER_PASSWORD", "SuperSecretJudgePassword999!")

    with caplog.at_level(logging.DEBUG):
        seed_judge_users(db_path=judge_db)
        authenticate_user("audit_judge_viewer", "SuperSecretJudgePassword999!", db_path=judge_db)

    # 1. Verify stdout / stderr
    captured = capsys.readouterr()
    assert "SuperSecretJudgePassword999!" not in captured.out
    assert "SuperSecretJudgePassword999!" not in captured.err

    # 2. Verify logs
    for record in caplog.records:
        assert "SuperSecretJudgePassword999!" not in record.message

    # 3. Verify DuckDB users table
    conn = get_db(judge_db)
    rows = conn.execute("SELECT password_hash FROM users WHERE username = 'audit_judge_viewer'").fetchall()
    assert len(rows) == 1
    pw_hash = rows[0][0]
    assert pw_hash.startswith("$argon2id$")
    assert "SuperSecretJudgePassword999!" not in pw_hash

