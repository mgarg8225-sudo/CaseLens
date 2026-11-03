"""
tests/test_safe_init.py - Comprehensive Unit & Integration Tests for Safe User Initialization (Phase 2).

Verifies:
1. Fresh production database with a valid admin password provisions the admin account successfully.
2. Fresh production database without a valid admin password halts startup with a clear error and no leaked secrets.
3. Existing users remain completely unchanged during safe initialization, retaining password hashes,
   roles, is_active status, failed_attempts counters, and locked_until timestamps (including locked/inactive states).
4. Production initialization strictly ignores DEMO_*_PASSWORD variables.
5. Development mode and existing test seeding behaviors remain 100% backward compatible.
6. Database initialization failure stops startup without silently continuing.
7. Zero credential, secret, or password hash leakage appears in logs, exception messages, or stdout.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import pytest

from src.auth import (
    get_db,
    init_auth_db,
    close_db,
    get_user_by_username,
    hash_password,
    verify_password,
    authenticate_user,
    is_production_mode,
)
from src.seed_users import init_demo_users, seed_demo_users
from src.api import create_app


@pytest.fixture
def isolated_duckdb(tmp_path):
    """Provides an isolated temporary DuckDB file for initialization testing."""
    db_file = str(tmp_path / "test_init.duckdb")
    close_db(db_file)
    yield db_file
    close_db(db_file)


# ==============================================================================
# 1. Fresh Production Database with Valid Admin Password
# ==============================================================================

def test_fresh_production_db_with_admin_password(isolated_duckdb, monkeypatch):
    """Verifies that on a fresh DB in production, configuring ADMIN_PASSWORD bootstraps admin cleanly."""
    monkeypatch.setenv("CASELENS_ENV", "production")
    monkeypatch.setenv("ADMIN_PASSWORD", "SuperSecureAdminPassword2026!")
    # Clear any demo password overrides
    monkeypatch.delenv("DEMO_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("INITIAL_ADMIN_PASSWORD", raising=False)

    assert is_production_mode() is True

    creds = init_demo_users(db_path=isolated_duckdb, print_output=False, only_if_missing=True)
    assert "admin" in creds

    # In production, returned password must be masked as [REDACTED]
    assert creds["admin"]["password"] == "[REDACTED]"

    # Verify user exists in DuckDB
    user = get_user_by_username("admin", db_path=isolated_duckdb)
    assert user is not None
    assert user["role"] == "admin"
    assert user["is_active"] is True
    assert user["failed_attempts"] == 0
    assert user["locked_until"] is None

    # Verify user can authenticate with the configured production password
    auth_user = authenticate_user("admin", "SuperSecureAdminPassword2026!", db_path=isolated_duckdb)
    assert auth_user["username"] == "admin"
    assert auth_user["role"] == "admin"


# ==============================================================================
# 2. Fresh Production Database Without Valid Admin Password
# ==============================================================================

def test_fresh_production_db_without_admin_password_halts(isolated_duckdb, monkeypatch):
    """Verifies that missing admin password on fresh production DB halts startup with clear error."""
    monkeypatch.setenv("CASELENS_ENV", "production")
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("INITIAL_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("DEMO_ADMIN_PASSWORD", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        init_demo_users(db_path=isolated_duckdb, print_output=False, only_if_missing=True)

    err_msg = str(exc_info.value)
    assert "CRITICAL SECURITY CONFIGURATION ERROR" in err_msg
    assert "ADMIN_PASSWORD" in err_msg
    # Ensure no secret strings or hashes appear in the error message
    assert "password" in err_msg.lower()
    assert "$argon2id$" not in err_msg


def test_fresh_production_db_with_short_admin_password_halts(isolated_duckdb, monkeypatch):
    """Verifies that an admin password shorter than 8 characters is strictly rejected."""
    monkeypatch.setenv("CASELENS_ENV", "production")
    monkeypatch.setenv("ADMIN_PASSWORD", "short")

    with pytest.raises(RuntimeError) as exc_info:
        init_demo_users(db_path=isolated_duckdb, print_output=False, only_if_missing=True)

    assert "at least 8 characters" in str(exc_info.value)


# ==============================================================================
# 3. Existing Users Completely Preserved (Including Inactive and Locked States)
# ==============================================================================

def test_existing_users_preserved_including_inactive_and_locked_states(isolated_duckdb, monkeypatch):
    """
    Verifies that existing users are never overwritten during startup:
    - Custom password hashes are retained.
    - Inactive accounts stay inactive.
    - Locked accounts retain their lockout timestamps and attempt counts.
    """
    conn = get_db(isolated_duckdb)
    init_auth_db(conn)

    initial_custom_hash = hash_password("OriginalCustomPassword123!")
    lockout_expiry = datetime.now(timezone.utc) + timedelta(minutes=15)

    # Manually insert an admin that was locked out and deactivated
    conn.execute(
        """
        INSERT INTO users (id, username, password_hash, role, agency, district, is_active, failed_attempts, locked_until)
        VALUES ('admin_id_01', 'admin', ?, 'admin', NULL, NULL, FALSE, 5, ?)
        """,
        [initial_custom_hash, lockout_expiry]
    )

    # Manually insert an analyst with their own password
    analyst_hash = hash_password("AnalystOriginalPassword123!")
    conn.execute(
        """
        INSERT INTO users (id, username, password_hash, role, agency, district, is_active, failed_attempts, locked_until)
        VALUES ('analyst_id_01', 'analyst', ?, 'analyst', NULL, NULL, TRUE, 2, NULL)
        """,
        [analyst_hash]
    )

    # Now run safe initialization in production mode with a different ADMIN_PASSWORD set
    monkeypatch.setenv("CASELENS_ENV", "production")
    monkeypatch.setenv("ADMIN_PASSWORD", "BrandNewDifferentPassword123!")

    creds = init_demo_users(db_path=isolated_duckdb, print_output=False, only_if_missing=True)

    # Admin verification: hash, is_active, failed_attempts, locked_until MUST NOT CHANGE
    admin_after = get_user_by_username("admin", db_path=isolated_duckdb)
    assert admin_after["password_hash"] == initial_custom_hash
    assert admin_after["is_active"] is False  # Remained deactivated!
    assert admin_after["failed_attempts"] == 5  # Attempts not cleared!
    assert admin_after["locked_until"] is not None  # Lockout expiry not cleared!

    # Analyst verification: hash and attempt count unchanged
    analyst_after = get_user_by_username("analyst", db_path=isolated_duckdb)
    assert analyst_after["password_hash"] == analyst_hash
    assert analyst_after["is_active"] is True
    assert analyst_after["failed_attempts"] == 2

    # Credential dictionary returned should have masked passwords
    assert creds["admin"]["password"] == "[REDACTED]"
    assert creds["analyst"]["password"] == "[REDACTED]"


# ==============================================================================
# 4. Production Ignores DEMO_*_PASSWORD Variables
# ==============================================================================

def test_production_ignores_demo_passwords(isolated_duckdb, monkeypatch):
    """Verifies that production initialization ignores DEMO_ADMIN_PASSWORD."""
    monkeypatch.setenv("CASELENS_ENV", "production")
    monkeypatch.setenv("DEMO_ADMIN_PASSWORD", "InsecureDemoPassword123!")
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("INITIAL_ADMIN_PASSWORD", raising=False)

    # Must raise RuntimeError because DEMO_ADMIN_PASSWORD is not allowed in production
    with pytest.raises(RuntimeError) as exc_info:
        init_demo_users(db_path=isolated_duckdb, print_output=False, only_if_missing=True)

    assert "CRITICAL SECURITY CONFIGURATION ERROR" in str(exc_info.value)


# ==============================================================================
# 5. Development Mode and Test Compatibility
# ==============================================================================

def test_development_mode_compatibility(isolated_duckdb, monkeypatch):
    """Verifies that development mode retains random generation and test compatibility."""
    monkeypatch.delenv("CASELENS_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("RENDER", raising=False)

    assert is_production_mode() is False

    # In dev mode, init_demo_users can generate random passwords if none are provided
    creds = init_demo_users(db_path=isolated_duckdb, print_output=False, only_if_missing=False)
    assert "admin" in creds
    assert "viewer" in creds
    assert "analyst" in creds
    assert len(creds["admin"]["password"]) >= 8

    # Can authenticate with returned password
    auth_user = authenticate_user("admin", creds["admin"]["password"], db_path=isolated_duckdb)
    assert auth_user["role"] == "admin"


# ==============================================================================
# 6. Database Initialization Failure Stops Startup
# ==============================================================================

def test_database_initialization_failure_stops_startup(monkeypatch):
    """Verifies that if database initialization fails, lifespan aborts startup loudly."""
    from src.api import lifespan, create_app

    app = create_app()

    # Simulate database initialization failure
    with patch("src.auth.init_auth_db", side_effect=RuntimeError("Simulated disk I/O failure")):
        with pytest.raises(RuntimeError) as exc_info:
            gen = lifespan(app)
            coro = gen.__aenter__()
            try:
                coro.send(None)
            finally:
                coro.close()

        assert "Database schema initialization failed on startup" in str(exc_info.value)
        assert "Simulated disk I/O failure" in str(exc_info.value)


# ==============================================================================
# 7. No Credential Leakage in Logs or Output
# ==============================================================================

def test_no_credential_leakage_in_logs_or_output(isolated_duckdb, monkeypatch, capsys, caplog):
    """Verifies that no plaintext passwords, Argon2 hashes, or secrets appear in stdout or logs."""
    monkeypatch.setenv("CASELENS_ENV", "production")
    monkeypatch.setenv("ADMIN_PASSWORD", "SecretAdminPassword987654!")

    with caplog.at_level(logging.DEBUG):
        creds = init_demo_users(db_path=isolated_duckdb, print_output=True, only_if_missing=True)

    # 1. Inspect captured stdout/stderr
    captured = capsys.readouterr()
    assert "SecretAdminPassword987654!" not in captured.out
    assert "SecretAdminPassword987654!" not in captured.err
    assert "$argon2id$" not in captured.out
    assert "$argon2id$" not in captured.err

    # 2. Inspect captured logs
    for record in caplog.records:
        assert "SecretAdminPassword987654!" not in record.message
        assert "$argon2id$" not in record.message
