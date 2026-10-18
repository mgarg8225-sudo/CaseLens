"""
tests/test_auth.py - Comprehensive Unit & Integration Tests for Phase A Authentication.

Requirements Covered:
1. Argon2 password hashing (no plaintext passwords, minimum length enforced).
2. JWT token generation & verification (30 min lifetime, correct claims).
3. Fail loudly when JWT_SECRET_KEY is missing.
4. Expired token rejected.
5. Tampered token rejected.
6. Wrong password rejected (generic error message).
7. Account lockout works (5 failed attempts locks account for 15 minutes).
8. Inactive / deactivated user rejected.
9. Admin-only user creation and deactivation endpoints.
10. Seeding script generates 9 distinct random passwords and valid user records.
"""

import os
import time
import pytest
from datetime import datetime, timedelta, timezone
from starlette.testclient import TestClient

from src.auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    get_jwt_secret_key,
    get_db,
    init_auth_db,
    close_db,
    create_user_record,
    set_user_active_status,
    list_all_users,
    authenticate_user,
    get_user_by_username,
    VALID_ROLES
)
from src.seed_users import seed_demo_users
from src.api import create_app


@pytest.fixture(scope="module")
def test_db_path(tmp_path_factory):
    """Provides a fresh temporary DuckDB database file for auth tests."""
    temp_dir = tmp_path_factory.mktemp("auth_test")
    db_file = os.path.join(temp_dir, "test_pipeline.duckdb")
    os.environ["JWT_SECRET_KEY"] = "super-secret-test-key-32-bytes-minimum!!"
    os.environ["LEAKY_DB_PATH"] = db_file
    yield db_file
    close_db(db_file)


@pytest.fixture(scope="module")
def client(test_db_path):
    """Initializes TestClient pointing to the test database."""
    app = create_app()
    return TestClient(app)


# ==============================================================================
# 1. Password Hashing & Security Tests
# ==============================================================================

def test_password_hashing_argon2():
    """Verifies Argon2 hashing, salt generation, and length enforcement."""
    pwd = "ValidPassword123!"
    h1 = hash_password(pwd)
    h2 = hash_password(pwd)

    # Must produce different hashes due to distinct random salts
    assert h1 != h2
    assert h1.startswith("$argon2id$")
    assert verify_password(h1, pwd) is True
    assert verify_password(h1, "WrongPassword123!") is False

    # Short password must be rejected
    with pytest.raises(ValueError):
        hash_password("short")


# ==============================================================================
# 2. JWT Token & Environment Variable Tests
# ==============================================================================

def test_jwt_token_creation_and_decoding():
    """Verifies access token issuance, claims, and successful decode."""
    token = create_access_token(
        user_id="user_123",
        username="police_officer",
        role="police",
        agency="police",
        district="D3_Rudrapur"
    )
    assert isinstance(token, str)
    assert token.count(".") == 2

    payload = decode_access_token(token)
    assert payload["sub"] == "user_123"
    assert payload["username"] == "police_officer"
    assert payload["role"] == "police"
    assert payload["agency"] == "police"
    assert payload["district"] == "D3_Rudrapur"
    assert payload["exp"] > payload["iat"]


def test_missing_secret_key_fails_loudly(monkeypatch):
    """Verifies that missing JWT_SECRET_KEY raises a loud RuntimeError."""
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    
    # Temporarily hide .env if present
    env_backup = None
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            env_backup = f.read()
        os.remove(".env")

    try:
        with pytest.raises(RuntimeError) as exc_info:
            get_jwt_secret_key()
        assert "CRITICAL SECURITY CONFIGURATION ERROR" in str(exc_info.value)
    finally:
        if env_backup:
            with open(".env", "w") as f:
                f.write(env_backup)
        os.environ["JWT_SECRET_KEY"] = "super-secret-test-key-32-bytes-minimum!!"


def test_expired_token_rejected():
    """Verifies that an expired JWT token is strictly rejected with HTTP 401."""
    # Issue a token that expired 5 minutes ago
    expired_token = create_access_token(
        user_id="user_exp",
        username="expired_user",
        role="viewer",
        expires_delta=timedelta(minutes=-5)
    )

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(expired_token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_tampered_token_rejected():
    """Verifies that tampering with a JWT token payload or signature is rejected with HTTP 401."""
    token = create_access_token(user_id="user_legit", username="legit_user", role="viewer")
    parts = token.split(".")
    # Tamper with the signature
    tampered_sig = parts[2][:-4] + "XXXX"
    tampered_token = f"{parts[0]}.{parts[1]}.{tampered_sig}"

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(tampered_token)
    assert exc_info.value.status_code == 401
    assert "invalid or tampered" in exc_info.value.detail.lower()


# ==============================================================================
# 3. User Creation, Password Validation & Lockout Tests
# ==============================================================================

def test_user_creation_and_validation(test_db_path):
    """Tests creating a valid user, duplicate rejection, and role validation."""
    from fastapi import HTTPException
    
    u = create_user_record(
        username="hospital_user1",
        password="SecurePass2026!",
        role="hospital",
        agency="hospital",
        district="D1_Suryapur",
        db_path=test_db_path
    )
    assert u["username"] == "hospital_user1"
    assert u["role"] == "hospital"
    assert "password_hash" not in u
    assert u["is_active"] is True

    # Duplicate username rejection
    with pytest.raises(HTTPException) as exc:
        create_user_record("hospital_user1", "AnotherPass2026!", "hospital", db_path=test_db_path)
    assert exc.value.status_code == 400
    assert "already exists" in exc.value.detail

    # Invalid role rejection
    with pytest.raises(HTTPException) as exc:
        create_user_record("bad_role_user", "ValidPass2026!", "super_hacker", db_path=test_db_path)
    assert exc.value.status_code == 400
    assert "Invalid role" in exc.value.detail


def test_wrong_password_rejected(test_db_path):
    """Verifies wrong password rejection with generic error message."""
    from fastapi import HTTPException
    create_user_record(
        username="police_test_pw",
        password="CorrectPassword123!",
        role="police",
        db_path=test_db_path
    )

    with pytest.raises(HTTPException) as exc:
        authenticate_user("police_test_pw", "WrongPassword123!", db_path=test_db_path)
    assert exc.value.status_code == 401
    assert "Invalid username or password" in exc.value.detail

    # Nonexistent user must return the exact same generic message
    with pytest.raises(HTTPException) as exc_nonexistent:
        authenticate_user("totally_nonexistent_user", "AnyPassword123!", db_path=test_db_path)
    assert exc_nonexistent.value.status_code == 401
    assert "Invalid username or password" in exc_nonexistent.value.detail


def test_account_lockout_after_5_failed_attempts(test_db_path):
    """
    Verifies:
    1. First 4 failed attempts return 401.
    2. 5th failed attempt locks the account for 15 minutes and returns 403.
    3. Subsequent attempts while locked return 403.
    """
    from fastapi import HTTPException
    username = "lockout_target_user"
    create_user_record(username, "MySecretPass123!", "forensic", db_path=test_db_path)

    # 4 Failed Attempts
    for i in range(1, 5):
        with pytest.raises(HTTPException) as exc:
            authenticate_user(username, "WrongGuess!", db_path=test_db_path)
        assert exc.value.status_code == 401

    # 5th Failed Attempt -> Triggers Lockout
    with pytest.raises(HTTPException) as exc5:
        authenticate_user(username, "WrongGuess!", db_path=test_db_path)
    assert exc5.value.status_code == 403
    assert "locked for 15 minutes" in exc5.value.detail

    # 6th Attempt with CORRECT password -> Still rejected because locked
    with pytest.raises(HTTPException) as exc6:
        authenticate_user(username, "MySecretPass123!", db_path=test_db_path)
    assert exc6.value.status_code == 403
    assert "temporarily locked" in exc6.value.detail


def test_inactive_user_rejected(test_db_path):
    """Verifies that deactivated users are rejected upon login."""
    from fastapi import HTTPException
    username = "deactivated_user"
    u = create_user_record(username, "MyPassword2026!", "court", db_path=test_db_path)

    # Deactivate account
    set_user_active_status(u["id"], is_active=False, db_path=test_db_path)

    # Login attempt with correct password must be rejected
    with pytest.raises(HTTPException) as exc:
        authenticate_user(username, "MyPassword2026!", db_path=test_db_path)
    assert exc.value.status_code == 403
    assert "deactivated" in exc.value.detail.lower()

    # Reactivate account
    set_user_active_status(u["id"], is_active=True, db_path=test_db_path)
    auth_ok = authenticate_user(username, "MyPassword2026!", db_path=test_db_path)
    assert auth_ok["username"] == username


# ==============================================================================
# 4. Seed Demo Users Script Verification
# ==============================================================================

def test_seed_demo_users(test_db_path):
    """
    Verifies:
    1. Seeding creates 9 demo users across all roles.
    2. Passwords generated are random and distinct.
    3. Each demo user can log in with their generated credentials.
    """
    creds = seed_demo_users(db_path=test_db_path, print_output=False)
    assert len(creds) == 9

    # Verify all roles represented
    roles_seeded = {u["role"] for u in creds.values()}
    for role in VALID_ROLES:
        assert role in roles_seeded

    # Verify all passwords are distinct
    passwords = [u["password"] for u in creds.values()]
    assert len(set(passwords)) == 9

    # Verify each can authenticate
    for username, data in creds.items():
        res = authenticate_user(username, data["password"], db_path=test_db_path)
        assert res["username"] == username
        assert res["role"] == data["role"]


# ==============================================================================
# 5. REST API Integration Endpoints Verification
# ==============================================================================

def test_api_login_endpoint(client, test_db_path):
    """Tests POST /api/v1/auth/login and subsequent authenticated requests."""
    creds = seed_demo_users(db_path=test_db_path, print_output=False)
    admin_cred = creds["admin_demo"]

    # 1. Login with valid credentials
    resp = client.post("/api/v1/auth/login", json={
        "username": admin_cred["username"],
        "password": admin_cred["password"]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["role"] == "admin"
    token = data["access_token"]

    # 2. Call /api/v1/auth/me with Bearer token
    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["username"] == admin_cred["username"]
    assert me_data["role"] == "admin"

    # 3. Admin creates another user via API
    new_user_resp = client.post(
        "/api/v1/auth/users",
        json={
            "username": "api_created_officer",
            "password": "SpecialPassword2026!",
            "role": "police",
            "district": "D3_Rudrapur"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert new_user_resp.status_code == 201
    created_id = new_user_resp.json()["id"]

    # 4. Admin lists all users
    list_resp = client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {token}"})
    assert list_resp.status_code == 200
    all_users = list_resp.json()
    assert any(u["username"] == "api_created_officer" for u in all_users)

    # 5. Admin deactivates created user
    deact_resp = client.patch(
        f"/api/v1/auth/users/{created_id}/deactivate",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert deact_resp.status_code == 200
    assert deact_resp.json()["is_active"] is False

    # 6. Non-admin user tries to access admin endpoint -> 403 Forbidden
    police_cred = creds["police_demo"]
    p_login = client.post("/api/v1/auth/login", json={
        "username": police_cred["username"],
        "password": police_cred["password"]
    })
    police_token = p_login.json()["access_token"]

    forbidden_resp = client.get(
        "/api/v1/auth/users",
        headers={"Authorization": f"Bearer {police_token}"}
    )
    assert forbidden_resp.status_code == 403
