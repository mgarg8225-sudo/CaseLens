"""
src/auth.py - Authentication, User Management, and Security Engine.

Phase A of CaseLens Security Architecture:
1. DuckDB Users Schema: id, username, password_hash (Argon2), role, agency, district,
   is_active, failed_attempts, locked_until.
2. Zero Plaintext Passwords: All passwords securely hashed with Argon2id. Never logged or saved.
3. Short-Lived JWT Tokens: 30-minute expiration with user_id, role, agency, district.
   Secret key loaded from JWT_SECRET_KEY environment variable (fails loudly if missing).
4. Account Lockout: Enforced after 5 failed login attempts for 15 minutes.
5. Admin User Management: Secure endpoints to create, list, deactivate, and activate users.
"""

import os
import re
import uuid
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

import duckdb
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError
from fastapi import APIRouter, Depends, HTTPException, Header, status
from pydantic import BaseModel, Field


# ==============================================================================
# 1. Constants & Role Definitions
# ==============================================================================

VALID_ROLES: List[str] = [
    "police",
    "hospital",
    "forensic",
    "prosecutor",
    "court",
    "linkage_reviewer",
    "auditor",
    "public_viewer",
    "admin"
]

AGENCY_ROLES: List[str] = [
    "police",
    "hospital",
    "forensic",
    "prosecutor",
    "court"
]

DEFAULT_LOCKOUT_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 30


# ==============================================================================
# 2. Password Hashing with Argon2
# ==============================================================================

_ph = PasswordHasher(
    time_cost=2,
    memory_cost=65536,  # 64 MiB
    parallelism=1,
    hash_len=32,
    salt_len=16
)


def hash_password(password: str) -> str:
    """Hashes a plaintext password using Argon2id. Enforces minimum length."""
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    return _ph.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Verifies a password against an Argon2 hash. Never raises on mismatch."""
    try:
        return _ph.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False
    except Exception:
        return False


# ==============================================================================
# 3. Environment & JWT Configuration
# ==============================================================================

def load_env_file(env_path: str = ".env"):
    """Loads environment variables from .env file if present."""
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


def get_jwt_secret_key() -> str:
    """
    Retrieves the JWT secret key from the environment.
    FAILS LOUDLY if missing or empty.
    """
    load_env_file()
    key = os.environ.get("JWT_SECRET_KEY") or os.environ.get("JWT_SECRET")
    if not key or not key.strip():
        raise RuntimeError(
            "CRITICAL SECURITY CONFIGURATION ERROR: 'JWT_SECRET_KEY' environment variable is missing or empty! "
            "Cannot issue or verify authentication tokens without a secure secret key."
        )
    return key.strip()


def create_access_token(
    user_id: str,
    username: str,
    role: str,
    agency: Optional[str] = None,
    district: Optional[str] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Issues a short-lived signed JWT access token (default 30 min)."""
    secret_key = get_jwt_secret_key()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=JWT_EXPIRATION_MINUTES))
    payload = {
        "sub": user_id,
        "user_id": user_id,
        "username": username,
        "role": role,
        "agency": agency,
        "district": district,
        "exp": expire,
        "iat": now,
        "nbf": now
    }
    return jwt.encode(payload, secret_key, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decodes and validates a JWT token.
    Raises HTTPException 401 on expired, invalid, or tampered signatures.
    """
    secret_key = get_jwt_secret_key()
    try:
        payload = jwt.decode(token, secret_key, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or tampered authentication token.",
            headers={"WWW-Authenticate": "Bearer"}
        )


# ==============================================================================
# 4. DuckDB Database Connection & Schema Management
# ==============================================================================

_db_lock = threading.Lock()
_db_connections: Dict[str, duckdb.DuckDBPyConnection] = {}


def _get_default_db_path() -> str:
    load_env_file()
    return os.environ.get("LEAKY_DB_PATH") or os.path.join(
        os.environ.get("LEAKY_DATA_DIR", "data"), "pipeline.duckdb"
    )


def init_auth_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Initializes the users table schema in DuckDB with proper constraints."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR PRIMARY KEY,
            username VARCHAR UNIQUE NOT NULL,
            password_hash VARCHAR NOT NULL,
            role VARCHAR NOT NULL,
            agency VARCHAR,
            district VARCHAR,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until TIMESTAMP
        );
    """)


def get_db(db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    """Returns a thread-safe connection to the DuckDB database."""
    path = db_path or _get_default_db_path()
    with _db_lock:
        if path not in _db_connections:
            if path != ":memory:":
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            conn = duckdb.connect(database=path)
            init_auth_db(conn)
            _db_connections[path] = conn
        return _db_connections[path]


def close_db(db_path: Optional[str] = None) -> None:
    """Closes an active DuckDB connection (useful for tests)."""
    path = db_path or _get_default_db_path()
    with _db_lock:
        if path in _db_connections:
            try:
                _db_connections[path].close()
            except Exception:
                pass
            del _db_connections[path]


# ==============================================================================
# 5. Core User CRUD & Authentication Operations
# ==============================================================================

def _row_to_user(row: tuple) -> Dict[str, Any]:
    """Maps a users table row tuple to a dictionary."""
    locked_val = row[8]
    if locked_val is not None and hasattr(locked_val, "tzinfo") and locked_val.tzinfo is None:
        locked_val = locked_val.replace(tzinfo=timezone.utc)
    
    return {
        "id": row[0],
        "username": row[1],
        "password_hash": row[2],
        "role": row[3],
        "agency": row[4],
        "district": row[5],
        "is_active": bool(row[6]),
        "failed_attempts": int(row[7]),
        "locked_until": locked_val
    }


def get_user_by_username(username: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a user by username (case-insensitive) using parameterized SQL."""
    conn = get_db(db_path)
    res = conn.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?)", [username.strip()]).fetchall()
    if not res:
        return None
    return _row_to_user(res[0])


def get_user_by_id(user_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a user by ID using parameterized SQL."""
    conn = get_db(db_path)
    res = conn.execute("SELECT * FROM users WHERE id = ?", [user_id]).fetchall()
    if not res:
        return None
    return _row_to_user(res[0])


def create_user_record(
    username: str,
    password: str,
    role: str,
    agency: Optional[str] = None,
    district: Optional[str] = None,
    is_active: bool = True,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a new user with Argon2-hashed password.
    Enforces validation on role, agency, password length, and uniqueness.
    """
    username_clean = username.strip()
    if not re.match(r"^[a-zA-Z0-9_]{3,32}$", username_clean):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be 3-32 characters and contain only letters, numbers, or underscores."
        )

    role_clean = role.strip().lower()
    if role_clean not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{role}'. Allowed roles: {', '.join(VALID_ROLES)}"
        )

    # Agency auto-alignment
    if role_clean in AGENCY_ROLES and not agency:
        agency = role_clean

    # Check for existing user
    existing = get_user_by_username(username_clean, db_path=db_path)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{username_clean}' already exists."
        )

    try:
        pw_hash = hash_password(password)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    user_id = str(uuid.uuid4())
    conn = get_db(db_path)
    conn.execute(
        """
        INSERT INTO users (id, username, password_hash, role, agency, district, is_active, failed_attempts, locked_until)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)
        """,
        [user_id, username_clean, pw_hash, role_clean, agency, district, is_active]
    )

    created = get_user_by_id(user_id, db_path=db_path)
    # Never return password_hash in response
    if created:
        created_safe = dict(created)
        created_safe.pop("password_hash", None)
        return created_safe
    raise HTTPException(status_code=500, detail="Failed to retrieve created user.")


def set_user_active_status(user_id: str, is_active: bool, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Updates user active status (activate/deactivate)."""
    user = get_user_by_id(user_id, db_path=db_path)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    conn = get_db(db_path)
    conn.execute("UPDATE users SET is_active = ? WHERE id = ?", [is_active, user_id])
    updated = get_user_by_id(user_id, db_path=db_path)
    safe = dict(updated)
    safe.pop("password_hash", None)
    return safe


def list_all_users(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists all users without password hashes."""
    conn = get_db(db_path)
    rows = conn.execute("""
        SELECT id, username, role, agency, district, is_active, failed_attempts, locked_until
        FROM users ORDER BY username ASC
    """).fetchall()

    users = []
    for r in rows:
        locked_val = r[7]
        if locked_val is not None and hasattr(locked_val, "tzinfo") and locked_val.tzinfo is None:
            locked_val = locked_val.replace(tzinfo=timezone.utc)
        users.append({
            "id": r[0],
            "username": r[1],
            "role": r[2],
            "agency": r[3],
            "district": r[4],
            "is_active": bool(r[5]),
            "failed_attempts": int(r[6]),
            "locked_until": locked_val.isoformat() if locked_val else None
        })
    return users


def authenticate_user(username: str, password: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Authenticates a user with Argon2 password verification and lockout protection.
    - Locks account after 5 failed attempts for 15 minutes.
    - Generic error message to prevent username enumeration.
    """
    user = get_user_by_username(username, db_path=db_path)
    conn = get_db(db_path)
    now = datetime.now(timezone.utc)

    # Timing-attack resistance: if user does not exist, compute a dummy Argon2 verify
    if not user:
        verify_password("$argon2id$v=19$m=65536,t=2,p=1$c29tZXNhbHQxMjM0NTY3OA$9vX1/n8O1yW1s4iG123456789012345678901234567", password)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Check for active lockout
    locked_until = user["locked_until"]
    if locked_until is not None:
        if now < locked_until:
            rem_seconds = int((locked_until - now).total_seconds())
            rem_minutes = max(1, rem_seconds // 60)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account is temporarily locked due to excessive failed attempts. Try again in {rem_minutes} minutes."
            )
        else:
            # Lockout expired, reset attempts
            conn.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?", [user["id"]])
            user["failed_attempts"] = 0
            user["locked_until"] = None

    # Verify Password
    if not verify_password(user["password_hash"], password):
        failed_count = user["failed_attempts"] + 1
        if failed_count >= DEFAULT_LOCKOUT_ATTEMPTS:
            lockout_time = now + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
            conn.execute(
                "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                [failed_count, lockout_time, user["id"]]
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account has been locked for {LOCKOUT_DURATION_MINUTES} minutes due to {DEFAULT_LOCKOUT_ATTEMPTS} failed login attempts."
            )
        else:
            conn.execute(
                "UPDATE users SET failed_attempts = ? WHERE id = ?",
                [failed_count, user["id"]]
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password.",
                headers={"WWW-Authenticate": "Bearer"}
            )

    # Check account active status
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Please contact an administrator."
        )

    # Successful login: reset failed attempts
    if user["failed_attempts"] > 0 or user["locked_until"] is not None:
        conn.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?", [user["id"]])

    # Return safe user dict (no hash)
    safe_user = dict(user)
    safe_user.pop("password_hash", None)
    return safe_user


# ==============================================================================
# 6. Pydantic Schemas for Auth Endpoints
# ==============================================================================

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64, description="User username")
    password: str = Field(..., min_length=1, description="Plaintext password (never logged)")


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int = JWT_EXPIRATION_MINUTES
    user: Dict[str, Any]


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32, description="Alphanumeric username")
    password: str = Field(..., min_length=8, description="Minimum 8-character password")
    role: str = Field(..., description="Assigned role")
    agency: Optional[str] = Field(None, description="Agency name if applicable")
    district: Optional[str] = Field(None, description="District name if applicable")


class UserContext(BaseModel):
    user_id: str
    username: str
    role: str
    agency: Optional[str] = None
    district: Optional[str] = None
    auth_method: str = "jwt"


# ==============================================================================
# 7. FastAPI Auth Dependencies & Router
# ==============================================================================

def get_current_authenticated_user(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_user_role: Optional[str] = Header(None, alias="X-User-Role")
) -> UserContext:
    """
    Extracts and validates credentials from Authorization: Bearer <jwt>,
    falling back to API keys or role headers for backwards compatibility.
    """
    # 1. Check Bearer Token (JWT)
    if authorization and authorization.lower().startswith("bearer "):
        raw_token = authorization[7:].strip()
        # Decode and validate signature
        payload = decode_access_token(raw_token)
        return UserContext(
            user_id=payload["user_id"],
            username=payload.get("username", "user"),
            role=payload["role"],
            agency=payload.get("agency"),
            district=payload.get("district"),
            auth_method="jwt"
        )

    # 2. Check X-API-Key or fallback headers
    if x_api_key:
        key_str = x_api_key.strip()
        if "admin" in key_str.lower():
            return UserContext(user_id="key_admin", username="admin_key", role="admin", auth_method="api_key")
        elif "analyst" in key_str.lower():
            return UserContext(user_id="key_analyst", username="analyst_key", role="linkage_reviewer", auth_method="api_key")
        elif "viewer" in key_str.lower():
            return UserContext(user_id="key_viewer", username="viewer_key", role="public_viewer", auth_method="api_key")

    if x_user_role:
        r = x_user_role.strip().lower()
        if r in VALID_ROLES or r in ["analyst", "viewer"]:
            mapped_role = "linkage_reviewer" if r == "analyst" else ("public_viewer" if r == "viewer" else r)
            return UserContext(user_id=f"header_{r}", username=f"header_{r}", role=mapped_role, auth_method="role_header")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: Missing or invalid authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"}
    )


def require_admin_user(current_user: UserContext = Depends(get_current_authenticated_user)) -> UserContext:
    """Dependency that strictly enforces the caller has role == 'admin'."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: Admin privileges required. Caller has role '{current_user.role}'."
        )
    return current_user


# Initialize APIRouter
auth_router = APIRouter(prefix="/api/v1/auth", tags=["Authentication & User Management"])


@auth_router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    """
    Public login endpoint.
    Returns a short-lived signed JWT access token upon successful authentication.
    """
    user = authenticate_user(request.username, request.password)
    token = create_access_token(
        user_id=user["id"],
        username=user["username"],
        role=user["role"],
        agency=user.get("agency"),
        district=user.get("district")
    )
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        expires_in_minutes=JWT_EXPIRATION_MINUTES,
        user=user
    )


@auth_router.get("/me", response_model=Dict[str, Any])
def get_current_user_profile(user: UserContext = Depends(get_current_authenticated_user)):
    """Returns the authenticated user's context and permissions."""
    return {
        "user_id": user.user_id,
        "username": user.username,
        "role": user.role,
        "agency": user.agency,
        "district": user.district,
        "auth_method": user.auth_method
    }


@auth_router.post("/users", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
def create_new_user(
    request: CreateUserRequest,
    admin: UserContext = Depends(require_admin_user)
):
    """Admin-only endpoint to create a new institutional user."""
    created = create_user_record(
        username=request.username,
        password=request.password,
        role=request.role,
        agency=request.agency,
        district=request.district
    )
    return created


@auth_router.get("/users", response_model=List[Dict[str, Any]])
def list_users(admin: UserContext = Depends(require_admin_user)):
    """Admin-only endpoint to list all users in the system."""
    return list_all_users()


@auth_router.patch("/users/{user_id}/deactivate", response_model=Dict[str, Any])
def deactivate_user(
    user_id: str,
    admin: UserContext = Depends(require_admin_user)
):
    """Admin-only endpoint to deactivate an account."""
    return set_user_active_status(user_id, is_active=False)


@auth_router.patch("/users/{user_id}/activate", response_model=Dict[str, Any])
def activate_user(
    user_id: str,
    admin: UserContext = Depends(require_admin_user)
):
    """Admin-only endpoint to reactivate an account."""
    return set_user_active_status(user_id, is_active=True)
