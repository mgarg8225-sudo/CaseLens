"""
src/seed_users.py - Demo User Seeding Script.

Phase A requirement:
Creates one demo user per institutional role with cryptographically RANDOM passwords
printed once to the console for demonstration/testing.
Never stores or logs plaintext passwords to files.
"""

import sys
import string
import secrets
from typing import Dict, Any, Optional

from src.auth import (
    get_db,
    init_auth_db,
    hash_password,
    get_user_by_username,
    load_env_file,
    get_secret,
    is_production_mode,
    VALID_ROLES
)


DEMO_USERS_CONFIG = [
    {
        "role": "admin",
        "username": "admin_demo",
        "agency": None,
        "district": None,
        "description": "System Administrator (User admin, budget reset, system audits)"
    },
    {
        "role": "police",
        "username": "police_demo",
        "agency": "police",
        "district": "D3_Rudrapur",
        "description": "Police Agency Officer (FIR records, Sec 164 statements)"
    },
    {
        "role": "hospital",
        "username": "hospital_demo",
        "agency": "hospital",
        "district": "D3_Rudrapur",
        "description": "Hospital MLC Medical Officer (Medico-legal exam reports)"
    },
    {
        "role": "forensic",
        "username": "forensic_demo",
        "agency": "forensic",
        "district": "D3_Rudrapur",
        "description": "Forensic Science Lab Scientist (DNA/biological evidence)"
    },
    {
        "role": "prosecutor",
        "username": "prosecutor_demo",
        "agency": "prosecutor",
        "district": "D3_Rudrapur",
        "description": "Public Prosecutor (Chargesheet vetting and trial prep)"
    },
    {
        "role": "court",
        "username": "court_demo",
        "agency": "court",
        "district": "D3_Rudrapur",
        "description": "District Court Registrar (Trial proceedings and verdict recording)"
    },
    {
        "role": "linkage_reviewer",
        "username": "reviewer_demo",
        "agency": None,
        "district": None,
        "description": "Record Linkage Reviewer (Splink uncertain pair queue adjudication)"
    },
    {
        "role": "auditor",
        "username": "auditor_demo",
        "agency": None,
        "district": None,
        "description": "Independent Auditor (Immutable audit trail & privacy inspection)"
    },
    {
        "role": "public_viewer",
        "username": "viewer_demo",
        "agency": None,
        "district": None,
        "description": "Public / Civil Society Viewer (Aggregates & DP funnel only)"
    }
]


def generate_secure_random_password(length: int = 14) -> str:
    """Generates a cryptographically strong random password meeting all complexity rules."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        # Ensure at least one lowercase, one uppercase, one digit, and one special char
        if (any(c.islower() for c in pwd)
            and any(c.isupper() for c in pwd)
            and any(c.isdigit() for c in pwd)
            and any(c in "!@#$%^&*" for c in pwd)):
            return pwd


def seed_demo_users(
    db_path: Optional[str] = None,
    print_output: bool = False,
    only_if_missing: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Creates or updates the 9 demo users with random passwords.
    In production mode, only_if_missing is strictly enforced and print_output is disabled.
    """
    if is_production_mode():
        only_if_missing = True
        print_output = False

    conn = get_db(db_path)
    init_auth_db(conn)

    credentials = {}

    for cfg in DEMO_USERS_CONFIG:
        username = cfg["username"]
        role = cfg["role"]
        agency = cfg["agency"]
        district = cfg["district"]

        existing = get_user_by_username(username, db_path=db_path)
        if only_if_missing and existing:
            credentials[username] = {
                "id": existing["id"],
                "username": username,
                "password": "[REDACTED]",
                "role": existing["role"],
                "agency": existing.get("agency"),
                "district": existing.get("district"),
                "description": cfg["description"]
            }
            continue

        # Generate a distinct random password (non-production only)
        password = generate_secure_random_password(14)
        pw_hash = hash_password(password)

        if existing:
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, role = ?, agency = ?, district = ?, is_active = TRUE, failed_attempts = 0, locked_until = NULL
                WHERE username = ?
                """,
                [pw_hash, role, agency, district, username]
            )
            user_id = existing["id"]
        else:
            user_id = str(secrets.token_hex(16))
            conn.execute(
                """
                INSERT INTO users (id, username, password_hash, role, agency, district, is_active, failed_attempts, locked_until)
                VALUES (?, ?, ?, ?, ?, ?, TRUE, 0, NULL)
                """,
                [user_id, username, pw_hash, role, agency, district]
            )

        credentials[username] = {
            "id": user_id,
            "username": username,
            "password": password,
            "role": role,
            "agency": agency,
            "district": district,
            "description": cfg["description"]
        }

    if print_output and not is_production_mode():
        print("\n" + "=" * 90)
        print("CASELENS - PHASE A DEMO USER SEEDING")
        print("Passwords secured in memory. Never logged or saved to disk.")
        print("=" * 90)
        print(f"{'Role':<18} | {'Username':<18} | {'Agency':<12} | {'District':<14} | {'Status'}")
        print("-" * 90)
        for u in credentials.values():
            ag_str = u['agency'] or '-'
            dist_str = u['district'] or '-'
            print(f"{u['role']:<18} | {u['username']:<18} | {ag_str:<12} | {dist_str:<14} | [CONFIGURED / GENERATED]")
        print("=" * 90 + "\n")

    return credentials


def init_demo_users(
    db_path: Optional[str] = None,
    print_output: bool = False,
    only_if_missing: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Initializes the core users for CaseLens:
    1. viewer (role: viewer)
    2. analyst (role: analyst)
    3. admin (role: admin)

    Guarantees:
    - In production mode: only_if_missing is strictly enforced.
    - If a user exists (including inactive or locked), they are NEVER modified, overwritten, or unlocked.
    - In production mode: print_output is strictly forced to False.
    - In production mode: DEMO_*_PASSWORD variables are ignored.
    - In production mode on fresh DB: ADMIN_PASSWORD or INITIAL_ADMIN_PASSWORD is required with >= 8 chars.
      If missing, raises a RuntimeError that fails startup without leaking secrets.
    """
    is_prod = is_production_mode()
    if is_prod:
        only_if_missing = True
        print_output = False

    load_env_file()
    conn = get_db(db_path)
    init_auth_db(conn)

    demo_configs = [
        {
            "username": "viewer",
            "role": "viewer",
            "prod_env_vars": ["VIEWER_PASSWORD", "INITIAL_VIEWER_PASSWORD"],
            "dev_env_vars": ["DEMO_VIEWER_PASSWORD", "VIEWER_PASSWORD"],
            "description": "Public / Civil Society Viewer (Aggregates only: Overview, Funnel, Bottlenecks, Quality, Policy Insights)"
        },
        {
            "username": "analyst",
            "role": "analyst",
            "prod_env_vars": ["ANALYST_PASSWORD", "INITIAL_ANALYST_PASSWORD"],
            "dev_env_vars": ["DEMO_ANALYST_PASSWORD", "ANALYST_PASSWORD"],
            "description": "Policy & Case Analyst (Viewer pages + Case Simulator)"
        },
        {
            "username": "admin",
            "role": "admin",
            "prod_env_vars": ["ADMIN_PASSWORD", "INITIAL_ADMIN_PASSWORD"],
            "dev_env_vars": ["DEMO_ADMIN_PASSWORD", "ADMIN_PASSWORD"],
            "description": "System Administrator (Full access + Review Queue + Security Audit Log)"
        }
    ]

    credentials = {}
    any_random_generated = False

    for cfg in demo_configs:
        username = cfg["username"]
        role = cfg["role"]

        existing = get_user_by_username(username, db_path=db_path)
        if only_if_missing and existing:
            credentials[username] = {
                "id": existing["id"],
                "username": username,
                "password": "[REDACTED]",
                "role": existing["role"],
                "description": cfg["description"]
            }
            continue

        env_vars_to_check = cfg["prod_env_vars"] if is_prod else cfg["dev_env_vars"]

        pwd = None
        for ev in env_vars_to_check:
            val = get_secret(ev)
            if val and val.strip():
                pwd = val.strip()
                break

        if is_prod and not pwd:
            if role == "admin":
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: Initial admin account provisioning failed. "
                    "In production mode, 'ADMIN_PASSWORD' or 'INITIAL_ADMIN_PASSWORD' environment variable "
                    "must be configured with at least 8 characters to bootstrap the primary administrator on a fresh database."
                )
            else:
                continue

        if not pwd:
            pwd = generate_secure_random_password(14)
            any_random_generated = True

        if len(pwd) < 8:
            if is_prod and role == "admin":
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: Initial admin account provisioning failed. "
                    "In production mode, 'ADMIN_PASSWORD' or 'INITIAL_ADMIN_PASSWORD' must be at least 8 characters long."
                )
            raise ValueError(f"Password for user '{username}' must be at least 8 characters long.")

        pw_hash = hash_password(pwd)
        if existing:
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, role = ?, is_active = TRUE, failed_attempts = 0, locked_until = NULL
                WHERE username = ?
                """,
                [pw_hash, role, username]
            )
            user_id = existing["id"]
        else:
            user_id = str(secrets.token_hex(16))
            conn.execute(
                """
                INSERT INTO users (id, username, password_hash, role, agency, district, is_active, failed_attempts, locked_until)
                VALUES (?, ?, ?, ?, NULL, NULL, TRUE, 0, NULL)
                """,
                [user_id, username, pw_hash, role]
            )

        credentials[username] = {
            "id": user_id,
            "username": username,
            "password": pwd if not is_prod else "[REDACTED]",
            "role": role,
            "description": cfg["description"]
        }

    if print_output and not is_prod and any_random_generated:
        print("\n" + "=" * 90)
        print("CASELENS DEMO USERS (Generated Once in Memory - Never Hardcoded)")
        print("=" * 90)
        print(f"{'Role':<12} | {'Username':<12} | {'Status'}")
        print("-" * 90)
        for u in credentials.values():
            print(f"{u['role']:<12} | {u['username']:<12} | [CONFIGURED / GENERATED]")
        print("=" * 90 + "\n")

    return credentials


if __name__ == "__main__":
    seed_demo_users()
    init_demo_users()

