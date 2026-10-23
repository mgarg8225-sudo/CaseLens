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
    print_output: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Creates or updates the 9 demo users with random passwords.
    Never prints or logs plaintext passwords.
    Returns in-memory dictionary of demo user credentials.
    """
    conn = get_db(db_path)
    init_auth_db(conn)

    credentials = {}

    for cfg in DEMO_USERS_CONFIG:
        username = cfg["username"]
        role = cfg["role"]
        agency = cfg["agency"]
        district = cfg["district"]

        # Generate a distinct random password
        password = generate_secure_random_password(14)
        pw_hash = hash_password(password)

        existing = get_user_by_username(username, db_path=db_path)
        if existing:
            # Update password hash and reset lockout
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

    if print_output:
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
    only_if_missing: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Initializes the 3 core demo users for CaseLens:
    1. viewer (role: viewer)
    2. analyst (role: analyst)
    3. admin (role: admin)

    Passwords are read securely via get_secret (st.secrets first, then environment):
    - DEMO_VIEWER_PASSWORD (or VIEWER_PASSWORD)
    - DEMO_ANALYST_PASSWORD (or ANALYST_PASSWORD)
    - DEMO_ADMIN_PASSWORD (or ADMIN_PASSWORD)
    If missing, cryptographically strong random passwords are generated in memory.
    Passwords are never hardcoded, printed, or stored in plaintext on disk.
    Argon2id hashes are saved into DuckDB users table.
    """
    load_env_file()
    conn = get_db(db_path)
    init_auth_db(conn)

    demo_configs = [
        {
            "username": "viewer",
            "role": "viewer",
            "env_vars": ["DEMO_VIEWER_PASSWORD", "VIEWER_PASSWORD"],
            "description": "Public / Civil Society Viewer (Aggregates only: Overview, Funnel, Bottlenecks, Quality, Policy Insights)"
        },
        {
            "username": "analyst",
            "role": "analyst",
            "env_vars": ["DEMO_ANALYST_PASSWORD", "ANALYST_PASSWORD"],
            "description": "Policy & Case Analyst (Viewer pages + Case Simulator)"
        },
        {
            "username": "admin",
            "role": "admin",
            "env_vars": ["DEMO_ADMIN_PASSWORD", "ADMIN_PASSWORD"],
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
                "password": "[HIDDEN]",
                "role": role,
                "description": cfg["description"]
            }
            continue

        # Read password from st.secrets or environment
        pwd = None
        for ev in cfg["env_vars"]:
            val = get_secret(ev)
            if val and val.strip():
                pwd = val.strip()
                break

        if not pwd:
            pwd = generate_secure_random_password(14)
            any_random_generated = True

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
            "password": pwd,
            "role": role,
            "description": cfg["description"]
        }

    if print_output and any_random_generated:
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

