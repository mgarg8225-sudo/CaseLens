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
    print_output: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Creates or updates the 9 demo users with random passwords.
    Prints credentials once to stdout if print_output is True.
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
        print("Random passwords generated once in memory for demo & testing. Never saved to disk.")
        print("=" * 90)
        print(f"{'Role':<18} | {'Username':<18} | {'Agency':<12} | {'District':<14} | {'Generated Password'}")
        print("-" * 90)
        for u in credentials.values():
            ag_str = u['agency'] or '-'
            dist_str = u['district'] or '-'
            print(f"{u['role']:<18} | {u['username']:<18} | {ag_str:<12} | {dist_str:<14} | {u['password']}")
        print("=" * 90 + "\n")

    return credentials


if __name__ == "__main__":
    seed_demo_users()
