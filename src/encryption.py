"""
src/encryption.py - Application-level Fernet Encryption for Identifying Survivor PII.

Security Guarantees:
1. Encrypts sensitive fields (survivor name, father/husband, guardian name, address,
   localities, DOB, contact details, email, free text/notes) with symmetric Fernet
   encryption before any persistence to database or disk.
2. Supports key rotation via MultiFernet without breaking backward compatibility.
3. Uses DATA_ENCRYPTION_KEY environment variable:
   - In production (CASELENS_ENV=production or RENDER=true): strictly fails closed during
     startup if missing, invalid, or matching a known default/placeholder. Never silently
     generates a replacement key in production.
   - In unknown/unverified environments: strictly fails closed if DATA_ENCRYPTION_KEY is
     unset or insecure. Never silently falls back to development keys.
   - In explicit development/testing modes (CASELENS_ENV=development or ENVIRONMENT=test):
     safely falls back to a deterministic local key or derives a key for testing convenience.
4. Preserves existing ciphertext byte-for-byte during re-processing; never overwrites
   ciphertext with an empty payload.
5. Recursively handles nested dictionary structures to prevent nested PII bypass.
6. Strictly forbids logging of decrypted identifying values, keys, or secrets.
7. All dashboard displays continue to show masked identifiers only.
"""

import os
import json
import base64
import copy
import hashlib
from typing import Dict, Any, List, Tuple, Union
from cryptography.fernet import Fernet, MultiFernet, InvalidToken


# Explicit canonical analytical fields that must NEVER be encrypted
NON_IDENTIFYING_ANALYTICAL_FIELDS = {
    "hospital_name",
    "court_name",
    "police_station",
    "police_station_name",
    "lab_name",
    "division_name",
    "district_name",
    "stage_name",
    "query_name",
    "case_ref",
    "case_id",
    "unique_id",
    "entity_id",
    "fir_number",
    "mlc_number",
    "fsl_case_number",
    "prosecution_file_no",
    "cnr_number",
    "district",
    "incident_date",
    "date_of_incident",
    "fir_date",
    "examination_date",
    "sample_receipt_date",
    "report_dispatch_date",
    "chargesheet_date",
    "verdict_date",
    "offense_category",
    "case_category",
    "injury_type",
    "evidence_type",
    "lab_division",
    "prosecutor_office",
    "case_status",
    "status",
    "stage",
    "age",
    "created_at",
    "updated_at",
    "payload_encrypted"
}

# Explicit canonical identifying fields matching all CaseLens dataset schemas
CANONICAL_IDENTIFYING_FIELDS = {
    # Names (survivor, victim, complainant, patient, subject, petitioner)
    "name",
    "full_name",
    "first_name",
    "last_name",
    "complainant_name",
    "patient_name",
    "subject_name",
    "victim_name",
    "survivor_name",
    "petitioner_victim",
    "petitioner",
    "person_name",
    "complainant",

    # Family & Relations
    "father_or_husband",
    "guardian_name",
    "parent_spouse_name",
    "father_name",
    "husband_name",
    "parent_name",
    "relative_name",
    "relation_name",
    "father",
    "husband",
    "guardian",

    # Addresses & Localities
    "address",
    "residential_address",
    "residential_locality",
    "residence_area",
    "locality",
    "residence",
    "street",
    "home_address",

    # Dates of Birth
    "dob",
    "date_of_birth",
    "birth_date",
    "birthdate",
    "victim_dob",
    "patient_dob",
    "survivor_dob",

    # Contacts & Digital Identifiers
    "phone",
    "mobile",
    "contact_number",
    "phone_number",
    "cell",
    "telephone",
    "email",
    "contact_email",
    "email_address",

    # Government / National Identifiers
    "aadhaar",
    "aadhaar_card",
    "national_id",
    "id_number",

    # Free text & notes containing identifying narratives
    "free_text",
    "case_notes",
    "notes",
    "remarks",
    "statement",
    "confidential_notes",
    "identifying_info",
    "personal_data"
}

# Backward-compatible alias for existing imports
IDENTIFYING_FIELDS = CANONICAL_IDENTIFYING_FIELDS

DEFAULT_LOCAL_KEY_SEED = b"caselens-default-local-encryption-key-seed-2026"
DEFAULT_LOCAL_KEY_B64 = base64.urlsafe_b64encode(hashlib.sha256(DEFAULT_LOCAL_KEY_SEED).digest()).decode("utf-8")

KNOWN_INSECURE_KEYS = {
    DEFAULT_LOCAL_KEY_SEED.decode("utf-8"),
    DEFAULT_LOCAL_KEY_B64,
    "your-data-encryption-key-here",
    "your-32-byte-base64-key-here",
    "changeme",
    "default",
    "secret",
    "test-encryption-key",
    "caselens-default-local-encryption-key-seed-2026",
}


def is_identifying_field(key: str) -> bool:
    """
    Determines whether a field key represents sensitive survivor PII that must be encrypted.
    Protects analytical/institutional fields while matching canonical fields and PII patterns.
    """
    k_lower = str(key).strip().lower()
    if k_lower in NON_IDENTIFYING_ANALYTICAL_FIELDS:
        return False
    if k_lower in CANONICAL_IDENTIFYING_FIELDS:
        return True

    # Institutional entity exceptions (e.g. hospital_name, court_complex)
    if any(k_lower.startswith(p) for p in ["hospital_", "court_", "police_station", "lab_", "division_", "district_"]):
        return False

    identifying_patterns = [
        "name", "victim", "survivor", "complainant", "petitioner", "patient",
        "father", "husband", "guardian", "parent", "spouse", "relative",
        "address", "locality", "residence", "phone", "mobile", "aadhaar",
        "dob", "birth_date", "birthdate", "email", "notes", "statement", "remark"
    ]
    return any(p in k_lower for p in identifying_patterns)


def is_production_mode() -> bool:
    """Checks whether the environment is explicitly configured for production or staging."""
    env = (
        os.environ.get("CASELENS_ENV")
        or os.environ.get("ENVIRONMENT")
        or os.environ.get("APP_ENV")
        or os.environ.get("FASTAPI_ENV")
        or os.environ.get("NODE_ENV")
        or ""
    ).strip().lower()
    is_prod_env = env in ["production", "prod", "staging", "preprod"]
    is_render = os.environ.get("RENDER", "").strip().lower() in ["true", "1"]
    return is_prod_env or is_render


def is_explicit_dev_mode() -> bool:
    """Checks whether the environment is explicitly declared for development or testing."""
    env = (
        os.environ.get("CASELENS_ENV")
        or os.environ.get("ENVIRONMENT")
        or os.environ.get("APP_ENV")
        or os.environ.get("FASTAPI_ENV")
        or os.environ.get("NODE_ENV")
        or ""
    ).strip().lower()
    is_dev_env = env in ["development", "dev", "test", "testing", "local"]
    is_pytest = "PYTEST_CURRENT_TEST" in os.environ
    return is_dev_env or is_pytest


def _clean_key_token(raw_token: str) -> str:
    """Strips outer quotes and whitespace from a key string."""
    token = raw_token.strip()
    while len(token) >= 2 and ((token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'"))):
        token = token[1:-1].strip()
    return token


def get_encryption_keys() -> List[bytes]:
    """
    Validates and returns the active encryption keys.
    Supports comma-separated keys for zero-downtime key rotation via MultiFernet.
    The first key is always the primary key used for all new encryptions.
    """
    raw_env_key = os.environ.get("DATA_ENCRYPTION_KEY", "").strip()
    key_tokens = [_clean_key_token(k) for k in raw_env_key.split(",") if _clean_key_token(k)]

    is_prod = is_production_mode()
    is_dev = is_explicit_dev_mode()

    # 1. Production Mode: Must fail closed if missing, insecure, or malformed
    if is_prod:
        if not key_tokens:
            raise RuntimeError(
                "CRITICAL SECURITY CONFIGURATION ERROR: 'DATA_ENCRYPTION_KEY' environment variable "
                "is missing or empty in production mode! The application cannot safely protect survivor "
                "data without a cryptographically secure 32-byte url-safe base64 Fernet key."
            )

        validated_keys = []
        for token in key_tokens:
            if token.lower() in {k.lower() for k in KNOWN_INSECURE_KEYS} or token == DEFAULT_LOCAL_KEY_B64:
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: 'DATA_ENCRYPTION_KEY' is configured with an insecure "
                    "default, seed, or placeholder value in production mode! A unique, secret 32-byte url-safe base64 "
                    "key generated via cryptography.fernet.Fernet.generate_key() must be provided."
                )
            try:
                k_bytes = token.encode("utf-8")
                Fernet(k_bytes)
                validated_keys.append(k_bytes)
            except Exception:
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: 'DATA_ENCRYPTION_KEY' is invalid in production mode! "
                    "It must be a valid 32-byte url-safe base64-encoded string (e.g., from Fernet.generate_key())."
                )
        return validated_keys

    # 2. Unknown / Unverified Deployment Environment: Do NOT silently fall back to dev key
    if not is_dev:
        if not key_tokens:
            raise RuntimeError(
                "CRITICAL SECURITY CONFIGURATION ERROR: 'DATA_ENCRYPTION_KEY' is not configured in an unverified "
                "deployment environment. To use the development fallback key, explicitly declare "
                "CASELENS_ENV=development or ENVIRONMENT=test."
            )
        validated_keys = []
        for token in key_tokens:
            if token.lower() in {k.lower() for k in KNOWN_INSECURE_KEYS}:
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: 'DATA_ENCRYPTION_KEY' matches an insecure default or "
                    "placeholder in an unverified deployment environment. Set a valid secret key or explicitly "
                    "declare CASELENS_ENV=development."
                )
            try:
                k_bytes = token.encode("utf-8")
                Fernet(k_bytes)
                validated_keys.append(k_bytes)
            except Exception:
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: 'DATA_ENCRYPTION_KEY' is malformed in an unverified "
                    "deployment environment. Set a valid 32-byte url-safe base64 key or declare CASELENS_ENV=development."
                )
        return validated_keys

    # 3. Explicit Development / Testing Mode
    if key_tokens:
        dev_keys = []
        for token in key_tokens:
            if token.lower() in {k.lower() for k in KNOWN_INSECURE_KEYS}:
                dev_keys.append(DEFAULT_LOCAL_KEY_B64.encode("utf-8"))
            else:
                try:
                    k_bytes = token.encode("utf-8")
                    Fernet(k_bytes)
                    dev_keys.append(k_bytes)
                except Exception:
                    # Deterministic fallback derivation for local test passphrases
                    hashed = hashlib.sha256(token.encode("utf-8")).digest()
                    dev_keys.append(base64.urlsafe_b64encode(hashed))
        return dev_keys

    return [DEFAULT_LOCAL_KEY_B64.encode("utf-8")]


def validate_encryption_config() -> bytes:
    """
    Validates DATA_ENCRYPTION_KEY configuration according to the environment.
    Returns the primary encryption key bytes. Fails closed when required.
    """
    keys = get_encryption_keys()
    return keys[0]


def get_fernet_cipher() -> Union[Fernet, MultiFernet]:
    """
    Returns a thread-safe Fernet or MultiFernet cipher initialized with DATA_ENCRYPTION_KEY.
    If multiple comma-separated keys are provided, returns a MultiFernet instance for
    zero-downtime key rotation while maintaining full backward compatibility.
    """
    keys = get_encryption_keys()
    if len(keys) == 1:
        return Fernet(keys[0])
    return MultiFernet([Fernet(k) for k in keys])


def rotate_payload(encrypted_bytes: bytes) -> bytes:
    """
    Re-encrypts a ciphertext token with the primary encryption key using MultiFernet.
    If only one key is configured, returns the ciphertext unchanged.
    """
    cipher = get_fernet_cipher()
    if isinstance(cipher, MultiFernet):
        return cipher.rotate(encrypted_bytes)
    return encrypted_bytes


def encrypt_payload(data: Dict[str, Any]) -> bytes:
    """
    Serializes a dictionary of identifying fields to JSON and encrypts it via Fernet.
    Uses ensure_ascii=False to natively preserve Indic/Unicode characters before UTF-8 encoding.
    Returns ciphertext bytes suitable for storing in a BYTEA column.
    """
    cipher = get_fernet_cipher()
    json_bytes = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return cipher.encrypt(json_bytes)


def decrypt_payload(encrypted_bytes: bytes) -> Dict[str, Any]:
    """
    Decrypts Fernet ciphertext bytes and deserializes JSON to dictionary.
    Raises ValueError if decryption fails or token is tampered with.
    """
    cipher = get_fernet_cipher()
    try:
        decrypted_bytes = cipher.decrypt(encrypted_bytes)
        return json.loads(decrypted_bytes.decode("utf-8"))
    except (InvalidToken, Exception) as e:
        raise ValueError(f"Decryption failed: invalid or tampered ciphertext: {e}")


def split_identifying_data(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Recursively inspects a dictionary and splits it into identifying and non-identifying sub-structures.
    Handles nested dictionaries without leaking sensitive keys into plaintext metadata.
    """
    identifying: Dict[str, Any] = {}
    non_identifying: Dict[str, Any] = {}

    for k, v in data.items():
        if is_identifying_field(k):
            # Identifying key: whole subtree belongs in the encrypted payload
            identifying[k] = copy.deepcopy(v)
        elif isinstance(v, dict):
            sub_id, sub_non_id = split_identifying_data(v)
            if sub_id:
                identifying[k] = sub_id
            if sub_non_id:
                non_identifying[k] = sub_non_id
        else:
            non_identifying[k] = copy.deepcopy(v)

    return identifying, non_identifying


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges update dictionary into base dictionary, preserving nested structures."""
    result = copy.deepcopy(base)
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def split_and_encrypt_case(raw_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Splits a case dictionary into non-identifying fields and an encrypted payload
    containing all identifying fields.

    Guarantees:
    1. If raw_case already contains 'payload_encrypted' and no new plaintext identifying
       fields are provided, the existing ciphertext is preserved byte-for-byte.
    2. If both existing ciphertext and new identifying fields are present, the existing
       ciphertext is safely decrypted, merged with the new fields, and re-encrypted.
    3. Never replaces existing ciphertext with an encrypted empty dictionary.
    4. Handles nested dictionaries recursively.
    """
    case_copy = dict(raw_case)
    existing_payload = case_copy.pop("payload_encrypted", None)

    identifying, non_identifying = split_identifying_data(case_copy)

    if existing_payload is not None:
        existing_bytes = bytes(existing_payload)
        if len(identifying) == 0:
            # Preservation guarantee: retain existing ciphertext byte-for-byte
            non_identifying["payload_encrypted"] = existing_bytes
            return non_identifying
        else:
            # Safe merge and re-encrypt
            existing_identifying = decrypt_payload(existing_bytes)
            merged_identifying = deep_merge(existing_identifying, identifying)
            non_identifying["payload_encrypted"] = encrypt_payload(merged_identifying)
            return non_identifying

    # No existing payload
    non_identifying["payload_encrypted"] = encrypt_payload(identifying)
    return non_identifying


def reconstruct_case(non_identifying: Dict[str, Any], payload_encrypted: bytes) -> Dict[str, Any]:
    """
    Reconstructs full case dictionary by decrypting payload_encrypted and merging.
    Decrypted values are kept in-memory only and never logged.
    """
    identifying = decrypt_payload(payload_encrypted)
    base = dict(non_identifying)
    base.pop("payload_encrypted", None)
    return deep_merge(base, identifying)
