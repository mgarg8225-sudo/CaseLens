"""
tests/test_encryption.py - Comprehensive Unit & Security Tests for Application Encryption.

Verifies:
1. Local Development Mode:
   - Safe deterministic fallback key when DATA_ENCRYPTION_KEY is unset or default.
   - Support for custom Fernet keys and developer passphrases.
2. Production Mode (CASELENS_ENV=production or RENDER=true):
   - Fails closed during startup if DATA_ENCRYPTION_KEY is missing or empty.
   - Fails closed if DATA_ENCRYPTION_KEY is set to known defaults, seeds, or placeholders.
   - Fails closed if DATA_ENCRYPTION_KEY is malformed or invalid.
   - Never silently generates or hashes replacement keys in production.
   - Successfully validates and operates with a genuine 32-byte url-safe base64 Fernet key.
3. Security & Zero-PII Isolation:
   - Encryption roundtrip preserves data fidelity.
   - Raw ciphertext bytes never contain plaintext survivor names or addresses.
   - Exception messages never expose secret keys.
4. Canonical Schema PII Fields:
   - father_or_husband, residential_locality, residence_area, victim_dob, patient_dob, email, contact_email.
   - Analytical fields (hospital_name, court_name, fir_number, district) remain non-encrypted.
5. Nested PII Dictionaries:
   - Deeply nested structures are recursively split and restored with 100% fidelity.
6. Ciphertext Preservation & Anti-Destruction:
   - Idempotent re-processing retains existing ciphertext byte-for-byte.
   - Merges new identifying fields safely without overwriting with empty dict.
7. Indic / Unicode Support:
   - Hindi Devanagari script round-trips cleanly with zero leakage.
8. Key Rotation (MultiFernet):
   - Supports comma-separated keys and zero-downtime payload rotation.
9. Unknown Deployment Environment Protection:
   - Fails closed if key is missing or insecure when environment is unspecified.
"""

import os
import pytest
from unittest.mock import patch
from cryptography.fernet import Fernet, MultiFernet

from src.encryption import (
    validate_encryption_config,
    get_encryption_keys,
    get_fernet_cipher,
    rotate_payload,
    encrypt_payload,
    decrypt_payload,
    split_and_encrypt_case,
    reconstruct_case,
    is_production_mode,
    is_explicit_dev_mode,
    DEFAULT_LOCAL_KEY_B64,
    KNOWN_INSECURE_KEYS
)


# ==============================================================================
# 1. Local Development Mode Tests
# ==============================================================================

def test_local_mode_fallback_when_unset():
    """Verifies that in local dev mode, missing key safely falls back to local default."""
    with patch.dict(os.environ, {"CASELENS_ENV": "development"}, clear=False):
        os.environ.pop("DATA_ENCRYPTION_KEY", None)
        os.environ.pop("RENDER", None)

        assert not is_production_mode()
        assert is_explicit_dev_mode()
        key_bytes = validate_encryption_config()
        assert key_bytes == DEFAULT_LOCAL_KEY_B64.encode("utf-8")

        cipher = get_fernet_cipher()
        assert isinstance(cipher, Fernet)


def test_local_mode_valid_fernet_key():
    """Verifies that a valid Fernet key in local dev mode is accepted directly."""
    valid_key = Fernet.generate_key().decode("utf-8")
    with patch.dict(os.environ, {"DATA_ENCRYPTION_KEY": valid_key, "CASELENS_ENV": "development"}):
        key_bytes = validate_encryption_config()
        assert key_bytes == valid_key.encode("utf-8")


def test_local_mode_passphrase_derivation():
    """Verifies that a non-base64 test passphrase in local mode is safely hashed without error."""
    with patch.dict(os.environ, {"DATA_ENCRYPTION_KEY": "my-local-dev-passphrase", "CASELENS_ENV": "development"}):
        key_bytes = validate_encryption_config()
        assert len(key_bytes) == 44  # Base64-encoded 32-byte digest
        cipher = get_fernet_cipher()
        assert isinstance(cipher, Fernet)


def test_local_mode_known_default_fallback():
    """Verifies that setting a known placeholder in local dev falls back safely to default."""
    assert len(KNOWN_INSECURE_KEYS) > 0
    with patch.dict(os.environ, {"DATA_ENCRYPTION_KEY": "your-data-encryption-key-here", "CASELENS_ENV": "development"}):
        key_bytes = validate_encryption_config()
        assert key_bytes == DEFAULT_LOCAL_KEY_B64.encode("utf-8")


def test_encrypt_decrypt_payload_direct():
    """Verifies that encrypt_payload and decrypt_payload serialize and deserialize directly."""
    sample = {"name": "Test Subject", "age": 25, "active": True}
    encrypted = encrypt_payload(sample)
    assert isinstance(encrypted, bytes)
    assert b"Test Subject" not in encrypted

    decrypted = decrypt_payload(encrypted)
    assert decrypted == sample

    # Tampered ciphertext fails with ValueError
    with pytest.raises(ValueError) as exc:
        decrypt_payload(b"gAAAAABtamperedtokeninvalid1234567890==")
    assert "Decryption failed" in str(exc.value)


# ==============================================================================
# 2. Production Mode Fail-Closed Tests
# ==============================================================================

@pytest.mark.parametrize("prod_env_var", [
    {"CASELENS_ENV": "production"},
    {"ENVIRONMENT": "production"},
    {"APP_ENV": "prod"},
    {"RENDER": "true"},
    {"RENDER": "1"},
])
def test_production_mode_missing_key_fails_closed(prod_env_var):
    """Verifies that missing DATA_ENCRYPTION_KEY raises RuntimeError in any production environment."""
    with patch.dict(os.environ, prod_env_var, clear=False):
        os.environ.pop("DATA_ENCRYPTION_KEY", None)
        assert is_production_mode()

        with pytest.raises(RuntimeError) as exc_info:
            validate_encryption_config()

        err_msg = str(exc_info.value)
        assert "CRITICAL SECURITY CONFIGURATION ERROR" in err_msg
        assert "missing or empty" in err_msg


def test_production_mode_empty_or_whitespace_key_fails_closed():
    """Verifies that empty string or whitespace-only key raises RuntimeError in production."""
    with patch.dict(os.environ, {"CASELENS_ENV": "production", "DATA_ENCRYPTION_KEY": "   "}):
        with pytest.raises(RuntimeError) as exc_info:
            validate_encryption_config()

        assert "missing or empty" in str(exc_info.value)


@pytest.mark.parametrize("insecure_key", [
    DEFAULT_LOCAL_KEY_B64,
    "caselens-default-local-encryption-key-seed-2026",
    "your-data-encryption-key-here",
    "your-32-byte-base64-key-here",
    "changeme",
    "default",
    "secret",
    "test-encryption-key",
])
def test_production_mode_known_insecure_defaults_rejected(insecure_key):
    """Verifies that known defaults or placeholders are strictly rejected in production."""
    with patch.dict(os.environ, {"CASELENS_ENV": "production", "DATA_ENCRYPTION_KEY": insecure_key}):
        with pytest.raises(RuntimeError) as exc_info:
            validate_encryption_config()

        err_msg = str(exc_info.value)
        assert "CRITICAL SECURITY CONFIGURATION ERROR" in err_msg
        assert "insecure default" in err_msg or "placeholder" in err_msg


def test_production_mode_invalid_key_format_fails_closed():
    """
    Verifies that an arbitrary non-Fernet string is NOT silently hashed in production;
    it must fail closed immediately.
    """
    with patch.dict(os.environ, {"CASELENS_ENV": "production", "DATA_ENCRYPTION_KEY": "invalid-passphrase-not-fernet"}):
        with pytest.raises(RuntimeError) as exc_info:
            validate_encryption_config()

        err_msg = str(exc_info.value)
        assert "CRITICAL SECURITY CONFIGURATION ERROR" in err_msg
        assert "invalid in production mode" in err_msg


def test_production_mode_valid_fernet_key_succeeds():
    """Verifies that a genuine 32-byte base64 Fernet key succeeds in production."""
    genuine_key = Fernet.generate_key().decode("utf-8")
    with patch.dict(os.environ, {"CASELENS_ENV": "production", "DATA_ENCRYPTION_KEY": genuine_key}):
        assert is_production_mode()
        key_bytes = validate_encryption_config()
        assert key_bytes == genuine_key.encode("utf-8")

        cipher = get_fernet_cipher()
        assert isinstance(cipher, Fernet)


# ==============================================================================
# 3. Encryption Security & Zero-PII Leakage Tests
# ==============================================================================

def test_encryption_roundtrip_and_zero_plaintext_leakage():
    """
    Verifies:
    1. Identifying fields are serialized and encrypted.
    2. Plaintext survivor identifiers never appear in the ciphertext byte payload.
    3. Reconstructing restores 100% data fidelity.
    """
    secret_key = Fernet.generate_key().decode("utf-8")
    with patch.dict(os.environ, {"DATA_ENCRYPTION_KEY": secret_key, "CASELENS_ENV": "production"}):
        raw_case = {
            "case_ref": "CASE-POCSO-2026-0042",
            "victim_name": "Aaradhya Sen",
            "guardian_name": "Vikram Sen",
            "address": "Flat 302, Green Park, Shivgarh",
            "phone": "+91-9876543210",
            "dob": "2008-11-23",
            "case_notes": "Survivor reported acute delay in forensic sample collection.",
            "district": "Shivgarh",
            "category": "Aggravated Sexual Assault",
            "status": "investigation",
            "stage": "POLICE_FIR"
        }

        # Encrypt
        encrypted = split_and_encrypt_case(raw_case)
        payload = encrypted["payload_encrypted"]
        assert isinstance(payload, bytes)

        # Non-identifying metadata remains accessible
        assert encrypted["case_ref"] == "CASE-POCSO-2026-0042"
        assert encrypted["district"] == "Shivgarh"
        assert encrypted["category"] == "Aggravated Sexual Assault"

        # Identifying fields are NOT in plaintext anywhere in the raw bytes
        assert b"Aaradhya Sen" not in payload
        assert b"Vikram Sen" not in payload
        assert b"Green Park" not in payload
        assert b"9876543210" not in payload
        assert b"acute delay" not in payload

        # Identifying fields are removed from the outer dict
        assert "victim_name" not in encrypted
        assert "guardian_name" not in encrypted
        assert "address" not in encrypted
        assert "phone" not in encrypted
        assert "case_notes" not in encrypted

        # Decrypt and verify complete round-trip reconstruction
        reconstructed = reconstruct_case(encrypted, payload)
        assert reconstructed["victim_name"] == "Aaradhya Sen"
        assert reconstructed["guardian_name"] == "Vikram Sen"
        assert reconstructed["address"] == "Flat 302, Green Park, Shivgarh"
        assert reconstructed["phone"] == "+91-9876543210"
        assert reconstructed["dob"] == "2008-11-23"
        assert reconstructed["case_notes"] == "Survivor reported acute delay in forensic sample collection."
        assert reconstructed["district"] == "Shivgarh"


def test_exception_never_logs_or_leaks_key():
    """Verifies that failure exceptions never reflect secret key values."""
    secret_candidate = "my-secret-that-must-never-be-echoed-12345"
    with patch.dict(os.environ, {"CASELENS_ENV": "production", "DATA_ENCRYPTION_KEY": secret_candidate}):
        with pytest.raises(RuntimeError) as exc_info:
            validate_encryption_config()
        # Exception message must NEVER contain the secret candidate
        assert secret_candidate not in str(exc_info.value)


# ==============================================================================
# 4. Canonical Dataset PII Fields Coverage Tests
# ==============================================================================

def test_canonical_dataset_pii_fields_encrypted():
    """
    Verifies that all canonical schema columns identified in security audit
    (father_or_husband, residential_locality, residence_area, victim_dob,
    patient_dob, email, contact_email) are encrypted into payload_encrypted
    and completely excluded from plaintext metadata.
    """
    case = {
        "fir_number": "FIR-2026-0101",
        "case_ref": "CASE-0101",
        "district": "North",
        "father_or_husband": "Ramesh Kumar Sharma",
        "residential_locality": "Sector 4, Rohini",
        "residence_area": "Ward 12, Block C",
        "victim_dob": "2006-03-15",
        "patient_dob": "2006-03-15",
        "email": "survivor.advocate@example.com",
        "contact_email": "case.worker@example.org",
        "hospital_name": "Deen Dayal Upadhyay Hospital",
        "court_name": "Rohini District Court",
        "police_station": "Samaypur Badli",
        "offense_category": "BNS Section 64"
    }

    encrypted = split_and_encrypt_case(case)
    payload = encrypted["payload_encrypted"]
    assert isinstance(payload, bytes)

    # 1. Verify analytical fields remain accessible in plaintext metadata
    assert encrypted["fir_number"] == "FIR-2026-0101"
    assert encrypted["case_ref"] == "CASE-0101"
    assert encrypted["district"] == "North"
    assert encrypted["hospital_name"] == "Deen Dayal Upadhyay Hospital"
    assert encrypted["court_name"] == "Rohini District Court"
    assert encrypted["police_station"] == "Samaypur Badli"
    assert encrypted["offense_category"] == "BNS Section 64"

    # 2. Verify all sensitive canonical fields are strictly removed from metadata
    assert "father_or_husband" not in encrypted
    assert "residential_locality" not in encrypted
    assert "residence_area" not in encrypted
    assert "victim_dob" not in encrypted
    assert "patient_dob" not in encrypted
    assert "email" not in encrypted
    assert "contact_email" not in encrypted

    # 3. Verify zero plaintext leakage inside ciphertext bytes
    assert b"Ramesh Kumar Sharma" not in payload
    assert b"Sector 4" not in payload
    assert b"Ward 12" not in payload
    assert b"2006-03-15" not in payload
    assert b"survivor.advocate@example.com" not in payload

    # 4. Verify 100% round-trip reconstruction fidelity
    reconstructed = reconstruct_case(encrypted, payload)
    assert reconstructed == case


# ==============================================================================
# 5. Nested PII Dictionaries & Deep Merge Tests
# ==============================================================================

def test_nested_pii_dictionaries_recursively_encrypted():
    """
    Verifies that nested dictionary structures containing PII fields are
    recursively split so sensitive keys never leak into plaintext metadata,
    while non-identifying nested fields and structures are preserved.
    """
    complex_case = {
        "case_ref": "CASE-NESTED-99",
        "district": "Central",
        "record_details": {
            "patient_name": "Meera Patel",
            "father_or_husband": "Anand Patel",
            "sub_profile": {
                "patient_dob": "2007-08-20",
                "phone": "+91-9123456789",
                "non_pii_score": 88.5
            }
        },
        "court_metadata": {
            "court_name": "Tis Hazari Court",
            "case_notes": "Immediate counseling order passed."
        }
    }

    encrypted = split_and_encrypt_case(complex_case)
    payload = encrypted["payload_encrypted"]

    # In outer dictionary
    assert encrypted["case_ref"] == "CASE-NESTED-99"
    assert encrypted["district"] == "Central"

    # In court_metadata: court_name is non-identifying, but case_notes is identifying
    assert encrypted["court_metadata"]["court_name"] == "Tis Hazari Court"
    assert "case_notes" not in encrypted["court_metadata"]

    # In record_details: patient_name, father_or_husband, patient_dob, phone are encrypted
    # Only non_pii_score remains in nested metadata
    assert "patient_name" not in encrypted["record_details"]
    assert "father_or_husband" not in encrypted["record_details"]
    assert encrypted["record_details"]["sub_profile"]["non_pii_score"] == 88.5
    assert "patient_dob" not in encrypted["record_details"]["sub_profile"]
    assert "phone" not in encrypted["record_details"]["sub_profile"]

    # Ciphertext zero-leakage
    assert b"Meera Patel" not in payload
    assert b"Anand Patel" not in payload
    assert b"9123456789" not in payload
    assert b"counseling order" not in payload

    # Full round-trip reconstruction
    reconstructed = reconstruct_case(encrypted, payload)
    assert reconstructed == complex_case


# ==============================================================================
# 6. Ciphertext Preservation & Anti-Destruction Tests
# ==============================================================================

def test_split_and_encrypt_preserves_existing_ciphertext_byte_for_byte():
    """
    Verifies that passing an already-encrypted case dictionary back into
    split_and_encrypt_case() preserves the existing ciphertext byte-for-byte
    and never overwrites it with an empty payload.
    """
    raw_case = {
        "case_ref": "CASE-PRESERVE-01",
        "victim_name": "Kavita Devi",
        "district": "East"
    }

    # Step 1: Initial encryption
    enc1 = split_and_encrypt_case(raw_case)
    initial_payload = enc1["payload_encrypted"]
    assert isinstance(initial_payload, bytes)
    assert decrypt_payload(initial_payload) == {"victim_name": "Kavita Devi"}

    # Step 2: Re-run split_and_encrypt_case without new plaintext identifying fields
    enc2 = split_and_encrypt_case(enc1)
    second_payload = enc2["payload_encrypted"]

    # Must be byte-for-byte identical
    assert initial_payload == second_payload
    assert decrypt_payload(second_payload) == {"victim_name": "Kavita Devi"}

    # Step 3: Run repeatedly in a loop 5 times
    current = enc2
    for _ in range(5):
        current = split_and_encrypt_case(current)
        assert current["payload_encrypted"] == initial_payload

    # Reconstructed data remains 100% intact
    reconstructed = reconstruct_case(current, current["payload_encrypted"])
    assert reconstructed["victim_name"] == "Kavita Devi"
    assert reconstructed["case_ref"] == "CASE-PRESERVE-01"


def test_split_and_encrypt_safe_merge_with_new_fields():
    """
    Verifies that when a record already contains ciphertext AND new plaintext
    identifying fields are supplied, the function safely decrypts existing fields,
    merges new fields, and re-encrypts without data loss.
    """
    case = {"case_ref": "CASE-MERGE-01", "victim_name": "Sunita"}
    enc = split_and_encrypt_case(case)

    # Add a new plaintext identifying field to the already-encrypted record
    enc["phone"] = "+91-9999988888"
    enc["father_or_husband"] = "Rajesh"

    merged_enc = split_and_encrypt_case(enc)
    merged_payload = merged_enc["payload_encrypted"]

    # Plaintext fields stripped from outer dict
    assert "phone" not in merged_enc
    assert "father_or_husband" not in merged_enc

    # Both old and new fields present in decrypted payload
    decrypted = decrypt_payload(merged_payload)
    assert decrypted["victim_name"] == "Sunita"
    assert decrypted["phone"] == "+91-9999988888"
    assert decrypted["father_or_husband"] == "Rajesh"


# ==============================================================================
# 7. Indic / Unicode Support Tests
# ==============================================================================

def test_hindi_devanagari_unicode_fidelity():
    """
    Verifies that Hindi / Devanagari text in survivor names, guardian names,
    localities, and case notes round-trips with 100% character fidelity
    and does not appear in ciphertext bytes.
    """
    hindi_case = {
        "case_ref": "CASE-HINDI-2026",
        "victim_name": "प्रिया शर्मा",
        "father_or_husband": "रमेश कुमार शर्मा",
        "residential_locality": "गांधी नगर, वार्ड नंबर ५",
        "case_notes": "पीड़िता को तत्काल कानूनी और चिकित्सीय सहायता प्रदान की गई।",
        "district": "नई दिल्ली"
    }

    encrypted = split_and_encrypt_case(hindi_case)
    payload = encrypted["payload_encrypted"]

    # Plaintext Devanagari UTF-8 bytes must NOT appear in ciphertext
    assert "प्रिया शर्मा".encode("utf-8") not in payload
    assert "गांधी नगर".encode("utf-8") not in payload

    # Reconstruct and check exact character equality
    reconstructed = reconstruct_case(encrypted, payload)
    assert reconstructed["victim_name"] == "प्रिया शर्मा"
    assert reconstructed["father_or_husband"] == "रमेश कुमार शर्मा"
    assert reconstructed["residential_locality"] == "गांधी नगर, वार्ड नंबर ५"
    assert reconstructed["case_notes"] == "पीड़िता को तत्काल कानूनी और चिकित्सीय सहायता प्रदान की गई।"


# ==============================================================================
# 8. Key Rotation (MultiFernet) Tests
# ==============================================================================

def test_multi_fernet_key_rotation_and_backward_compatibility():
    """
    Verifies that setting comma-separated keys in DATA_ENCRYPTION_KEY:
    1. Successfully initializes MultiFernet.
    2. Decrypts payloads created with the older secondary key.
    3. Encrypts all new payloads with the primary (first) key.
    4. Allows rotate_payload() to migrate ciphertext to the primary key.
    """
    primary_key = Fernet.generate_key().decode("utf-8")
    old_key = Fernet.generate_key().decode("utf-8")

    # Step 1: Encrypt a payload using ONLY the old key
    old_cipher = Fernet(old_key.encode("utf-8"))
    old_token = old_cipher.encrypt(b'{"victim_name": "Archived Case"}')

    # Step 2: Configure dual keys: "primary,old"
    dual_key_config = f"{primary_key}, {old_key}"
    with patch.dict(os.environ, {"DATA_ENCRYPTION_KEY": dual_key_config, "CASELENS_ENV": "development"}):
        keys = get_encryption_keys()
        assert len(keys) == 2
        assert keys[0] == primary_key.encode("utf-8")
        assert keys[1] == old_key.encode("utf-8")

        cipher = get_fernet_cipher()
        assert isinstance(cipher, MultiFernet)

        # MultiFernet can decrypt the payload created by the old key
        decrypted = decrypt_payload(old_token)
        assert decrypted == {"victim_name": "Archived Case"}

        # MultiFernet encrypts new payloads using the primary key
        new_token = encrypt_payload({"victim_name": "New Case"})
        assert Fernet(primary_key.encode("utf-8")).decrypt(new_token) == b'{"victim_name": "New Case"}'

        # Re-encrypt / rotate old token to primary key
        rotated_token = rotate_payload(old_token)
        assert Fernet(primary_key.encode("utf-8")).decrypt(rotated_token) == b'{"victim_name": "Archived Case"}'


# ==============================================================================
# 9. Unknown Deployment Environment Fail-Closed Tests
# ==============================================================================

def test_unknown_environment_missing_key_fails_closed():
    """
    Verifies that in an unspecified / unknown deployment environment (neither
    production nor explicit development), a missing key strictly raises RuntimeError.
    """
    with patch.dict(os.environ, {}, clear=True):
        # Ensure neither prod nor dev flags exist
        assert not is_production_mode()
        assert not is_explicit_dev_mode()

        with pytest.raises(RuntimeError) as exc_info:
            validate_encryption_config()

        err_msg = str(exc_info.value)
        assert "CRITICAL SECURITY CONFIGURATION ERROR" in err_msg
        assert "unverified deployment environment" in err_msg


def test_unknown_environment_insecure_key_fails_closed():
    """
    Verifies that in an unknown deployment environment, an insecure placeholder key
    strictly raises RuntimeError and does NOT silently fall back.
    """
    with patch.dict(os.environ, {"DATA_ENCRYPTION_KEY": "your-data-encryption-key-here"}, clear=True):
        assert not is_production_mode()
        assert not is_explicit_dev_mode()

        with pytest.raises(RuntimeError) as exc_info:
            validate_encryption_config()

        err_msg = str(exc_info.value)
        assert "CRITICAL SECURITY CONFIGURATION ERROR" in err_msg
        assert "insecure default or placeholder" in err_msg


def test_unknown_environment_valid_key_succeeds():
    """Verifies that in an unknown deployment environment, a genuine Fernet key succeeds."""
    valid_key = Fernet.generate_key().decode("utf-8")
    with patch.dict(os.environ, {"DATA_ENCRYPTION_KEY": valid_key}, clear=True):
        assert not is_production_mode()
        assert not is_explicit_dev_mode()

        key_bytes = validate_encryption_config()
        assert key_bytes == valid_key.encode("utf-8")
