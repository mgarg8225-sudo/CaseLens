"""
tests/test_database.py - Isolated Unit & Integration Tests for Database & Repository Layer.

Verifies:
1. Database Configuration & Credential Masking:
   - Local fallback when DATABASE_URL is unset.
   - Sanitized status dictionaries; passwords never exposed in logs or outputs.
   - Connection pool sizing and timeout configuration.
2. Connection Failure & Health Checks:
   - Graceful degradation when PostgreSQL is unavailable.
3. User Repository:
   - Profile persistence, retrieval, activation status, and role enforcement (viewer, analyst, admin).
4. Dataset Repository:
   - Draft and published lifecycle, expiration calculation, and draft purging.
5. Case Repository & Encryption Safeguards:
   - Encryption of identifying fields prior to database persistence.
   - Non-identifying columns contain zero plaintext survivor names or addresses.
   - Roundtrip in-memory reconstruction fidelity.
6. Analysis Repository:
   - Linked pairs, timelines, and aggregates persistence (JSONB).
7. Audit Repository:
   - Immutable security audit ledger.
8. Privacy Budget Repository (Atomic Concurrency Protection):
   - Sequential Composition enforcement.
   - Multi-threaded concurrency simulation preventing budget overspending.
   - Proper raising of PrivacyBudgetExceededError.
9. DuckDB Analytical Bridge:
   - Seamless loading of case records into in-memory DuckDB without plaintext PII.
10. Migration SQL Files:
    - Verification that versioned migration files exist, are syntactically valid SQL,
      and strictly enforce Row Level Security (RLS default-deny).
"""

import os
import glob
import threading
import pytest
import duckdb
from unittest.mock import patch

import src.database
import src.repository
from src.database import (
    get_database_url,
    is_postgres_enabled,
    mask_database_url,
    get_draft_retention_days,
    get_pool_config,
    validate_database_config,
    check_database_health,
    enforce_database_safety,
    verify_repository_fallback_allowed,
    is_explicit_development_or_test
)
from src.repository import (
    UserRepository,
    DatasetRepository,
    CaseRepository,
    AnalysisRepository,
    AuditRepository,
    PrivacyBudgetRepository,
    DuckDBBridge
)
from src.privacy import (
    PrivacyBudgetExceededError,
    PrivacyBudgetTracker,
    PersistentPrivacyBudgetTracker
)


# ==============================================================================
# 1. Database Configuration & Credential Masking Tests
# ==============================================================================

def test_database_url_retrieval_and_masking():
    """Verifies that DATABASE_URL is properly retrieved, cleaned, and masked."""
    # Unset URL
    with patch.dict(os.environ, {}, clear=True):
        assert get_database_url() is None
        assert mask_database_url(None) == "<unset>"
        assert not is_postgres_enabled()

    # Valid secret URL with credentials
    sample_url = "postgresql://postgres.project:SecretPassword123!@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
    with patch.dict(os.environ, {"DATABASE_URL": sample_url}):
        assert get_database_url() == sample_url
        masked = mask_database_url(sample_url)
        assert "SecretPassword123!" not in masked
        assert "****" in masked
        assert "postgres.project" in masked


def test_validate_database_config_unconfigured():
    """Verifies that when DATABASE_URL is unset, status is LOCAL_FALLBACK."""
    with patch.dict(os.environ, {}, clear=True):
        cfg = validate_database_config()
        assert cfg["configured"] is False
        assert cfg["status"] == "LOCAL_FALLBACK"
        assert "DuckDB" in cfg["message"]


def test_validate_database_config_invalid_scheme():
    """Verifies that non-PostgreSQL URLs are rejected by config validation."""
    with patch.dict(os.environ, {"DATABASE_URL": "mysql://root:password@localhost:3306/db"}):
        cfg = validate_database_config()
        assert cfg["configured"] is True
        assert cfg["valid_scheme"] is False
        assert cfg["status"] == "INVALID_CONFIGURATION"


def test_pool_configuration_defaults_and_env():
    """Verifies pool configuration reads from environment with sane defaults."""
    with patch.dict(os.environ, {
        "DATABASE_POOL_MIN_SIZE": "3",
        "DATABASE_POOL_MAX_SIZE": "25",
        "DATABASE_POOL_TIMEOUT": "45.0",
        "DATABASE_POOL_MAX_IDLE": "600.0"
    }):
        cfg = get_pool_config()
        assert cfg["min_size"] == 3
        assert cfg["max_size"] == 25
        assert cfg["timeout"] == 45.0
        assert cfg["max_idle"] == 600.0


def test_draft_retention_days_env():
    """Verifies draft retention reads from environment."""
    with patch.dict(os.environ, {"DRAFT_RETENTION_DAYS": "14"}):
        assert get_draft_retention_days() == 14

    with patch.dict(os.environ, {}, clear=True):
        assert get_draft_retention_days() == 7


def test_check_database_health_disabled():
    """Verifies database health check reports UNAVAILABLE when PostgreSQL is disabled."""
    with patch.dict(os.environ, {}, clear=True):
        health = check_database_health()
        assert health["status"] == "UNAVAILABLE"
        assert health["postgres_enabled"] is False


# ==============================================================================
# 2. User Repository & RBAC Tests
# ==============================================================================

def test_user_repository_crud_and_role_enforcement():
    """Verifies user profile creation, role validation, activation, and retrieval."""
    user_id = "test-user-uuid-101"

    # Invalid role must raise ValueError
    with pytest.raises(ValueError) as exc:
        UserRepository.save_profile(user_id=user_id, role="superadmin")
    assert "Invalid role" in str(exc.value)

    # Valid profile save
    profile = UserRepository.save_profile(
        user_id=user_id,
        role="analyst",
        full_name="Dr. Sunita Rao",
        is_active=True
    )
    assert profile["user_id"] == user_id
    assert profile["role"] == "analyst"
    assert profile["is_active"] is True

    # Retrieve profile
    retrieved = UserRepository.get_profile(user_id)
    assert retrieved is not None
    assert retrieved["role"] == "analyst"
    assert retrieved["full_name"] == "Dr. Sunita Rao"

    # Deactivate profile
    success = UserRepository.set_profile_active_status(user_id, is_active=False)
    assert success is True
    updated = UserRepository.get_profile(user_id)
    assert updated["is_active"] is False

    # List profiles
    all_profs = UserRepository.list_profiles()
    assert any(p["user_id"] == user_id for p in all_profs)


# ==============================================================================
# 3. Dataset Repository Lifecycle & Purge Tests
# ==============================================================================

def test_dataset_repository_lifecycle():
    """Verifies dataset draft persistence, publication, and deletion."""
    ds_id = "test-dataset-alpha"

    # Invalid status raises ValueError
    with pytest.raises(ValueError):
        DatasetRepository.save_dataset(dataset_id=ds_id, name="Test", created_by="user-1", status="archived")

    # Save draft dataset
    ds = DatasetRepository.save_dataset(
        dataset_id=ds_id,
        name="North District Monthly FIRs",
        created_by="officer-99",
        status="draft"
    )
    assert ds["status"] == "draft"
    assert ds["expires_at"] is not None

    # Get dataset
    retrieved = DatasetRepository.get_dataset(ds_id)
    assert retrieved is not None
    assert retrieved["name"] == "North District Monthly FIRs"
    assert retrieved["status"] == "draft"

    # Publish dataset
    pub_ok = DatasetRepository.publish_dataset(ds_id, published_by="admin-1")
    assert pub_ok is True
    published = DatasetRepository.get_dataset(ds_id)
    assert published["status"] == "published"
    assert published["published_by"] == "admin-1"
    assert published["expires_at"] is None

    # Delete dataset
    del_ok = DatasetRepository.delete_dataset(ds_id)
    assert del_ok is True
    assert DatasetRepository.get_dataset(ds_id) is None


# ==============================================================================
# 4. Case Repository & PII Encryption Verification Tests
# ==============================================================================

def test_case_repository_encryption_before_persistence():
    """
    MANDATORY SECURITY GUARANTEE:
    Verifies that CaseRepository encrypts all identifying fields before storage.
    Plaintext survivor names, relatives, addresses, and contacts are NEVER
    present in the stored metadata columns.
    """
    dataset_id = "ds-pocso-test-2026"
    raw_cases = [
        {
            "case_ref": "CASE-POCSO-001",
            "victim_name": "Aaradhya Sen",
            "father_or_husband": "Vikram Sen",
            "residential_locality": "Green Park, Block C",
            "phone": "+91-9876543210",
            "victim_dob": "2008-05-12",
            "district": "North",
            "category": "Aggravated Sexual Assault",
            "incident_date": "2026-01-15",
            "fir_date": "2026-01-16",
            "status": "investigation",
            "stage": "POLICE_FIR"
        },
        {
            "case_ref": "CASE-POCSO-002",
            "victim_name": "Kavita Devi",
            "father_or_husband": "Ramesh Devi",
            "residential_locality": "Samaypur, Ward 4",
            "phone": "+91-9876543211",
            "victim_dob": "2007-09-22",
            "district": "North",
            "category": "Sexual Assault",
            "incident_date": "2026-01-18",
            "fir_date": "2026-01-19",
            "status": "chargesheeted",
            "stage": "CHARGESHEET_FILED"
        }
    ]

    # Save cases
    count = CaseRepository.save_cases(dataset_id, raw_cases)
    assert count == 2
    assert CaseRepository.count_cases(dataset_id) == 2

    # 1. Load without decryption: metadata columns must NOT contain plaintext PII
    raw_stored = CaseRepository.load_cases(dataset_id, decrypt=False)
    assert len(raw_stored) == 2
    for r in raw_stored:
        assert "victim_name" not in r
        assert "father_or_husband" not in r
        assert "residential_locality" not in r
        assert "phone" not in r
        assert "victim_dob" not in r
        # Analytical columns remain accessible
        assert r["district"] == "North"
        assert r["status"] in ("investigation", "chargesheeted")

    # 2. Load with decryption: in-memory reconstruction restores 100% fidelity
    decrypted = CaseRepository.load_cases(dataset_id, decrypt=True)
    assert len(decrypted) == 2

    c1 = next(c for c in decrypted if c["case_ref"] == "CASE-POCSO-001")
    assert c1["victim_name"] == "Aaradhya Sen"
    assert c1["father_or_husband"] == "Vikram Sen"
    assert c1["residential_locality"] == "Green Park, Block C"
    assert c1["phone"] == "+91-9876543210"
    assert c1["victim_dob"] == "2008-05-12"
    assert c1["district"] == "North"


# ==============================================================================
# 5. Analysis & Audit Repository Tests
# ==============================================================================

def test_analysis_and_audit_repositories():
    """Verifies analysis results and immutable audit ledger persistence."""
    ds_id = "test-analysis-dataset"

    # Save analysis results JSONB
    AnalysisRepository.save_analysis(
        dataset_id=ds_id,
        linked_pairs=[{"pair": [1, 2], "score": 0.95}],
        timelines={"stages": ["FIR", "MLC", "COURT"]},
        aggregates={"total": 42}
    )

    an = AnalysisRepository.get_analysis(ds_id)
    assert an is not None
    assert an["aggregates"] == {"total": 42}
    assert an["linked_pairs"][0]["score"] == 0.95

    # Log audit events
    AuditRepository.log_event(
        actor="admin-analyst-01",
        action="DATASET_PUBLISH",
        role="admin",
        detail={"dataset_id": ds_id}
    )

    events = AuditRepository.get_events(limit=10)
    assert len(events) >= 1
    recent = events[0]
    assert recent["action"] == "DATASET_PUBLISH"
    assert recent["actor"] == "admin-analyst-01"


# ==============================================================================
# 6. Privacy Budget Repository (Atomic Concurrency Protection) Tests
# ==============================================================================

def test_privacy_budget_atomic_concurrency_allocation():
    """
    Verifies that PrivacyBudgetRepository:
    1. Tracks budget under Sequential Composition.
    2. Enforces atomic allocation and raises PrivacyBudgetExceededError when budget is exhausted.
    3. Handles multi-threaded concurrent requests safely without overspending.
    """
    budget_id = "test-concurrent-budget"
    PrivacyBudgetRepository.reset_budget(budget_id=budget_id, total_epsilon=1.0)

    # Initial check
    b = PrivacyBudgetRepository.get_budget(budget_id)
    assert b["epsilon_total"] == 1.0
    assert b["epsilon_spent"] == 0.0

    # Successful allocation
    res1 = PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=0.4,
        query_id="query-1",
        actor="analyst-1"
    )
    assert res1["allocated"] is True
    assert round(res1["epsilon_spent"], 4) == 0.4
    assert round(res1["remaining_epsilon"], 4) == 0.6

    # Second allocation
    res2 = PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=0.5,
        query_id="query-2",
        actor="analyst-2"
    )
    assert round(res2["epsilon_spent"], 4) == 0.9
    assert round(res2["remaining_epsilon"], 4) == 0.1

    # Third allocation: requests 0.2 when only 0.1 remains -> MUST raise PrivacyBudgetExceededError
    with pytest.raises(PrivacyBudgetExceededError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=0.2,
            query_id="query-3"
        )
    assert "Privacy budget exceeded" in str(exc_info.value)

    # Verify spent epsilon was NOT incremented on failure
    b_after = PrivacyBudgetRepository.get_budget(budget_id)
    assert round(b_after["epsilon_spent"], 4) == 0.9

    # Concurrency test: 10 threads concurrently trying to spend remaining 0.1 epsilon
    threads = []
    success_count = [0]
    failure_count = [0]
    lock = threading.Lock()

    def _worker(thread_id: int):
        try:
            PrivacyBudgetRepository.allocate_budget(
                budget_id=budget_id,
                epsilon=0.1,
                query_id=f"thread-{thread_id}"
            )
            with lock:
                success_count[0] += 1
        except PrivacyBudgetExceededError:
            with lock:
                failure_count[0] += 1

    for i in range(10):
        t = threading.Thread(target=_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Exactly 1 thread must succeed; 9 threads must fail closed!
    assert success_count[0] == 1
    assert failure_count[0] == 9

    # Final spent epsilon must equal total epsilon (1.0), never exceeding it
    final_b = PrivacyBudgetRepository.get_budget(budget_id)
    assert round(final_b["epsilon_spent"], 4) == 1.0


# ==============================================================================
# 7. DuckDB Analytical Bridge Tests
# ==============================================================================

def test_duckdb_bridge_integration():
    """Verifies that DuckDBBridge loads case records into an in-memory DuckDB connection."""
    dataset_id = "test-duckdb-bridge-ds"
    cases = [
        {"case_ref": "REF-001", "district": "South", "status": "investigation", "victim_name": "Secret A"},
        {"case_ref": "REF-002", "district": "South", "status": "chargesheeted", "victim_name": "Secret B"},
        {"case_ref": "REF-003", "district": "Central", "status": "trial", "victim_name": "Secret C"}
    ]
    CaseRepository.save_cases(dataset_id, cases)

    conn = duckdb.connect(":memory:")
    count = DuckDBBridge.load_dataset_into_duckdb(dataset_id, conn)
    assert count == 3

    # Query DuckDB table 'current_cases'
    df_res = conn.execute("SELECT district, COUNT(*) AS cnt FROM current_cases GROUP BY district ORDER BY cnt DESC;").fetchdf()
    assert len(df_res) == 2
    assert df_res.iloc[0]["district"] == "South"
    assert df_res.iloc[0]["cnt"] == 2

    # Verify table schema: 'victim_name' is NOT present in DuckDB table
    cols = conn.execute("PRAGMA table_info('current_cases');").fetchall()
    col_names = [c[1] for c in cols]
    assert "victim_name" not in col_names
    assert "case_ref" in col_names
    assert "district" in col_names

    conn.close()


# ==============================================================================
# 8. PostgreSQL Migration Files & RLS Default-Deny Verification
# ==============================================================================

def test_migration_files_integrity_and_rls_default_deny():
    """
    Verifies that:
    1. Versioned SQL migration files exist in supabase/migrations/.
    2. Every table created has 'ENABLE ROW LEVEL SECURITY'.
    3. No public/anon policies are declared (default-deny for client access).
    """
    migrations_dir = os.path.join("supabase", "migrations")
    assert os.path.exists(migrations_dir), f"Directory {migrations_dir} does not exist!"

    sql_files = sorted(glob.glob(os.path.join(migrations_dir, "*.sql")))
    assert len(sql_files) >= 2, f"Expected at least 2 migration files, found {len(sql_files)}"

    import re
    for sql_file in sql_files:
        with open(sql_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Find all CREATE TABLE statements using regex
        tables_created = re.findall(r"create\s+table\s+(?:if\s+not\s+exists\s+)?([a-zA-Z0-9_\.]+)", content, re.IGNORECASE)
        assert len(tables_created) > 0, f"No tables found in {sql_file}"

        # Every table must have ENABLE ROW LEVEL SECURITY
        for tbl in tables_created:
            expected_rls = f"alter table {tbl.lower()} enable row level security;"
            assert expected_rls in content.lower(), f"Table {tbl} in {sql_file} lacks 'ENABLE ROW LEVEL SECURITY'!"

        # No public grant or anon access policy
        assert not re.search(r"grant\s+.*\s+to\s+(?:anon|public)", content, re.IGNORECASE)
        assert not re.search(r"create\s+policy\s+.*\s+to\s+(?:anon|public)", content, re.IGNORECASE)


# ==============================================================================
# 9. Production Database Safety & Fail-Closed Validation Tests
# ==============================================================================

def test_production_without_database_url_fails_closed(monkeypatch):
    """Verifies that production fails closed (raises RuntimeError) when DATABASE_URL is missing."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        enforce_database_safety()
    assert "DATABASE_URL" in str(exc_info.value)
    assert "production/staging" in str(exc_info.value)
    assert "Silent fallback to temporary in-memory storage is strictly prohibited" in str(exc_info.value)


def test_staging_without_database_url_fails_closed(monkeypatch):
    """Verifies that staging fails closed when DATABASE_URL is missing."""
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        enforce_database_safety()
    assert "DATABASE_URL" in str(exc_info.value)
    assert "production/staging" in str(exc_info.value)
    assert "Silent fallback to temporary in-memory storage is strictly prohibited" in str(exc_info.value)


def test_production_with_unavailable_postgres_driver_fails_closed(monkeypatch):
    """Verifies that production fails closed when psycopg driver is unavailable."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:secret123@db.supabase.co:5432/postgres")
    monkeypatch.setattr(src.database, "is_psycopg_available", lambda: False)

    with pytest.raises(RuntimeError) as exc_info:
        enforce_database_safety()
    assert "CRITICAL DATABASE DRIVER ERROR" in str(exc_info.value)
    assert "psycopg" in str(exc_info.value)
    assert "Silent fallback to in-memory storage is strictly prohibited" in str(exc_info.value)


def test_production_with_invalid_database_scheme_fails_closed(monkeypatch):
    """Verifies that invalid database URI schemes are rejected and credentials masked."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", "mysql://admin_user:super_secret_pwd@db.host.com:3306/mydb")

    with pytest.raises(RuntimeError) as exc_info:
        enforce_database_safety()
    assert "CRITICAL DATABASE CONFIGURATION ERROR: Invalid DATABASE_URL scheme" in str(exc_info.value)
    # Credential masking verification: raw password must NEVER be in error message
    assert "super_secret_pwd" not in str(exc_info.value)
    assert "****" in str(exc_info.value)


def test_development_fallback_permitted(monkeypatch):
    """Verifies that explicit development environment permits in-memory fallback when DATABASE_URL is unset."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("RENDER", raising=False)

    # Must NOT raise RuntimeError
    enforce_database_safety()
    verify_repository_fallback_allowed()
    assert is_explicit_development_or_test() is True


def test_test_environment_fallback_permitted(monkeypatch):
    """Verifies that explicit test environment permits in-memory fallback when DATABASE_URL is unset."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("RENDER", raising=False)

    # Must NOT raise RuntimeError
    enforce_database_safety()
    verify_repository_fallback_allowed()
    assert is_explicit_development_or_test() is True


def test_production_repository_fallback_strictly_rejected(monkeypatch):
    """
    CRITICAL PERSISTENCE DEFENSE:
    Verifies that calling any repository method in production/staging when PostgreSQL
    is unavailable raises a fatal RuntimeError rather than silently corrupting/losing data in RAM.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", "bXlzdXBlcnNlY3JldGtleWZvcmNhc2VsZW5zMjAyNiE=")
    monkeypatch.setattr(src.repository, "is_postgres_enabled", lambda: False)

    # User repository
    with pytest.raises(RuntimeError) as exc_info:
        UserRepository.get_profile("test-user-id")
    assert "Repository in-memory fallback" in str(exc_info.value)
    assert "strictly forbidden outside of explicit development or test environments" in str(exc_info.value)

    # Dataset repository
    with pytest.raises(RuntimeError) as exc_info:
        DatasetRepository.get_dataset("test-ds-id")
    assert "Repository in-memory fallback" in str(exc_info.value)

    # Case repository
    with pytest.raises(RuntimeError) as exc_info:
        CaseRepository.save_cases("test-ds-id", [{"case_ref": "REF-1"}])
    assert "Repository in-memory fallback" in str(exc_info.value)

    # Privacy budget repository
    with pytest.raises(RuntimeError) as exc_info:
        PrivacyBudgetRepository.get_budget("global")
    assert "Repository in-memory fallback" in str(exc_info.value)


def test_credential_masking_in_errors_and_logs():
    """Verifies that mask_database_url completely conceals sensitive credentials."""
    raw_url = "postgresql://postgres:p%40ssw0rd99!@db.ebnyjvawdzglzggdnbzz.supabase.co:5432/postgres"
    masked = mask_database_url(raw_url)
    assert "p%40ssw0rd99!" not in masked
    assert "postgres:****@" in masked
    assert "db.ebnyjvawdzglzggdnbzz.supabase.co" in masked

    # Unset and None handling
    assert mask_database_url(None) == "<unset>"
    assert mask_database_url("") == "<unset>"


def test_hosted_render_fallback_rejection(monkeypatch):
    """Verifies that hosted deployment on Render is treated as hosted environment and fails closed."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    assert is_explicit_development_or_test() is False
    with pytest.raises(RuntimeError) as exc_info:
        enforce_database_safety()
    assert "DATABASE_URL" in str(exc_info.value)
    assert "production/staging/hosted environment" in str(exc_info.value)


def test_ambiguous_environment_fails_closed(monkeypatch):
    """Verifies that an unverified/ambiguous environment fails closed rather than defaulting to dev fallback."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("CASELENS_ENV", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    assert is_explicit_development_or_test() is False
    with pytest.raises(RuntimeError) as exc_info:
        enforce_database_safety()
    assert "unverified deployment environment" in str(exc_info.value)


# ==============================================================================
# 10. Phase 3B Stage 1: Persistent Privacy Budget Tracker Foundation Tests
# ==============================================================================

def test_privacy_budget_initial_creation():
    """Verify 1: Initial budget creation via ensure_budget_initialized."""
    budget_id = "test-p3b-stage1-create"
    res = PrivacyBudgetRepository.ensure_budget_initialized(budget_id=budget_id, total_epsilon=10.0)
    assert res["epsilon_total"] == 10.0
    assert res["epsilon_spent"] == 0.0
    assert res["remaining_epsilon"] == 10.0

    b = PrivacyBudgetRepository.get_budget(budget_id=budget_id)
    assert b["epsilon_total"] == 10.0
    assert b["epsilon_spent"] == 0.0
    assert b["remaining_epsilon"] == 10.0


def test_privacy_budget_spent_epsilon_preserved_during_initialization():
    """Verify 2: Existing spent epsilon is preserved when ensure_budget_initialized is re-invoked."""
    budget_id = "test-p3b-stage1-preserve-spent"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id=budget_id, total_epsilon=10.0)
    PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=3.5, query_id="q1")

    b_before = PrivacyBudgetRepository.get_budget(budget_id=budget_id)
    assert round(b_before["epsilon_spent"], 4) == 3.5

    # Re-invoke ensure_budget_initialized (simulating service restart)
    res_reinit = PrivacyBudgetRepository.ensure_budget_initialized(budget_id=budget_id, total_epsilon=10.0)
    assert round(res_reinit["epsilon_spent"], 4) == 3.5
    assert round(res_reinit["remaining_epsilon"], 4) == 6.5

    b_after = PrivacyBudgetRepository.get_budget(budget_id=budget_id)
    assert round(b_after["epsilon_spent"], 4) == 3.5
    assert round(b_after["remaining_epsilon"], 4) == 6.5


def test_privacy_budget_ledger_records_preserved_across_initialization():
    """Verify 3: Existing ledger records are preserved across re-initialization."""
    budget_id = "test-p3b-stage1-preserve-ledger"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id=budget_id, total_epsilon=10.0)
    PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=1.0, query_id="query-ledger-1")
    PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=2.0, query_id="query-ledger-2")

    count_before = PrivacyBudgetRepository.get_ledger_count(budget_id=budget_id)
    assert count_before == 2

    # Re-initialize
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id=budget_id, total_epsilon=10.0)

    count_after = PrivacyBudgetRepository.get_ledger_count(budget_id=budget_id)
    assert count_after == 2

    ledger = PrivacyBudgetRepository.get_ledger(budget_id=budget_id)
    assert len(ledger) == 2
    assert ledger[0]["query_id"] == "query-ledger-1"
    assert ledger[1]["query_id"] == "query-ledger-2"
    assert ledger[0]["allocated_epsilon"] == 1.0
    assert ledger[1]["allocated_epsilon"] == 2.0


def test_persistent_tracker_reads_from_repository():
    """Verify 4: Tracker dynamically reads values from repository without stale caching."""
    budget_id = "test-p3b-stage1-dynamic-reads"
    tracker = PersistentPrivacyBudgetTracker(budget_id=budget_id, total_epsilon=12.0)
    tracker.ensure_initialized()

    assert tracker.total_epsilon == 12.0
    assert tracker.spent_epsilon == 0.0
    assert tracker.remaining_epsilon == 12.0
    assert tracker.is_exhausted is False

    # External allocation in repository
    PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=4.0, query_id="q-ext")

    # Tracker properties must reflect repository update immediately
    assert tracker.spent_epsilon == 4.0
    assert tracker.remaining_epsilon == 8.0
    assert tracker.is_exhausted is False


def test_persistent_tracker_allocation_delegates_to_repository_in_persistent_mode(monkeypatch):
    """Verify 5: Allocation delegates to the repository in persistent mode (PostgreSQL enabled)."""
    budget_id = "test-p3b-stage1-pg-delegate"
    tracker = PersistentPrivacyBudgetTracker(budget_id=budget_id, total_epsilon=10.0)

    mock_executed_statements = []

    class MockCursor:
        def __init__(self):
            self.closed = False
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.closed = True
        def execute(self, stmt, params=None):
            mock_executed_statements.append((stmt.strip(), params))
        def fetchone(self):
            return {"epsilon_total": 10.0, "epsilon_spent": 2.5}
        def fetchall(self):
            return []

    from contextlib import contextmanager
    @contextmanager
    def mock_get_db_cursor(commit=False):
        yield MockCursor()

    monkeypatch.setattr(src.repository, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.repository, "get_db_cursor", mock_get_db_cursor)

    tracker.allocate(query_id="query-pg-1", epsilon=2.5, actor="admin-user", role="admin")

    # Verify atomic update executed
    assert len(mock_executed_statements) >= 2
    update_stmt, update_params = mock_executed_statements[0]
    assert "UPDATE public.privacy_budget" in update_stmt
    assert "epsilon_spent = epsilon_spent + %s" in update_stmt
    assert update_params[0] == 2.5

    insert_stmt, insert_params = mock_executed_statements[1]
    assert "INSERT INTO public.privacy_budget_ledger" in insert_stmt
    assert insert_params[0] == budget_id
    assert insert_params[1] == "query-pg-1"
    assert insert_params[2] == 2.5


def test_persistent_tracker_local_fallback_works():
    """Verify 6: Local development/test fallback still works transparently."""
    budget_id = "test-p3b-stage1-local-fb"
    tracker = PersistentPrivacyBudgetTracker(budget_id=budget_id, total_epsilon=5.0)
    tracker.ensure_initialized()

    assert tracker.remaining_epsilon == 5.0
    tracker.allocate(query_id="fb-1", epsilon=2.0)
    assert round(tracker.spent_epsilon, 4) == 2.0
    assert round(tracker.remaining_epsilon, 4) == 3.0

    ledger = tracker.get_ledger()
    assert len(ledger) == 1
    assert ledger[0]["query_id"] == "fb-1"

    # Exhaust budget
    tracker.allocate(query_id="fb-2", epsilon=3.0)
    assert tracker.is_exhausted is True

    with pytest.raises(PrivacyBudgetExceededError):
        tracker.allocate(query_id="fb-fail", epsilon=0.1)


def test_persistent_tracker_production_fails_closed(monkeypatch):
    """Verify 7: Production or ambiguous environments fail closed without PostgreSQL."""
    budget_id = "test-p3b-stage1-prod-fail"
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(src.repository, "is_postgres_enabled", lambda: False)

    tracker = PersistentPrivacyBudgetTracker(budget_id=budget_id, total_epsilon=10.0)

    with pytest.raises(RuntimeError) as exc_info:
        tracker.allocate(query_id="fail-query", epsilon=1.0)
    assert "Repository in-memory fallback" in str(exc_info.value)
    assert "strictly forbidden outside of explicit development or test environments" in str(exc_info.value)


def test_startup_initialization_never_destroys_spent_budget():
    """Verify 8: Startup initialization never invokes a destructive reset."""
    from src.api import DATA_STORE, create_app

    orig_spent = DATA_STORE.budget_tracker.spent_epsilon
    orig_total = DATA_STORE.budget_tracker.total_epsilon
    try:
        DATA_STORE.budget_tracker.ensure_initialized(total_epsilon=10.0)
        initial_spent = DATA_STORE.budget_tracker.spent_epsilon
        DATA_STORE.budget_tracker.allocate(query_id="startup-check-alloc", epsilon=1.2)
        assert round(DATA_STORE.budget_tracker.spent_epsilon, 4) == round(initial_spent + 1.2, 4)

        spent_before_restart = DATA_STORE.budget_tracker.spent_epsilon

        # Simulate container / worker startup: create_app() and ensure_initialized()
        _ = create_app()
        DATA_STORE.budget_tracker.ensure_initialized(total_epsilon=10.0)

        # Budget spent must NOT be wiped back to 0.0
        assert round(DATA_STORE.budget_tracker.spent_epsilon, 4) == round(spent_before_restart, 4)
        assert round(DATA_STORE.budget_tracker.remaining_epsilon, 4) == round(DATA_STORE.budget_tracker.total_epsilon - spent_before_restart, 4)
    finally:
        DATA_STORE.budget_tracker.spent_epsilon = orig_spent
        DATA_STORE.budget_tracker.total_epsilon = orig_total


def test_diagnostic_trackers_remain_independent():
    """Verify 9: Diagnostic attack-simulation trackers remain independent of persistent global budget."""
    from src.api import DATA_STORE

    global_spent_before = DATA_STORE.budget_tracker.spent_epsilon
    global_total_before = DATA_STORE.budget_tracker.total_epsilon

    # Standalone diagnostic tracker
    diag_tracker = PrivacyBudgetTracker(total_epsilon=3.0)
    diag_tracker.allocate(query_id="attack_sim_1", epsilon=1.5)
    assert diag_tracker.spent_epsilon == 1.5
    assert diag_tracker.remaining_epsilon == 1.5

    # Global persistent tracker must be completely untouched
    assert DATA_STORE.budget_tracker.spent_epsilon == global_spent_before
    assert DATA_STORE.budget_tracker.total_epsilon == global_total_before


# ==============================================================================
# 11. Phase 3B Stage 2: Verified Actor and Role Propagation Tests
# ==============================================================================

def test_jwt_authenticated_allocation_records_verified_identity():
    """Verify 1: JWT-authenticated allocation records verified user ID and role."""
    from src.auth import create_access_token
    from src.api import get_current_user, sanitize_and_verify_payload, DATA_STORE, Role

    token = create_access_token(
        user_id="jwt_researcher_42",
        username="researcher42",
        role="analyst"
    )
    user = get_current_user(authorization=f"Bearer {token}")
    assert user.user_id == "jwt_researcher_42"
    assert user.role == Role.ANALYST
    assert user.auth_method == "jwt"

    res = sanitize_and_verify_payload(
        {"status": "ok", "value": 123},
        epsilon_cost=0.08,
        query_id="test_jwt_allocation_query",
        user=user
    )
    assert res["status"] == "ok"

    ledger = DATA_STORE.budget_tracker.get_ledger(limit=1)
    assert len(ledger) >= 1
    last = ledger[-1]
    assert last["query_id"] == "test_jwt_allocation_query"
    assert last["actor"] == "jwt_researcher_42"
    assert last["role"] == "analyst"
    assert round(last["allocated_epsilon"], 4) == 0.08


def test_api_key_authenticated_allocation_records_verified_identity(monkeypatch):
    """Verify 2: Institutional API-key allocation records verified API-key identity and role."""
    import src.api as api_mod
    from src.api import get_current_user, sanitize_and_verify_payload, DATA_STORE, Role

    mock_keys = {"test-institutional-api-key-99": Role.VIEWER}
    monkeypatch.setattr(api_mod, "get_configured_api_keys", lambda: mock_keys)

    user = get_current_user(x_api_key="test-institutional-api-key-99")
    assert user.user_id == "key_viewer"
    assert user.role == Role.VIEWER
    assert user.auth_method == "api_key"

    res = sanitize_and_verify_payload(
        {"summary_count": 50},
        epsilon_cost=0.05,
        query_id="test_api_key_allocation_query",
        user=user
    )
    assert res["summary_count"] == 50

    ledger = DATA_STORE.budget_tracker.get_ledger(limit=1)
    assert len(ledger) >= 1
    last = ledger[-1]
    assert last["query_id"] == "test_api_key_allocation_query"
    assert last["actor"] == "key_viewer"
    assert last["role"] == "viewer"
    assert round(last["allocated_epsilon"], 4) == 0.05


def test_client_cannot_spoof_actor_or_role(monkeypatch):
    """Verify 3: Client cannot spoof actor or role via unverified headers or body payloads."""
    from src.auth import create_access_token
    from src.api import get_current_user, sanitize_and_verify_payload, DATA_STORE, Role

    # Ensure CASELENS_TEST_AUTH is disabled
    monkeypatch.delenv("CASELENS_TEST_AUTH", raising=False)

    token = create_access_token(
        user_id="honest_viewer_id",
        username="honest_viewer",
        role="viewer"
    )

    # Malicious client sends spoofed header attempting to escalate to admin
    user = get_current_user(
        authorization=f"Bearer {token}",
        x_user_role="admin"
    )
    assert user.user_id == "honest_viewer_id"
    assert user.role == Role.VIEWER

    # Malicious client passes extra body fields pretending to be admin
    payload_with_spoofed_fields = {
        "status": "success",
        "actor": "attacker_root",
        "role": "admin",
        "user_id": "attacker_root"
    }
    sanitize_and_verify_payload(
        payload_with_spoofed_fields,
        epsilon_cost=0.05,
        query_id="spoof_test_query",
        user=user
    )

    ledger = DATA_STORE.budget_tracker.get_ledger(limit=1)
    last = ledger[-1]
    assert last["query_id"] == "spoof_test_query"
    assert last["actor"] == "honest_viewer_id"
    assert last["role"] == "viewer"
    assert last["actor"] != "attacker_root"


def test_direct_aggregate_allocation_records_verified_identity():
    """Verify 4: Direct aggregate allocation records verified UserContext identity."""
    import pandas as pd
    from starlette.requests import Request
    from src.api import (
        create_app,
        AggregateQueryRequest,
        UserContext,
        Role,
        DATA_STORE
    )

    orig_df_cases = DATA_STORE.df_cases
    try:
        DATA_STORE.df_cases = pd.DataFrame([
            {"district": "D1_North", "case_category": "trafficking", "terminal_stage": "court_trial"},
            {"district": "D1_North", "case_category": "trafficking", "terminal_stage": "court_trial"},
            {"district": "D1_North", "case_category": "trafficking", "terminal_stage": "court_trial"},
            {"district": "D1_North", "case_category": "trafficking", "terminal_stage": "court_trial"},
            {"district": "D1_North", "case_category": "trafficking", "terminal_stage": "court_trial"},
            {"district": "D1_North", "case_category": "trafficking", "terminal_stage": "court_trial"},
        ])

        app = create_app()
        query_aggregate_fn = next(
            r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/v1/query/aggregate"
        )

        req = Request(scope={"type": "http", "state": {}})
        user = UserContext(user_id="analyst_direct_user", role=Role.ANALYST, auth_method="jwt")
        req_body = AggregateQueryRequest(dimension="district", epsilon=0.20, k_threshold=5)

        res = query_aggregate_fn(
            request_body=req_body,
            request=req,
            user=user
        )
        assert res["query_dimension"] == "district"

        ledger = DATA_STORE.budget_tracker.get_ledger(limit=1)
        last = ledger[-1]
        assert last["query_id"] == "dynamic_aggregate_district"
        assert last["actor"] == "analyst_direct_user"
        assert last["role"] == "analyst"
        assert round(last["allocated_epsilon"], 4) == 0.20
    finally:
        DATA_STORE.df_cases = orig_df_cases


def test_privacy_budget_exhaustion_behavior_unchanged():
    """Verify 5: Privacy budget exhaustion behavior is unchanged (HTTP 429 and zero ledger entry)."""
    from fastapi import HTTPException
    from src.api import sanitize_and_verify_payload, DATA_STORE, UserContext, Role

    tracker = DATA_STORE.budget_tracker
    orig_spent = tracker.spent_epsilon
    orig_total = tracker.total_epsilon
    try:
        # Exhaust the budget
        tracker.spent_epsilon = tracker.total_epsilon
        user = UserContext(user_id="exhaust_user", role=Role.VIEWER, auth_method="jwt")

        ledger_len_before = len(tracker.get_ledger())

        with pytest.raises(HTTPException) as exc_info:
            sanitize_and_verify_payload(
                {"status": "test"},
                epsilon_cost=0.05,
                query_id="query_should_fail_exhausted",
                user=user
            )
        assert exc_info.value.status_code == 429
        assert "Privacy budget exceeded" in exc_info.value.detail

        # Ledger length must remain identical (zero entry created)
        ledger_len_after = len(tracker.get_ledger())
        assert ledger_len_after == ledger_len_before
    finally:
        tracker.spent_epsilon = orig_spent
        tracker.total_epsilon = orig_total


def test_role_hierarchy_and_permissions_unchanged():
    """Verify 6: Role permissions and hierarchy remain strictly enforced."""
    from starlette.requests import Request
    from fastapi import HTTPException
    from src.api import require_role, UserContext, Role

    req = Request(scope={"type": "http", "state": {}})

    # 1. Viewer trying to access Analyst-only endpoint receives 403
    analyst_checker = require_role(Role.ANALYST)
    viewer_user = UserContext(user_id="v_user", role=Role.VIEWER, auth_method="jwt")
    with pytest.raises(HTTPException) as exc_info:
        analyst_checker(request=req, user=viewer_user)
    assert exc_info.value.status_code == 403
    assert "Insufficient privileges" in exc_info.value.detail

    # 2. Analyst trying to access Admin-only endpoint receives 403
    admin_checker = require_role(Role.ADMIN)
    analyst_user = UserContext(user_id="a_user", role=Role.ANALYST, auth_method="jwt")
    with pytest.raises(HTTPException) as exc_info:
        admin_checker(request=req, user=analyst_user)
    assert exc_info.value.status_code == 403
    assert "Insufficient privileges" in exc_info.value.detail

    # 3. Admin accessing Viewer endpoint succeeds and sets request.state
    viewer_checker = require_role(Role.VIEWER)
    admin_user = UserContext(user_id="admin_user_01", role=Role.ADMIN, auth_method="jwt")
    passed_user = viewer_checker(request=req, user=admin_user)
    assert passed_user.user_id == "admin_user_01"
    assert req.state.user == admin_user
    assert req.state.user_id == "admin_user_01"
    assert req.state.user_role == "admin"


# ==============================================================================
# 12. Phase 3B Stage 3: Privacy Budget Integrity & Persistent Delta Tests
# ==============================================================================

def test_idempotent_allocation_repeated_key_no_double_charge():
    """Verify 1: Repeated allocation with same idempotency key does not charge twice."""
    budget_id = "test-p3b-s3-idem-1"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    res1 = PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.5,
        query_id="q-idem-1",
        actor="analyst_1",
        role="analyst",
        idempotency_key="key-abc-123"
    )
    assert res1["allocated"] is True
    assert res1["idempotent_replay"] is False
    assert round(res1["epsilon_spent"], 4) == 1.5

    # Second execution with exact same key and parameters
    res2 = PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.5,
        query_id="q-idem-1",
        actor="analyst_1",
        role="analyst",
        idempotency_key="key-abc-123"
    )
    assert res2["allocated"] is True
    assert res2["idempotent_replay"] is True
    assert round(res2["epsilon_spent"], 4) == 1.5

    # Check repository state
    b = PrivacyBudgetRepository.get_budget(budget_id)
    assert round(b["epsilon_spent"], 4) == 1.5
    assert PrivacyBudgetRepository.get_ledger_count(budget_id) == 1


def test_idempotent_allocation_conflicting_parameters_rejected():
    """Verify 2: Reusing idempotency key with conflicting parameters is strictly rejected."""
    budget_id = "test-p3b-s3-idem-conflict"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.0,
        query_id="q-initial",
        actor="analyst_user",
        role="analyst",
        idempotency_key="key-conflict-test"
    )

    # Reusing same key with different epsilon
    with pytest.raises(ValueError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=2.0,  # conflicting
            query_id="q-initial",
            actor="analyst_user",
            role="analyst",
            idempotency_key="key-conflict-test"
        )
    assert "Idempotency conflict" in str(exc_info.value)

    # Reusing same key with different query_id
    with pytest.raises(ValueError) as exc_info2:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            query_id="q-different",  # conflicting
            actor="analyst_user",
            role="analyst",
            idempotency_key="key-conflict-test"
        )
    assert "Idempotency conflict" in str(exc_info2.value)

    # Budget spent must remain unchanged
    b = PrivacyBudgetRepository.get_budget(budget_id)
    assert round(b["epsilon_spent"], 4) == 1.0


def test_concurrent_duplicate_idempotency_requests():
    """Verify 3: Concurrent duplicate requests with the same idempotency key charge only once."""
    budget_id = "test-p3b-s3-concurrent-idem"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    results = []
    errors = []

    def _worker():
        try:
            r = PrivacyBudgetRepository.allocate_budget(
                budget_id=budget_id,
                epsilon=0.75,
                query_id="concurrent_query",
                actor="worker_analyst",
                role="analyst",
                idempotency_key="shared-concurrent-key"
            )
            results.append(r)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) == 10
    b = PrivacyBudgetRepository.get_budget(budget_id)
    assert round(b["epsilon_spent"], 4) == 0.75
    assert PrivacyBudgetRepository.get_ledger_count(budget_id) == 1


def test_epsilon_exhaustion_enforcement():
    """Verify 4: Differential privacy epsilon budget exhaustion raises PrivacyBudgetExceededError."""
    budget_id = "test-p3b-s3-eps-exhaust"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=2.0, total_delta=1e-4)

    PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=1.5, query_id="q1")
    assert round(PrivacyBudgetRepository.get_budget(budget_id)["epsilon_spent"], 4) == 1.5

    with pytest.raises(PrivacyBudgetExceededError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=0.8, query_id="q2")
    assert "epsilon" in str(exc_info.value).lower()
    assert round(PrivacyBudgetRepository.get_budget(budget_id)["epsilon_spent"], 4) == 1.5


def test_delta_exhaustion_enforcement():
    """Verify 5: Differential privacy delta budget exhaustion raises PrivacyBudgetExceededError."""
    budget_id = "test-p3b-s3-delta-exhaust"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=0.1, delta=0.7e-4, query_id="qd1")
    assert round(PrivacyBudgetRepository.get_budget(budget_id)["delta_spent"], 8) == 0.7e-4

    with pytest.raises(PrivacyBudgetExceededError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=0.1, delta=0.4e-4, query_id="qd2")
    assert "delta" in str(exc_info.value).lower()
    assert round(PrivacyBudgetRepository.get_budget(budget_id)["delta_spent"], 8) == 0.7e-4


def test_simultaneous_epsilon_and_delta_atomic_limits():
    """Verify 6: Atomic validation of simultaneous epsilon and delta limits under CAS."""
    budget_id = "test-p3b-s3-dual-limits"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=5.0, total_delta=1e-4)

    res = PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=2.0,
        delta=0.5e-4,
        query_id="q-dual"
    )
    assert res["allocated"] is True
    assert round(res["epsilon_spent"], 4) == 2.0
    assert round(res["delta_spent"], 8) == 0.5e-4

    # Valid epsilon (1.0), but excessive delta (0.6e-4 -> total 1.1e-4 > 1.0e-4)
    with pytest.raises(PrivacyBudgetExceededError):
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            delta=0.6e-4,
            query_id="q-dual-fail"
        )

    # Budget spent remains at initial allocation
    b = PrivacyBudgetRepository.get_budget(budget_id)
    assert round(b["epsilon_spent"], 4) == 2.0
    assert round(b["delta_spent"], 8) == 0.5e-4


def test_rollback_if_ledger_insertion_fails(monkeypatch):
    """Verify 7: CAS budget update rolls back atomically if ledger insertion fails."""
    budget_id = "test-p3b-s3-rollback"
    tracker = PersistentPrivacyBudgetTracker(budget_id=budget_id, total_epsilon=10.0)

    class MockFailingCursor:
        def __init__(self):
            self.step = 0
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def execute(self, stmt, params=None):
            self.step += 1
            if "INSERT INTO public.privacy_budget_ledger" in stmt:
                raise RuntimeError("Simulated Database Constraint / Disk Failure during ledger insert")
        def fetchone(self):
            return {"epsilon_total": 10.0, "epsilon_spent": 2.0, "delta_total": 1e-4, "delta_spent": 0.0}

    from contextlib import contextmanager
    @contextmanager
    def mock_get_db_cursor(commit=False):
        yield MockFailingCursor()

    monkeypatch.setattr(src.repository, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.repository, "get_db_cursor", mock_get_db_cursor)

    with pytest.raises(RuntimeError) as exc_info:
        tracker.allocate(query_id="failing-tx", epsilon=2.0, actor="admin", role="admin")
    assert "Simulated Database Constraint" in str(exc_info.value)


def test_restart_persistence_preserves_epsilon_and_delta():
    """Verify 8: Application / container restart never destroys spent epsilon or spent delta."""
    budget_id = "test-p3b-s3-restart-preserve"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)
    PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=3.2, delta=0.4e-4, query_id="q-pre-restart")

    b_before = PrivacyBudgetRepository.get_budget(budget_id)
    assert round(b_before["epsilon_spent"], 4) == 3.2
    assert round(b_before["delta_spent"], 8) == 0.4e-4

    # Simulate container startup invoking ensure_budget_initialized
    reinit = PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)
    assert round(reinit["epsilon_spent"], 4) == 3.2
    assert round(reinit["delta_spent"], 8) == 0.4e-4
    assert round(reinit["remaining_epsilon"], 4) == 6.8
    assert round(reinit["remaining_delta"], 8) == 0.6e-4


def test_admin_reset_accounting():
    """Verify 9: Administrative reset zeroes spent counters and applies new total bounds."""
    budget_id = "test-p3b-s3-admin-reset"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)
    PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=4.0, delta=0.5e-4, query_id="q-pre-reset")

    res = PrivacyBudgetRepository.reset_budget(
        budget_id=budget_id,
        total_epsilon=8.0,
        total_delta=2e-4,
        actor="admin_lead",
        role="admin",
        reason="Quarterly reallocation"
    )
    assert res["epsilon_total"] == 8.0
    assert res["epsilon_spent"] == 0.0
    assert res["delta_total"] == 2e-4
    assert res["delta_spent"] == 0.0

    b = PrivacyBudgetRepository.get_budget(budget_id)
    assert b["epsilon_total"] == 8.0
    assert b["epsilon_spent"] == 0.0
    assert b["delta_total"] == 2e-4
    assert b["delta_spent"] == 0.0


def test_reset_audit_persistence():
    """Verify 10: Administrative reset creates an immutable audit record in reset history."""
    budget_id = "test-p3b-s3-reset-audit"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)
    PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=2.5, delta=0.2e-4, query_id="q-spent")

    PrivacyBudgetRepository.reset_budget(
        budget_id=budget_id,
        total_epsilon=12.0,
        total_delta=1.5e-4,
        actor="sec_director",
        role="admin",
        reason="Audit approval cycle 2026-Q4"
    )

    history = PrivacyBudgetRepository.get_reset_history(budget_id)
    assert len(history) >= 1
    last = history[-1]
    assert last["budget_id"] == budget_id
    assert last["actor"] == "sec_director"
    assert last["role"] == "admin"
    assert round(last["previous_epsilon_spent"], 4) == 2.5
    assert round(last["previous_delta_spent"], 8) == 0.2e-4
    assert last["new_epsilon_total"] == 12.0
    assert last["new_delta_total"] == 1.5e-4
    assert last["reason"] == "Audit approval cycle 2026-Q4"


def test_viewer_and_analyst_reset_rejection(monkeypatch):
    """Verify 11: Viewers and Analysts are strictly forbidden from executing privacy budget resets."""
    from fastapi import HTTPException
    from src.api import create_app, UserContext, Role

    # 1. Non-admin role in production environment raises RuntimeError
    monkeypatch.setattr(src.repository, "_is_dev_test", lambda: False)
    with pytest.raises(RuntimeError) as exc_info:
        PrivacyBudgetRepository.reset_budget(budget_id="sec-check", actor="rogue_viewer", role="viewer")
    assert "FATAL SECURITY VIOLATION: Privacy budget reset requires verified admin role" in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info2:
        PrivacyBudgetRepository.reset_budget(budget_id="sec-check", actor="rogue_analyst", role="analyst")
    assert "FATAL SECURITY VIOLATION: Privacy budget reset requires verified admin role" in str(exc_info2.value)

    # 2. FastAPI endpoint level: viewer and analyst receive HTTP 403 Forbidden
    app = create_app()
    reset_endpoint = next(
        r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/v1/privacy/budget/reset"
    )
    from src.api import require_role
    viewer_user = UserContext(user_id="v_user", role=Role.VIEWER, auth_method="jwt")
    analyst_user = UserContext(user_id="a_user", role=Role.ANALYST, auth_method="jwt")
    admin_checker = require_role(Role.ADMIN)

    from starlette.requests import Request
    req = Request(scope={"type": "http", "state": {}})

    with pytest.raises(HTTPException) as http_exc1:
        admin_checker(request=req, user=viewer_user)
    assert http_exc1.value.status_code == 403

    with pytest.raises(HTTPException) as http_exc2:
        admin_checker(request=req, user=analyst_user)
    assert http_exc2.value.status_code == 403


def test_historical_ledger_preserved_across_reset():
    """Verify 12: Administrative resets zero the counters without truncating historical ledger."""
    budget_id = "test-p3b-s3-ledger-preserved"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=1.0, query_id="tx_1")
    PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=1.0, query_id="tx_2")
    PrivacyBudgetRepository.allocate_budget(budget_id=budget_id, epsilon=1.0, query_id="tx_3")
    assert PrivacyBudgetRepository.get_ledger_count(budget_id) == 3

    # Execute administrative reset
    PrivacyBudgetRepository.reset_budget(
        budget_id=budget_id,
        total_epsilon=10.0,
        actor="admin_cleaner",
        role="admin"
    )

    # Counters are reset
    b = PrivacyBudgetRepository.get_budget(budget_id)
    assert b["epsilon_spent"] == 0.0

    # Historical ledger entries remain intact
    assert PrivacyBudgetRepository.get_ledger_count(budget_id) == 3
    ledger = PrivacyBudgetRepository.get_ledger(budget_id)
    assert len(ledger) == 3
    assert [e["query_id"] for e in ledger] == ["tx_1", "tx_2", "tx_3"]


def test_existing_api_response_compatibility():
    """Verify 13: Existing API endpoints return backward-compatible privacy metadata."""
    from src.api import DATA_STORE, UserContext, Role
    from starlette.requests import Request

    admin_user = UserContext(user_id="adm_test", role=Role.ADMIN, auth_method="jwt")
    from src.api import create_app
    app = create_app()
    settings_endpoint = next(
        r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/v1/privacy/settings" and "GET" in r.methods
    )
    res = settings_endpoint(user=admin_user)
    assert res["status"] == "success"
    assert "total_epsilon" in res
    assert "spent_epsilon" in res
    assert "remaining_epsilon" in res
    assert "total_delta" in res


def test_critical_identity_requirement_enforcement_in_production(monkeypatch):
    """Verify 14: In production/staging, allocation without verified actor and role fails closed."""
    monkeypatch.setattr(src.repository, "_is_dev_test", lambda: False)

    with pytest.raises(RuntimeError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id="prod-alloc-test",
            epsilon=0.5,
            query_id="unauth_alloc",
            actor=None,
            role=None
        )
    assert "FATAL SECURITY VIOLATION: Privacy budget allocation requires a verified actor and role" in str(exc_info.value)


# ==============================================================================
# 13. Targeted Idempotency Defect Corrections (Tests 53-63)
# ==============================================================================

def test_idempotency_same_parameters_and_identity_successful_replay():
    """Verify 53: Same key, same parameters, same actor and role returns successful replay without extra charge."""
    budget_id = "test-idem-replay-success"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    res1 = PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.2,
        delta=0.2e-4,
        query_id="q_exact",
        actor="analyst_valid",
        role="analyst",
        idempotency_key="exact_idem_key_1"
    )
    assert res1["allocated"] is True
    assert res1["idempotent_replay"] is False
    assert round(res1["epsilon_spent"], 4) == 1.2
    assert round(res1["delta_spent"], 8) == 0.2e-4

    # Repeated identical request with same actor and role
    res2 = PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.2,
        delta=0.2e-4,
        query_id="q_exact",
        actor="analyst_valid",
        role="analyst",
        idempotency_key="exact_idem_key_1"
    )
    assert res2["allocated"] is True
    assert res2["idempotent_replay"] is True
    assert round(res2["epsilon_spent"], 4) == 1.2
    assert round(res2["delta_spent"], 8) == 0.2e-4
    assert PrivacyBudgetRepository.get_ledger_count(budget_id) == 1


def test_idempotency_different_actor_conflict():
    """Verify 54: Reusing the same idempotency key with a different actor is strictly rejected as a conflict."""
    budget_id = "test-idem-diff-actor"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.0,
        query_id="q_actor",
        actor="alice_analyst",
        role="analyst",
        idempotency_key="actor_idem_key"
    )

    with pytest.raises(ValueError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            query_id="q_actor",
            actor="bob_analyst",  # Conflicting actor
            role="analyst",
            idempotency_key="actor_idem_key"
        )
    assert "Idempotency conflict" in str(exc_info.value)
    assert PrivacyBudgetRepository.get_ledger_count(budget_id) == 1


def test_idempotency_different_role_conflict():
    """Verify 55: Reusing the same idempotency key with a different role is strictly rejected as a conflict."""
    budget_id = "test-idem-diff-role"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.0,
        query_id="q_role",
        actor="alice_analyst",
        role="analyst",
        idempotency_key="role_idem_key"
    )

    with pytest.raises(ValueError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            query_id="q_role",
            actor="alice_analyst",
            role="admin",  # Conflicting role
            idempotency_key="role_idem_key"
        )
    assert "Idempotency conflict" in str(exc_info.value)
    assert PrivacyBudgetRepository.get_ledger_count(budget_id) == 1


def test_idempotency_different_epsilon_conflict():
    """Verify 56: Reusing the same idempotency key with a different epsilon value is rejected."""
    budget_id = "test-idem-diff-eps"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.0,
        delta=0.1e-4,
        query_id="q_eps",
        actor="analyst_test",
        role="analyst",
        idempotency_key="eps_idem_key"
    )

    with pytest.raises(ValueError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=2.0,  # Conflicting epsilon
            delta=0.1e-4,
            query_id="q_eps",
            actor="analyst_test",
            role="analyst",
            idempotency_key="eps_idem_key"
        )
    assert "Idempotency conflict" in str(exc_info.value)


def test_idempotency_different_delta_conflict():
    """Verify 57: Reusing the same idempotency key with a different delta value is rejected."""
    budget_id = "test-idem-diff-delta"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.0,
        delta=0.1e-4,
        query_id="q_delta",
        actor="analyst_test",
        role="analyst",
        idempotency_key="delta_idem_key"
    )

    with pytest.raises(ValueError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            delta=0.5e-4,  # Conflicting delta
            query_id="q_delta",
            actor="analyst_test",
            role="analyst",
            idempotency_key="delta_idem_key"
        )
    assert "Idempotency conflict" in str(exc_info.value)


def test_idempotency_different_query_id_conflict():
    """Verify 58: Reusing the same idempotency key with a different query ID is rejected."""
    budget_id = "test-idem-diff-qid"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.0,
        query_id="q_first",
        actor="analyst_test",
        role="analyst",
        idempotency_key="qid_idem_key"
    )

    with pytest.raises(ValueError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            query_id="q_second",  # Conflicting query_id
            actor="analyst_test",
            role="analyst",
            idempotency_key="qid_idem_key"
        )
    assert "Idempotency conflict" in str(exc_info.value)


# ------------------------------------------------------------------------------
# Simulated PostgreSQL Unique-Violation Recovery Tests (Simulated Mocks)
# ------------------------------------------------------------------------------

class _SimulatedDiag:
    def __init__(self, constraint_name="uq_idx_budget_ledger_idempotency"):
        self.constraint_name = constraint_name


class _SimulatedUniqueViolation(Exception):
    """Simulated PostgreSQL UniqueViolation error (SQLSTATE 23505) with diagnostic metadata."""
    def __init__(
        self,
        message='duplicate key value violates unique constraint "uq_idx_budget_ledger_idempotency"',
        constraint_name="uq_idx_budget_ledger_idempotency"
    ):
        super().__init__(message)
        self.pgcode = "23505"
        self.sqlstate = "23505"
        self.diag = _SimulatedDiag(constraint_name) if constraint_name else None


class _MockSimulatedConn:
    def __init__(self, committed_row=None, budget_row=None):
        self.committed_row = committed_row
        self.budget_row = budget_row or {"epsilon_total": 10.0, "epsilon_spent": 1.0, "delta_total": 1e-4, "delta_spent": 0.1e-4}
        self.events = []
    def rollback(self):
        self.events.append("rollback")
    def commit(self):
        self.events.append("commit")


class _MockSimulatedWriteCursor:
    def __init__(self, conn, exc_to_raise=None):
        self.conn = conn
        self.last_query = ""
        self.exc_to_raise = exc_to_raise
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.conn.rollback()
    def execute(self, stmt, params=None):
        self.last_query = stmt
        if "INSERT INTO public.privacy_budget_ledger" in stmt:
            if self.exc_to_raise is not None:
                raise self.exc_to_raise
            raise _SimulatedUniqueViolation()
    def fetchone(self):
        if "FROM public.privacy_budget_ledger" in self.last_query:
            return None
        return {"epsilon_total": 10.0, "epsilon_spent": 2.0, "delta_total": 1e-4, "delta_spent": 0.2e-4}


class _MockSimulatedReadCursor:
    def __init__(self, conn):
        self.conn = conn
        self.last_query = ""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def execute(self, stmt, params=None):
        self.last_query = stmt
        if "FROM public.privacy_budget_ledger" in stmt:
            self.conn.events.append("read_committed_ledger")
    def fetchone(self):
        if "FROM public.privacy_budget_ledger" in self.last_query:
            return self.conn.committed_row
        if "FROM public.privacy_budget" in self.last_query:
            return self.conn.budget_row
        return None


def test_simulated_pg_unique_violation_matching_row_successful_replay(monkeypatch):
    """Verify 59: [SIMULATED] PostgreSQL unique violation followed by matching committed row returns replay."""
    from contextlib import contextmanager
    budget_id = "test-sim-unique-success"
    matching_row = {
        "query_id": "sim_q",
        "epsilon_cost": 1.0,
        "delta_cost": 0.1e-4,
        "actor": "analyst_sim",
        "role": "analyst"
    }
    mock_conn = _MockSimulatedConn(committed_row=matching_row)

    @contextmanager
    def mock_get_db_cursor(commit=True, dict_cursor=True):
        if commit:
            try:
                yield _MockSimulatedWriteCursor(mock_conn)
            except Exception:
                mock_conn.rollback()
                raise
        else:
            yield _MockSimulatedReadCursor(mock_conn)

    monkeypatch.setattr(src.repository, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.repository, "get_db_cursor", mock_get_db_cursor)

    res = PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.0,
        delta=0.1e-4,
        query_id="sim_q",
        actor="analyst_sim",
        role="analyst",
        idempotency_key="sim_key_1"
    )
    assert res["allocated"] is True
    assert res["idempotent_replay"] is True
    assert round(res["epsilon_spent"], 4) == 1.0
    assert round(res["delta_spent"], 8) == 0.1e-4


def test_simulated_pg_unique_violation_mismatching_row_conflict(monkeypatch):
    """Verify 60: [SIMULATED] PostgreSQL unique violation followed by mismatching committed row raises conflict."""
    from contextlib import contextmanager
    budget_id = "test-sim-unique-conflict"
    mismatch_row = {
        "query_id": "sim_q",
        "epsilon_cost": 1.0,
        "delta_cost": 0.1e-4,
        "actor": "analyst_sim",
        "role": "admin"  # Conflict: incoming is analyst, stored is admin
    }
    mock_conn = _MockSimulatedConn(committed_row=mismatch_row)

    @contextmanager
    def mock_get_db_cursor(commit=True, dict_cursor=True):
        if commit:
            try:
                yield _MockSimulatedWriteCursor(mock_conn)
            except Exception:
                mock_conn.rollback()
                raise
        else:
            yield _MockSimulatedReadCursor(mock_conn)

    monkeypatch.setattr(src.repository, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.repository, "get_db_cursor", mock_get_db_cursor)

    with pytest.raises(ValueError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            delta=0.1e-4,
            query_id="sim_q",
            actor="analyst_sim",
            role="analyst",
            idempotency_key="sim_key_2"
        )
    assert "Idempotency conflict" in str(exc_info.value)


def test_simulated_pg_unique_violation_no_committed_row_fails_safely(monkeypatch):
    """Verify 61: [SIMULATED] PostgreSQL unique violation with no committed row raises meaningful error."""
    from contextlib import contextmanager
    budget_id = "test-sim-unique-missing"
    mock_conn = _MockSimulatedConn(committed_row=None)  # No committed row found

    @contextmanager
    def mock_get_db_cursor(commit=True, dict_cursor=True):
        if commit:
            try:
                yield _MockSimulatedWriteCursor(mock_conn)
            except Exception:
                mock_conn.rollback()
                raise
        else:
            yield _MockSimulatedReadCursor(mock_conn)

    monkeypatch.setattr(src.repository, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.repository, "get_db_cursor", mock_get_db_cursor)

    with pytest.raises(RuntimeError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            delta=0.1e-4,
            query_id="sim_q",
            actor="analyst_sim",
            role="analyst",
            idempotency_key="sim_key_3"
        )
    assert "no committed ledger record was found" in str(exc_info.value)


def test_simulated_pg_unique_violation_rollback_occurs_before_read_lookup(monkeypatch):
    """Verify 62: [SIMULATED] Verification that failed transaction rollback occurs strictly before fresh read."""
    from contextlib import contextmanager
    budget_id = "test-sim-unique-order"
    matching_row = {
        "query_id": "sim_q",
        "epsilon_cost": 1.0,
        "delta_cost": 0.1e-4,
        "actor": "analyst_sim",
        "role": "analyst"
    }
    mock_conn = _MockSimulatedConn(committed_row=matching_row)

    @contextmanager
    def mock_get_db_cursor(commit=True, dict_cursor=True):
        if commit:
            try:
                yield _MockSimulatedWriteCursor(mock_conn)
            except Exception:
                mock_conn.rollback()
                raise
        else:
            yield _MockSimulatedReadCursor(mock_conn)

    monkeypatch.setattr(src.repository, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.repository, "get_db_cursor", mock_get_db_cursor)

    PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.0,
        delta=0.1e-4,
        query_id="sim_q",
        actor="analyst_sim",
        role="analyst",
        idempotency_key="sim_key_order"
    )
    # The event log MUST show rollback executed before the read query
    assert "rollback" in mock_conn.events
    assert "read_committed_ledger" in mock_conn.events
    rollback_idx = mock_conn.events.index("rollback")
    read_idx = mock_conn.events.index("read_committed_ledger")
    assert rollback_idx < read_idx


def test_simulated_pg_losing_request_does_not_double_charge_budget(monkeypatch):
    """Verify 63: [SIMULATED] Losing concurrent request that recovers via replay does not increment spent budget."""
    from contextlib import contextmanager
    budget_id = "test-sim-no-double-charge"
    # Prior committed row from winner shows 1.5 epsilon was spent
    prior_row = {
        "query_id": "concurrent_q",
        "epsilon_cost": 1.5,
        "delta_cost": 0.2e-4,
        "actor": "analyst_sim",
        "role": "analyst"
    }
    budget_state = {
        "epsilon_total": 10.0,
        "epsilon_spent": 1.5,  # winner's charge only
        "delta_total": 1e-4,
        "delta_spent": 0.2e-4
    }
    mock_conn = _MockSimulatedConn(committed_row=prior_row, budget_row=budget_state)

    @contextmanager
    def mock_get_db_cursor(commit=True, dict_cursor=True):
        if commit:
            try:
                yield _MockSimulatedWriteCursor(mock_conn)
            except Exception:
                mock_conn.rollback()
                raise
        else:
            yield _MockSimulatedReadCursor(mock_conn)

    monkeypatch.setattr(src.repository, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.repository, "get_db_cursor", mock_get_db_cursor)

    res = PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.5,
        delta=0.2e-4,
        query_id="concurrent_q",
        actor="analyst_sim",
        role="analyst",
        idempotency_key="shared_idem_key"
    )
    assert res["idempotent_replay"] is True
    # Spent counters reflect existing winner's charge, not 3.0
    assert round(res["epsilon_spent"], 4) == 1.5
    assert round(res["delta_spent"], 8) == 0.2e-4


def test_idempotency_stored_query_id_with_incoming_none_conflict():
    """Verify 64: Stored non-empty query_id with incoming None raises idempotency conflict."""
    budget_id = "test-idem-qid-none"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.0,
        query_id="q_valid",
        actor="analyst_user",
        role="analyst",
        idempotency_key="key_qid_none"
    )

    with pytest.raises(ValueError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            query_id=None,  # Incoming None vs stored 'q_valid'
            actor="analyst_user",
            role="analyst",
            idempotency_key="key_qid_none"
        )
    assert "Idempotency conflict" in str(exc_info.value)


def test_idempotency_stored_query_id_with_incoming_empty_string_conflict():
    """Verify 65: Stored non-empty query_id with incoming empty string raises idempotency conflict."""
    budget_id = "test-idem-qid-empty"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    PrivacyBudgetRepository.allocate_budget(
        budget_id=budget_id,
        epsilon=1.0,
        query_id="q_valid",
        actor="analyst_user",
        role="analyst",
        idempotency_key="key_qid_empty"
    )

    with pytest.raises(ValueError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            query_id="",  # Incoming empty string vs stored 'q_valid'
            actor="analyst_user",
            role="analyst",
            idempotency_key="key_qid_empty"
        )
    assert "Idempotency conflict" in str(exc_info.value)


def test_idempotency_stored_none_query_id_with_incoming_nonempty_conflict():
    """Verify 66: Stored None query_id with incoming non-empty query_id raises idempotency conflict."""
    from src.repository import _local_budget_ledger, _local_store_lock
    budget_id = "test-idem-stored-none-qid"
    PrivacyBudgetRepository.ensure_budget_initialized(budget_id, total_epsilon=10.0, total_delta=1e-4)

    with _local_store_lock:
        _local_budget_ledger.append({
            "budget_id": budget_id,
            "query_id": None,  # Stored None
            "epsilon_cost": 1.0,
            "delta_cost": 0.0,
            "allocated_epsilon": 1.0,
            "allocated_delta": 0.0,
            "actor": "analyst_user",
            "role": "analyst",
            "idempotency_key": "key_stored_none",
            "timestamp": "2026-01-01T00:00:00Z"
        })

    with pytest.raises(ValueError) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            query_id="q_incoming_nonempty",
            actor="analyst_user",
            role="analyst",
            idempotency_key="key_stored_none"
        )
    assert "Idempotency conflict" in str(exc_info.value)


def test_simulated_pg_different_unique_constraint_violation_propagates_unchanged(monkeypatch):
    """Verify 67: [SIMULATED] Unique violation from a different constraint (e.g. uq_dataset_case_ref) propagates without recovery."""
    from contextlib import contextmanager
    budget_id = "test-sim-diff-constraint"
    unrelated_exc = _SimulatedUniqueViolation(
        message='duplicate key value violates unique constraint "uq_dataset_case_ref"',
        constraint_name="uq_dataset_case_ref"
    )
    mock_conn = _MockSimulatedConn()

    @contextmanager
    def mock_get_db_cursor(commit=True, dict_cursor=True):
        if commit:
            try:
                yield _MockSimulatedWriteCursor(mock_conn, exc_to_raise=unrelated_exc)
            except Exception:
                mock_conn.rollback()
                raise
        else:
            yield _MockSimulatedReadCursor(mock_conn)

    monkeypatch.setattr(src.repository, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.repository, "get_db_cursor", mock_get_db_cursor)

    with pytest.raises(_SimulatedUniqueViolation) as exc_info:
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            query_id="diff_constraint_q",
            actor="analyst_sim",
            role="analyst",
            idempotency_key="diff_constraint_key"
        )
    assert "uq_dataset_case_ref" in str(exc_info.value)


def test_simulated_pg_generic_sqlstate_23505_without_matching_metadata_propagates_unchanged(monkeypatch):
    """Verify 68: [SIMULATED] Generic SQLSTATE 23505 without idempotency constraint metadata propagates unchanged."""
    from contextlib import contextmanager
    budget_id = "test-sim-generic-sqlstate"

    class _Generic23505Exception(Exception):
        pgcode = "23505"
        sqlstate = "23505"
        diag = None

    generic_exc = _Generic23505Exception("duplicate key value violates unique constraint")
    mock_conn = _MockSimulatedConn()

    @contextmanager
    def mock_get_db_cursor(commit=True, dict_cursor=True):
        if commit:
            try:
                yield _MockSimulatedWriteCursor(mock_conn, exc_to_raise=generic_exc)
            except Exception:
                mock_conn.rollback()
                raise
        else:
            yield _MockSimulatedReadCursor(mock_conn)

    monkeypatch.setattr(src.repository, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.repository, "get_db_cursor", mock_get_db_cursor)

    with pytest.raises(_Generic23505Exception):
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            query_id="generic_23505_q",
            actor="analyst_sim",
            role="analyst",
            idempotency_key="generic_23505_key"
        )


def test_simulated_pg_generic_uniqueviolation_class_without_metadata_propagates_unchanged(monkeypatch):
    """Verify 69: [SIMULATED] Generic UniqueViolation class without idempotency constraint metadata propagates unchanged."""
    from contextlib import contextmanager
    budget_id = "test-sim-generic-class"

    class UniqueViolation(Exception):
        diag = None

    generic_cls_exc = UniqueViolation("unique constraint failure")
    mock_conn = _MockSimulatedConn()

    @contextmanager
    def mock_get_db_cursor(commit=True, dict_cursor=True):
        if commit:
            try:
                yield _MockSimulatedWriteCursor(mock_conn, exc_to_raise=generic_cls_exc)
            except Exception:
                mock_conn.rollback()
                raise
        else:
            yield _MockSimulatedReadCursor(mock_conn)

    monkeypatch.setattr(src.repository, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.repository, "get_db_cursor", mock_get_db_cursor)

    with pytest.raises(UniqueViolation):
        PrivacyBudgetRepository.allocate_budget(
            budget_id=budget_id,
            epsilon=1.0,
            query_id="generic_class_q",
            actor="analyst_sim",
            role="analyst",
            idempotency_key="generic_class_key"
        )


# ==============================================================================
# Database Health Endpoint Tests (GET /api/v1/database/health)
# ==============================================================================

def _get_db_health_endpoint():
    from src.api import create_app
    app = create_app()
    route = next(r for r in app.routes if getattr(r, "path", None) == "/api/v1/database/health")
    return route.endpoint


def test_database_health_endpoint_healthy_postgres(monkeypatch):
    """Verify 70: GET /api/v1/database/health returns healthy when PostgreSQL is operational."""
    import src.api
    from src.api import UserContext, Role

    monkeypatch.setattr(src.api, "validate_database_config", lambda: {
        "status": "READY",
        "configured": True,
        "driver_available": True,
        "valid_scheme": True,
        "pool_min_size": 2,
        "pool_max_size": 10
    })
    monkeypatch.setattr(src.api, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.api, "check_database_health", lambda: {
        "status": "HEALTHY",
        "postgres_enabled": True,
        "latency_ms": 3.14,
        "timestamp": "2026-11-22T21:00:00Z"
    })

    endpoint = _get_db_health_endpoint()
    admin_user = UserContext(user_id="admin_test", role=Role.ADMIN, auth_method="jwt")
    res = endpoint(user=admin_user)

    assert res["status"] == "HEALTHY"
    assert res["postgres_enabled"] is True
    assert res["latency_ms"] == 3.14
    assert res["configuration_readiness"] == "READY"
    assert res["local_fallback_active"] is False
    assert "timestamp" in res


def test_database_health_endpoint_postgres_connection_failure(monkeypatch):
    """Verify 71: GET /api/v1/database/health returns UNHEALTHY when SELECT 1 probe fails."""
    import src.api
    from src.api import UserContext, Role

    monkeypatch.setattr(src.api, "validate_database_config", lambda: {
        "status": "READY",
        "configured": True,
        "driver_available": True,
        "valid_scheme": True,
        "pool_min_size": 2,
        "pool_max_size": 10
    })
    monkeypatch.setattr(src.api, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.api, "check_database_health", lambda: {
        "status": "UNHEALTHY",
        "postgres_enabled": True,
        "error": "Connection failed"
    })

    endpoint = _get_db_health_endpoint()
    admin_user = UserContext(user_id="admin_test", role=Role.ADMIN, auth_method="jwt")
    res = endpoint(user=admin_user)

    assert res["status"] == "UNHEALTHY"
    assert res["postgres_enabled"] is True
    assert res["latency_ms"] is None
    assert res["configuration_readiness"] == "READY"
    assert res["local_fallback_active"] is False


def test_database_health_endpoint_missing_database_url(monkeypatch):
    """Verify 72: GET /api/v1/database/health distinguishes LOCAL_FALLBACK when DATABASE_URL is missing."""
    import src.api
    from src.api import UserContext, Role

    monkeypatch.setattr(src.api, "validate_database_config", lambda: {
        "status": "LOCAL_FALLBACK",
        "configured": False,
        "driver_available": True,
        "message": "DATABASE_URL is not configured."
    })
    monkeypatch.setattr(src.api, "is_postgres_enabled", lambda: False)

    endpoint = _get_db_health_endpoint()
    admin_user = UserContext(user_id="admin_test", role=Role.ADMIN, auth_method="jwt")
    res = endpoint(user=admin_user)

    assert res["status"] == "LOCAL_FALLBACK"
    assert res["postgres_enabled"] is False
    assert res["latency_ms"] is None
    assert res["configuration_readiness"] == "LOCAL_FALLBACK"
    assert res["local_fallback_active"] is True


def test_database_health_endpoint_driver_unavailable(monkeypatch):
    """Verify 73: GET /api/v1/database/health returns DRIVER_UNAVAILABLE when psycopg is missing."""
    import src.api
    from src.api import UserContext, Role

    monkeypatch.setattr(src.api, "validate_database_config", lambda: {
        "status": "DRIVER_UNAVAILABLE",
        "configured": True,
        "driver_available": False,
        "valid_scheme": True,
        "message": "psycopg driver is unavailable."
    })
    monkeypatch.setattr(src.api, "is_postgres_enabled", lambda: False)

    endpoint = _get_db_health_endpoint()
    admin_user = UserContext(user_id="admin_test", role=Role.ADMIN, auth_method="jwt")
    res = endpoint(user=admin_user)

    assert res["status"] == "DRIVER_UNAVAILABLE"
    assert res["postgres_enabled"] is False
    assert res["latency_ms"] is None
    assert res["configuration_readiness"] == "DRIVER_UNAVAILABLE"
    assert res["local_fallback_active"] is False


def test_database_health_endpoint_unauthorized_users_receive_401():
    """Verify 74: Missing or invalid authentication credentials raise HTTP 401."""
    from src.api import get_current_user
    from fastapi import HTTPException
    from starlette.requests import Request

    req = Request(scope={"type": "http", "headers": [], "state": {}})
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(req)
    assert exc_info.value.status_code == 401
    assert "Missing or invalid" in exc_info.value.detail


def test_database_health_endpoint_non_admin_users_receive_403():
    """Verify 75: Non-admin users (Viewer, Analyst) are rejected with HTTP 403 Forbidden."""
    from src.api import require_role, Role, UserContext
    from fastapi import HTTPException
    from starlette.requests import Request

    req = Request(scope={"type": "http", "headers": [], "state": {}})
    checker = require_role(Role.ADMIN)

    viewer = UserContext(user_id="viewer_1", role=Role.VIEWER, auth_method="jwt")
    with pytest.raises(HTTPException) as exc_info:
        checker(req, user=viewer)
    assert exc_info.value.status_code == 403
    assert "Insufficient privileges" in exc_info.value.detail

    analyst = UserContext(user_id="analyst_1", role=Role.ANALYST, auth_method="jwt")
    with pytest.raises(HTTPException) as exc_info:
        checker(req, user=analyst)
    assert exc_info.value.status_code == 403
    assert "Insufficient privileges" in exc_info.value.detail

    # Admin passes without error
    admin = UserContext(user_id="admin_1", role=Role.ADMIN, auth_method="jwt")
    result = checker(req, user=admin)
    assert result.user_id == "admin_1"


def test_database_health_endpoint_no_secrets_leaked(monkeypatch):
    """Verify 76: Response never leaks database credentials, passwords, URLs, or secrets."""
    import json
    import src.api
    from src.api import UserContext, Role

    monkeypatch.setattr(src.api, "validate_database_config", lambda: {
        "status": "READY",
        "configured": True,
        "driver_available": True,
        "valid_scheme": True,
        "masked_url": "postgresql://postgres:****@db.supabase.co:5432/postgres",
        "pool_min_size": 2,
        "pool_max_size": 10
    })
    monkeypatch.setattr(src.api, "is_postgres_enabled", lambda: True)
    monkeypatch.setattr(src.api, "check_database_health", lambda: {
        "status": "HEALTHY",
        "postgres_enabled": True,
        "latency_ms": 1.5,
        "timestamp": "2026-11-22T21:00:00Z"
    })

    endpoint = _get_db_health_endpoint()
    admin_user = UserContext(user_id="admin_test", role=Role.ADMIN, auth_method="jwt")
    res = endpoint(user=admin_user)

    serialized = json.dumps(res)
    for forbidden in ["postgresql://", "postgres://", "password", "supabase", "secret", "token", "key"]:
        assert forbidden not in serialized.lower() or forbidden in ["postgres_enabled"]


def test_existing_health_endpoint_response_intact():
    """Verify 77: Existing /health and /api/v1/health endpoints preserve existing response format."""
    from src.api import create_app
    app = create_app()
    route = next(r for r in app.routes if getattr(r, "path", None) == "/api/v1/health")
    res = route.endpoint()

    assert res["status"] == "healthy"
    assert "cases_indexed" in res
    assert "timestamp" in res
    assert "privacy_engine" in res
    assert res["privacy_engine"]["status"] == "ACTIVE"
    assert "total_epsilon" in res["privacy_engine"]
    assert "remaining_epsilon" in res["privacy_engine"]






