"""
src/database.py - Supabase PostgreSQL Persistent Infrastructure & Connection Pooling.

Architectural Guarantees:
1. Lazy & Defensive PostgreSQL Driver Loading:
   - Defensively imports psycopg and psycopg_pool.
   - Gracefully handles Windows environments where libpq or pq.dll is blocked by Application Control / AppLocker.
2. Connection Pooling:
   - Configurable pool limits (DATABASE_POOL_MIN_SIZE, DATABASE_POOL_MAX_SIZE, DATABASE_POOL_TIMEOUT, DATABASE_POOL_MAX_IDLE).
   - Thread-safe singleton pool lifecycle with automatic connection health verification.
3. Safe Resource Management:
   - Context managers (get_db_connection, get_db_cursor) ensure deterministic connection release and rollback on error.
4. Graceful Local Fallback:
   - Seamlessly operates in DuckDB-only mode when DATABASE_URL is unset or PostgreSQL drivers are unavailable.
5. Strict Credential Isolation:
   - Uses server-side environment variables only (DATABASE_URL). Never hardcodes secrets or exposes credentials to logs or frontend.
"""

import os
import re
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Any, Optional

DEVELOPMENT_ENVIRONMENTS = {"development", "dev", "test", "testing", "local"}
PRODUCTION_ENVIRONMENTS = {"production", "prod", "staging", "stage", "preprod"}

# Defensive import of psycopg and psycopg_pool
# In Windows environments with strict Application Control policies, loading pq.dll can fail.
try:
    import psycopg
    from psycopg.rows import dict_row
except (ImportError, Exception):
    psycopg = None
    dict_row = None

try:
    import psycopg_pool
    from psycopg_pool import ConnectionPool
except (ImportError, Exception):
    psycopg_pool = None
    ConnectionPool = None

logger = logging.getLogger("caselens.database")

DEFAULT_DRAFT_RETENTION_DAYS = 7
DEFAULT_POOL_MIN_SIZE = 1
DEFAULT_POOL_MAX_SIZE = 10
DEFAULT_POOL_TIMEOUT = 30.0
DEFAULT_POOL_MAX_IDLE = 300.0

_db_pool: Optional[Any] = None
_db_pool_lock = threading.RLock()
_db_pool_url: str = ""


def is_psycopg_available() -> bool:
    """Returns True if the psycopg driver and psycopg_pool are successfully loaded."""
    return psycopg is not None and psycopg_pool is not None


def get_database_url() -> Optional[str]:
    """Retrieves and normalizes DATABASE_URL from environment variables."""
    raw = os.environ.get("DATABASE_URL", "").strip()
    while len(raw) >= 2 and ((raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'"))):
        raw = raw[1:-1].strip()
    return raw if raw else None


def is_postgres_enabled() -> bool:
    """
    Returns True if PostgreSQL is enabled and available:
    Requires DATABASE_URL to be set and psycopg driver to be available.
    """
    return bool(get_database_url()) and is_psycopg_available()


def is_production_or_staging() -> bool:
    """
    Returns True if the environment is explicitly configured for production or staging.
    """
    raw_env = (
        os.environ.get("CASELENS_ENV")
        or os.environ.get("ENVIRONMENT")
        or os.environ.get("APP_ENV")
        or os.environ.get("FASTAPI_ENV")
        or os.environ.get("NODE_ENV")
        or ""
    ).strip().lower()

    return raw_env in PRODUCTION_ENVIRONMENTS


def is_explicit_development_or_test() -> bool:
    """
    Returns True ONLY if the environment is explicitly declared for local development or testing.
    Ambiguous, unset, production, staging, or hosted environments return False.
    """
    # Production or staging can NEVER be considered development or test
    if is_production_or_staging():
        return False

    raw_env = (
        os.environ.get("CASELENS_ENV")
        or os.environ.get("ENVIRONMENT")
        or os.environ.get("APP_ENV")
        or os.environ.get("FASTAPI_ENV")
        or os.environ.get("NODE_ENV")
        or ""
    ).strip().lower()

    is_dev = raw_env in DEVELOPMENT_ENVIRONMENTS
    is_pytest = "PYTEST_CURRENT_TEST" in os.environ

    # Hosted cloud environments (e.g. Render) cannot run in local fallback mode
    is_hosted = bool(os.environ.get("RENDER") or os.environ.get("RENDER_SERVICE_ID"))
    if is_hosted:
        return False

    return is_dev or is_pytest


def enforce_database_safety() -> None:
    """
    Validates database configuration and enforces strict fail-closed behavior
    in production, staging, and ambiguous/unverified deployment environments.

    Guarantees:
    1. Identifies production and staging environments.
    2. Requires DATABASE_URL in production, staging, and unverified environments.
    3. Requires the PostgreSQL driver (psycopg, psycopg_pool) to load successfully.
    4. Rejects invalid database configuration (e.g. malformed URI schemes).
    5. Raises a clear RuntimeError instead of silently allowing RAM fallback.
    6. Allows local in-memory fallback ONLY in explicitly recognized development or test environments.
    7. Does not expose database credentials in errors or logs.
    """
    is_dev_test = is_explicit_development_or_test()
    is_hosted = bool(os.environ.get("RENDER") or os.environ.get("RENDER_SERVICE_ID"))
    db_url = get_database_url()

    # 1. Explicit Development / Test mode without DATABASE_URL: permitted to use local fallback
    if is_dev_test and not db_url:
        logger.info("Running in explicit local development/test mode with DuckDB/in-memory fallback.")
        return

    # 2. Non-development environments (Production, Staging, Hosted, or Ambiguous/Unverified deployments):
    # Must FAIL CLOSED if DATABASE_URL is missing
    if not db_url:
        if is_production_or_staging() or is_hosted:
            raise RuntimeError(
                "CRITICAL DATABASE CONFIGURATION ERROR: 'DATABASE_URL' environment variable is missing or empty "
                "in production/staging/hosted environment. Silent fallback to temporary in-memory storage is strictly prohibited."
            )
        else:
            raise RuntimeError(
                "CRITICAL DATABASE CONFIGURATION ERROR: 'DATABASE_URL' environment variable is missing or empty "
                "in an unverified deployment environment. To allow local in-memory fallback, explicitly declare "
                "ENVIRONMENT=development or ENVIRONMENT=test."
            )

    # 3. Validate URL structure and scheme
    clean_url = db_url.lower()
    valid_scheme = clean_url.startswith("postgresql://") or clean_url.startswith("postgres://")
    if not valid_scheme:
        masked = mask_database_url(db_url)
        raise RuntimeError(
            f"CRITICAL DATABASE CONFIGURATION ERROR: Invalid DATABASE_URL scheme '{masked}'. "
            "URL must start with 'postgresql://' or 'postgres://'."
        )

    # 4. Require PostgreSQL driver availability
    if not is_psycopg_available():
        if is_production_or_staging() or is_hosted:
            raise RuntimeError(
                "CRITICAL DATABASE DRIVER ERROR: PostgreSQL driver 'psycopg' or 'psycopg_pool' is not installed "
                "or failed to load in production/staging/hosted environment. Silent fallback to in-memory storage is strictly prohibited."
            )
        elif not is_dev_test:
            raise RuntimeError(
                "CRITICAL DATABASE DRIVER ERROR: PostgreSQL driver 'psycopg' or 'psycopg_pool' is unavailable "
                "in an unverified deployment environment. Set ENVIRONMENT=development to permit local fallback."
            )
        else:
            raise RuntimeError(
                "DATABASE DRIVER ERROR: DATABASE_URL is configured, but 'psycopg' or 'psycopg_pool' is not available "
                "in this local environment. Check requirements or unset DATABASE_URL to use local fallback."
            )


def verify_repository_fallback_allowed() -> None:
    """
    Verifies that local in-memory fallback (_local_cases, _local_datasets, _local_profiles)
    is permitted for the current operational context.
    Raises RuntimeError if invoked in production, staging, hosted, or ambiguous deployment environments.
    """
    is_hosted = bool(os.environ.get("RENDER") or os.environ.get("RENDER_SERVICE_ID"))
    if is_production_or_staging() or is_hosted or not is_explicit_development_or_test():
        raise RuntimeError(
            "FATAL SECURITY VIOLATION: Repository in-memory fallback (_local_cases/_local_datasets/_local_profiles) "
            "is strictly forbidden outside of explicit development or test environments. "
            "Persistent PostgreSQL connection is required."
        )


def mask_database_url(url: Optional[str]) -> str:
    """
    Masks credentials in a database connection URL for safe logging and diagnostics.
    Never exposes passwords, tokens, or raw credentials.
    """
    if not url:
        return "<unset>"
    try:
        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)
    except Exception:
        return "<masked-db-url>"


def get_draft_retention_days() -> int:
    """Returns configured draft retention duration in days (default: 7)."""
    raw = os.environ.get("DRAFT_RETENTION_DAYS", "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return DEFAULT_DRAFT_RETENTION_DAYS


def get_pool_config() -> Dict[str, Any]:
    """Returns connection pool sizing and timeout configuration from environment."""
    min_size = int(os.environ.get("DATABASE_POOL_MIN_SIZE", str(DEFAULT_POOL_MIN_SIZE)))
    max_size = int(os.environ.get("DATABASE_POOL_MAX_SIZE", str(DEFAULT_POOL_MAX_SIZE)))
    timeout = float(os.environ.get("DATABASE_POOL_TIMEOUT", str(DEFAULT_POOL_TIMEOUT)))
    max_idle = float(os.environ.get("DATABASE_POOL_MAX_IDLE", str(DEFAULT_POOL_MAX_IDLE)))
    return {
        "min_size": max(1, min_size),
        "max_size": max(min_size, max_size),
        "timeout": max(1.0, timeout),
        "max_idle": max(1.0, max_idle)
    }


def validate_database_config() -> Dict[str, Any]:
    """
    Validates database configuration and returns a sanitized status dictionary.
    Safe for health check endpoints; never exposes secrets.
    """
    db_url = get_database_url()
    driver_avail = is_psycopg_available()

    if not db_url:
        return {
            "configured": False,
            "driver_available": driver_avail,
            "status": "LOCAL_FALLBACK",
            "message": "DATABASE_URL is not configured. Running in DuckDB local-only mode."
        }

    # Validate URL structure
    clean_url = db_url.lower()
    valid_scheme = clean_url.startswith("postgresql://") or clean_url.startswith("postgres://")
    if not valid_scheme:
        return {
            "configured": True,
            "valid_scheme": False,
            "driver_available": driver_avail,
            "status": "INVALID_CONFIGURATION",
            "message": "DATABASE_URL must start with 'postgresql://' or 'postgres://'."
        }

    if not driver_avail:
        return {
            "configured": True,
            "valid_scheme": True,
            "driver_available": False,
            "status": "DRIVER_UNAVAILABLE",
            "message": "DATABASE_URL is set but psycopg driver is unavailable in this environment."
        }

    cfg = get_pool_config()
    return {
        "configured": True,
        "valid_scheme": True,
        "driver_available": True,
        "masked_url": mask_database_url(db_url),
        "pool_min_size": cfg["min_size"],
        "pool_max_size": cfg["max_size"],
        "status": "READY"
    }


def init_db_pool(db_url: Optional[str] = None) -> Any:
    """
    Initializes and opens a reusable psycopg_pool.ConnectionPool.
    Thread-safe singleton with health check validation on checkout.
    """
    global _db_pool, _db_pool_url
    if not db_url:
        db_url = get_database_url()

    if not db_url:
        raise RuntimeError("Cannot initialize connection pool: DATABASE_URL is not configured.")

    if not is_psycopg_available():
        raise RuntimeError("Cannot initialize connection pool: psycopg_pool is not available in current environment.")

    cfg = get_pool_config()
    with _db_pool_lock:
        if _db_pool is None or _db_pool_url != db_url:
            if _db_pool is not None:
                try:
                    _db_pool.close()
                except Exception:
                    pass

            kwargs: Dict[str, Any] = {
                "conninfo": db_url,
                "min_size": cfg["min_size"],
                "max_size": cfg["max_size"],
                "timeout": cfg["timeout"],
                "max_idle": cfg["max_idle"],
                "open": True
            }
            if hasattr(ConnectionPool, "check_connection"):
                kwargs["check"] = ConnectionPool.check_connection

            _db_pool = ConnectionPool(**kwargs)
            _db_pool_url = db_url
            logger.info(
                "Initialized PostgreSQL connection pool (%s, min=%d, max=%d)",
                mask_database_url(db_url),
                cfg["min_size"],
                cfg["max_size"]
            )
        return _db_pool


def get_db_pool() -> Any:
    """Returns the active ConnectionPool, initializing if necessary."""
    if _db_pool is None:
        return init_db_pool()
    return _db_pool


def close_db_pool() -> None:
    """Safely closes the active ConnectionPool and releases all connections."""
    global _db_pool, _db_pool_url
    with _db_pool_lock:
        if _db_pool is not None:
            try:
                _db_pool.close()
                logger.info("Closed PostgreSQL connection pool.")
            except Exception as e:
                logger.warning("Error while closing connection pool: %s", e)
            finally:
                _db_pool = None
                _db_pool_url = ""


def check_database_health() -> Dict[str, Any]:
    """
    Performs an active database ping (SELECT 1) and measures round-trip latency.
    Never raises an unhandled exception.
    """
    if not is_postgres_enabled():
        return {
            "status": "UNAVAILABLE",
            "postgres_enabled": False,
            "message": "PostgreSQL persistence is not configured or driver is unavailable."
        }

    start_time = datetime.now(timezone.utc)
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute("SELECT 1 AS ping;")
            res = cur.fetchone()
            ping_val = res.get("ping") if isinstance(res, dict) else (res[0] if res else None)
            if ping_val != 1:
                return {"status": "DEGRADED", "error": f"Unexpected ping result: {ping_val}"}

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000.0
        return {
            "status": "HEALTHY",
            "postgres_enabled": True,
            "latency_ms": round(elapsed_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error("PostgreSQL health check failed: %s", e)
        return {
            "status": "UNHEALTHY",
            "postgres_enabled": True,
            "error": "Connection failed"
        }


@contextmanager
def get_db_connection():
    """
    Context manager that acquires a connection from the pool and returns it upon exit.
    Rolls back any uncommitted transaction on exception.
    """
    if not is_postgres_enabled():
        raise RuntimeError("Cannot acquire database connection: PostgreSQL is not configured or driver is unavailable.")

    pool = get_db_pool()
    conn = None
    try:
        with pool.connection() as conn:
            yield conn
    except Exception:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        raise


@contextmanager
def get_db_cursor(commit: bool = True, dict_cursor: bool = True):
    """
    Context manager that acquires a connection and yields a cursor within a transaction.
    Automatically commits if commit=True and no exception occurs; rolls back on error.
    """
    with get_db_connection() as conn:
        cursor_kwargs: Dict[str, Any] = {}
        if dict_cursor and dict_row is not None:
            cursor_kwargs["row_factory"] = dict_row

        cur = conn.cursor(**cursor_kwargs)
        try:
            yield cur
            if commit:
                conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass
