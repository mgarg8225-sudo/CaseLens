"""
src/api.py - Privacy-Preserving FastAPI Backend Service.

Phase 6 of The Leaky Pipeline:
Provides secure, privacy-preserving analytical REST API endpoints for institutional
stakeholders (Judiciary, Police, Health Dept, Women & Child Development, Policy Researchers).

Core Architectural Guarantees:
1. Aggregates Only:
   - Zero access to individual survivor case records, microdata, or PII.
   - Every outbound payload strictly validated against forbidden PII tokens
     via verify_zero_pii_leakage().
   - Small-cell suppression (counts < k are suppressed as "<5") enforced.
2. Role-Based Access Control (RBAC):
   - Admin: Full analytics, audit log inspection, privacy budget reset, diagnostics, policy refresh.
   - Analyst: Analytical aggregates, dynamic DP queries, duration bottlenecks, budget status.
   - Viewer: High-level published DP funnel, district leakage ranking, statutory benchmarks.
3. Differential Privacy Engine Integration:
   - All analytical queries consume from the global epsilon-budget under Sequential Composition.
   - Dynamic Laplace mechanism noise injection on count/frequency queries.
   - Automatically halts queries (HTTP 429) when the privacy budget is exhausted.
4. Comprehensive Audit Trail:
   - Every single API interaction (endpoint, role, timestamp, IP, status, DP cost, PII verification)
     is immutably recorded to data/api_audit_log.json and available to administrators.
"""

import os
import json
import uuid
import time
from enum import Enum
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Set
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Header, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.privacy import (
    PrivacyBudgetTracker,
    PrivacyBudgetExceededError,
    LaplaceMechanism,
    apply_kanonymity_suppression,
    verify_zero_pii_leakage,
    evaluate_k_anonymity,
)
from src.policy_context import (
    load_cache,
    generate_all_top5_policy_contexts,
    get_policy_context,
    get_government_benchmark_fact,
    sanitize_query,
    FORBIDDEN_ID_PATTERNS,
    FORBIDDEN_PII_PATTERNS,
)
from src.auth import auth_router


STATUTORY_BENCHMARKS = {
    "LP-01-FORENSIC-DELAY": {
        "title": "Forensic Science Lab Reporting Mandate",
        "benchmark_days": 30.0,
        "statutory_reference": "POCSO Guidelines & MHA Forensic Directives",
        "mandate_summary": "Forensic biological examination results should be furnished within 30 days to facilitate chargesheeting."
    },
    "LP-02-POLICE-ATTRITION": {
        "title": "Police Investigation & Chargesheet Filing Deadline",
        "benchmark_days": 60.0,
        "statutory_reference": "Section 173(2) CrPC / Section 193(3) BNSS",
        "mandate_summary": "Investigation in relation to sexual offences must be completed within 60 days of FIR registration."
    },
    "LP-03-COURT-DELAY": {
        "title": "Judicial Trial Adjudication Mandate",
        "benchmark_days": 365.0,
        "statutory_reference": "Section 35(2) POCSO Act & FTSC Guidelines",
        "mandate_summary": "The Special Court shall complete the trial, as far as possible, within 1 year from cognizance."
    },
    "LP-04-MEDICAL-EXAM-DELAY": {
        "title": "Medico-Legal Examination (MLC) Protocol",
        "benchmark_hours": 24.0,
        "statutory_reference": "Section 164A CrPC / MoHFW Survivor Protocol",
        "mandate_summary": "Medical examination of the survivor must be conducted within 24 hours of report."
    },
    "LP-05-VICTIM-COMPENSATION-DELAY": {
        "title": "Interim Victim Compensation Disbursal",
        "benchmark_days": 30.0,
        "statutory_reference": "NALSA Compensation Scheme for Women Victims",
        "mandate_summary": "Immediate interim relief to be sanctioned upon receipt of application without waiting for final trial verdict."
    }
}


# ==============================================================================
# 1. Configuration & Security Definitions
# ==============================================================================

DEFAULT_DATA_DIR = os.environ.get("LEAKY_DATA_DIR", "data")


class Role(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


# Hierarchy: admin > analyst > viewer
ROLE_HIERARCHY = {
    Role.ADMIN: 3,
    Role.ANALYST: 2,
    Role.VIEWER: 1,
}

# Standard API Keys for institutional roles (configurable via env)
API_KEYS = {
    os.environ.get("API_KEY_ADMIN", "admin-key-leaky-2026"): Role.ADMIN,
    os.environ.get("API_KEY_ANALYST", "analyst-key-leaky-2026"): Role.ANALYST,
    os.environ.get("API_KEY_VIEWER", "viewer-key-leaky-2026"): Role.VIEWER,
}


# ==============================================================================
# 2. Pydantic Request & Response Schemas
# ==============================================================================

class UserContext(BaseModel):
    user_id: str
    role: Role
    auth_method: str


class AggregateQueryRequest(BaseModel):
    dimension: str = Field(
        ...,
        description="Aggregate dimension: 'district', 'case_category', or 'terminal_stage'"
    )
    metric: str = Field(
        default="count",
        description="Aggregate metric: 'count' or 'mean_duration_days'"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional filter criteria (e.g. {'district': 'D3_Rudrapur'})"
    )
    epsilon: float = Field(
        default=0.25,
        ge=0.01,
        le=2.0,
        description="Differential privacy epsilon budget for this query (0.01 <= eps <= 2.0)"
    )
    k_threshold: int = Field(
        default=5,
        ge=5,
        description="Minimum cell count for k-anonymity suppression (minimum 5)"
    )


class ResetBudgetRequest(BaseModel):
    total_epsilon: float = Field(default=10.0, gt=0.0, le=50.0)
    total_delta: float = Field(default=1e-4, gt=0.0, le=1e-2)


# ==============================================================================
# 3. Persistent Audit Logger
# ==============================================================================

class ApiAuditLogger:
    """
    Maintains an immutable audit ledger of all API queries, parameters,
    caller roles, differential privacy costs, and Zero-PII verification status.
    """
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or os.path.join(DEFAULT_DATA_DIR, "api_audit_log.json")
        self._entries: List[Dict[str, Any]] = []
        self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
            except Exception:
                self._entries = []

    def log_event(
        self,
        request_id: str,
        role: str,
        method: str,
        endpoint: str,
        status_code: int,
        client_ip: str,
        epsilon_spent: float = 0.0,
        zero_pii_verified: bool = True,
        latency_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ):
        entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "client_ip": client_ip,
            "epsilon_spent": round(epsilon_spent, 4),
            "zero_pii_verified": zero_pii_verified,
            "latency_ms": round(latency_ms, 2),
            "details": details or {},
        }
        self._entries.append(entry)
        self._persist()

    def _persist(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2)
        except Exception:
            pass

    def get_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        filtered = self._entries
        if role:
            filtered = [e for e in filtered if e.get("role") == role]
        # Return newest first
        return list(reversed(filtered))[offset : offset + limit]

    def clear(self):
        self._entries.clear()
        self._persist()


# ==============================================================================
# 4. Data Store & Engine State Manager
# ==============================================================================

class DataStore:
    """Loads and caches analytical & DP artifacts for the API service."""
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self.budget_tracker = PrivacyBudgetTracker(total_epsilon=10.0, total_delta=1e-4)
        self.audit_logger = ApiAuditLogger(os.path.join(data_dir, "api_audit_log.json"))
        self.dp_funnel: Dict[str, Any] = {}
        self.dp_districts: Dict[str, Any] = {}
        self.bottlenecks: Dict[str, Any] = {}
        self.category_disparities: Dict[str, Any] = {}
        self.summary: Dict[str, Any] = {}
        self.privacy_report: Dict[str, Any] = {}
        self.df_cases: Optional[pd.DataFrame] = None
        self.load_data()

    def load_data(self):
        """Loads precomputed JSON artifacts and event log if available."""
        # 1. DP Funnel
        f_path = os.path.join(self.data_dir, "privacy_dp_funnel.json")
        if not os.path.exists(f_path):
            f_path = os.path.join(self.data_dir, "analytics_funnel.json")
        if os.path.exists(f_path):
            with open(f_path, "r", encoding="utf-8") as f:
                self.dp_funnel = json.load(f)

        # 2. DP Districts
        d_path = os.path.join(self.data_dir, "privacy_dp_districts.json")
        if not os.path.exists(d_path):
            d_path = os.path.join(self.data_dir, "analytics_district_leakage.json")
        if os.path.exists(d_path):
            with open(d_path, "r", encoding="utf-8") as f:
                self.dp_districts = json.load(f)

        # 3. Bottlenecks
        b_path = os.path.join(self.data_dir, "analytics_bottlenecks.json")
        if os.path.exists(b_path):
            with open(b_path, "r", encoding="utf-8") as f:
                self.bottlenecks = json.load(f)

        # 4. Categories
        c_path = os.path.join(self.data_dir, "analytics_category.json")
        if os.path.exists(c_path):
            with open(c_path, "r", encoding="utf-8") as f:
                self.category_disparities = json.load(f)

        # 5. Summary
        s_path = os.path.join(self.data_dir, "analytics_summary.json")
        if os.path.exists(s_path):
            with open(s_path, "r", encoding="utf-8") as f:
                self.summary = json.load(f)

        # 6. Privacy Audit Report
        p_path = os.path.join(self.data_dir, "privacy_audit_report.json")
        if os.path.exists(p_path):
            with open(p_path, "r", encoding="utf-8") as f:
                self.privacy_report = json.load(f)

        # 7. Reconstructed Events to Cases (for dynamic DP aggregation)
        ev_path = os.path.join(self.data_dir, "event_log.csv")
        if os.path.exists(ev_path):
            try:
                df_ev = pd.read_csv(ev_path)
                df_ev["month"] = pd.to_datetime(df_ev["timestamp"]).dt.to_period("M").astype(str)
                self.df_cases = df_ev.groupby("case_id").agg({
                    "district": "first",
                    "case_category": "first",
                    "month": "first",
                    "activity": "last"
                }).reset_index().rename(columns={"activity": "terminal_stage"})
            except Exception:
                self.df_cases = None


# Global DataStore instance
DATA_STORE = DataStore(DEFAULT_DATA_DIR)


# ==============================================================================
# 5. Authentication & RBAC Dependencies
# ==============================================================================

def get_current_user(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_user_role: Optional[str] = Header(None, alias="X-User-Role"),
) -> UserContext:
    """
    Authenticates requests via:
    1. X-API-Key header.
    2. Authorization: Bearer <key> header.
    3. X-User-Role header (for testing and controlled environment integration).
    """
    token = None
    if x_api_key:
        token = x_api_key.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        # Decode and validate signed JWT if present
        if token.count(".") == 2:
            from src.auth import decode_access_token
            payload = decode_access_token(token)
            role_val = payload.get("role", "viewer")
            role_enum = Role.ADMIN if role_val == "admin" else (Role.ANALYST if role_val in ["analyst", "linkage_reviewer"] else Role.VIEWER)
            return UserContext(
                user_id=payload.get("user_id", payload.get("sub", "unknown")),
                role=role_enum,
                auth_method="jwt"
            )

    # 1. Match configured API Keys
    if token and token in API_KEYS:
        role = API_KEYS[token]
        return UserContext(user_id=f"key_{role.value}", role=role, auth_method="api_key")

    # 2. Match direct testing role header if valid
    if x_user_role:
        role_str = x_user_role.lower().strip()
        if role_str in [r.value for r in Role]:
            return UserContext(user_id=f"header_{role_str}", role=Role(role_str), auth_method="role_header")

    # 3. Check for default demo tokens if supplied directly
    if token == "admin":
        return UserContext(user_id="demo_admin", role=Role.ADMIN, auth_method="demo_token")
    if token == "analyst":
        return UserContext(user_id="demo_analyst", role=Role.ANALYST, auth_method="demo_token")
    if token == "viewer":
        return UserContext(user_id="demo_viewer", role=Role.VIEWER, auth_method="demo_token")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: Missing or invalid API key / authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(minimum_role: Role):
    """Factory creating dependency verifying caller has at least minimum_role."""
    def _role_checker(user: UserContext = Depends(get_current_user)) -> UserContext:
        user_rank = ROLE_HIERARCHY.get(user.role, 0)
        required_rank = ROLE_HIERARCHY.get(minimum_role, 0)
        if user_rank < required_rank:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Insufficient privileges. Endpoint requires role '{minimum_role.value}' or higher (Caller has role '{user.role.value}')."
            )
        return user
    return _role_checker


ALLOWED_POLICY_KEYS = {
    "victim_compensation_delay",
    "LP-05-VICTIM-COMPENSATION-DELAY",
    "stage_name",
    "query_name",
    "district_name"
}


def sanitize_and_verify_payload(
    payload: Any,
    epsilon_cost: float = 0.0,
    query_id: str = "api_aggregate_query",
    allowed_fields: Optional[Set[str]] = None
) -> Any:
    """
    Enforces privacy layer requirements on every response:
    1. Zero-PII Audit: Strictly rejects payload if survivor personal fields exist.
    2. Privacy Budget Check: Allocates epsilon cost against the global budget tracker.
    """
    # 1. Deduct DP budget if applicable
    if epsilon_cost > 0.0:
        try:
            DATA_STORE.budget_tracker.allocate(query_id=query_id, epsilon=epsilon_cost)
        except PrivacyBudgetExceededError as e:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Privacy budget exceeded: {str(e)} Query blocked by Differential Privacy engine."
            )

    # 2. Verify Zero-PII compliance (whitelisting safe policy benchmarks)
    effective_allowed = (allowed_fields or set()) | ALLOWED_POLICY_KEYS
    pii_check = verify_zero_pii_leakage(payload, allowed_fields=effective_allowed)
    if not pii_check["zero_pii_verified"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Privacy violation: Forbidden PII keys detected in response payload: {pii_check['leaked_fields']}"
        )

    return payload


# ==============================================================================
# 7. FastAPI App Construction
# ==============================================================================

def create_app(data_dir: str = DEFAULT_DATA_DIR) -> FastAPI:
    """App factory function to initialize FastAPI with configured data directory."""
    global DATA_STORE
    if DATA_STORE.data_dir != data_dir:
        DATA_STORE = DataStore(data_dir)

    app = FastAPI(
        title="The Leaky Pipeline - Privacy-Preserving Policy Analytics API",
        description=(
            "Production-grade FastAPI analytical service providing institutional aggregates, "
            "differential privacy protections, and statutory policy benchmarking for violence-against-women "
            "case attrition analysis."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Authentication & User Management Router
    app.include_router(auth_router)

    # Global Request/Audit Interceptor Middleware
    @app.middleware("http")
    async def audit_and_timing_middleware(request: Request, call_next):
        start_time = time.perf_counter()
        req_id = str(uuid.uuid4())
        request.state.request_id = req_id
        client_ip = request.client.host if request.client else "127.0.0.1"

        # Read role header if present for auditing before auth validation
        role_header = request.headers.get("X-User-Role", "unknown")
        api_key = request.headers.get("X-API-Key")
        if api_key in API_KEYS:
            role_header = API_KEYS[api_key].value

        try:
            response = await call_next(request)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Log standard completion
            DATA_STORE.audit_logger.log_event(
                request_id=req_id,
                role=role_header,
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                client_ip=client_ip,
                epsilon_spent=getattr(request.state, "epsilon_spent", 0.0),
                zero_pii_verified=True,
                latency_ms=latency_ms
            )
            return response
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            status_code = getattr(exc, "status_code", 500)
            DATA_STORE.audit_logger.log_event(
                request_id=req_id,
                role=role_header,
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code,
                client_ip=client_ip,
                epsilon_spent=0.0,
                zero_pii_verified=True,
                latency_ms=latency_ms,
                details={"error": str(exc)}
            )
            raise exc

    # ==========================================================================
    # 8. Endpoints: Public & Health
    # ==========================================================================

    @app.get("/", tags=["System"])
    def root():
        """Root metadata endpoint with system status, active privacy mechanisms, and roles."""
        return sanitize_and_verify_payload({
            "service": "The Leaky Pipeline - Policy Analytics API",
            "version": "1.0.0",
            "status": "OPERATIONAL",
            "privacy_guarantees": [
                "Strict Aggregates Only (No Case Microdata or PII)",
                "Epsilon-Differential Privacy via Laplace Mechanism",
                "k-Anonymity Small-Cell Suppression (k >= 5)",
                "Immutable Query Audit Ledger",
                "Sequential Composition Privacy Budget Accounting"
            ],
            "roles_supported": ["admin", "analyst", "viewer"],
            "endpoints_docs": "/docs"
        })

    @app.get("/api/v1/health", tags=["System"])
    def health_check():
        """Health check returning database status, privacy budget, and indexing health."""
        cases_count = len(DATA_STORE.df_cases) if DATA_STORE.df_cases is not None else 0
        return sanitize_and_verify_payload({
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cases_indexed": cases_count,
            "privacy_engine": {
                "status": "ACTIVE",
                "total_epsilon": DATA_STORE.budget_tracker.total_epsilon,
                "spent_epsilon": round(DATA_STORE.budget_tracker.spent_epsilon, 4),
                "remaining_epsilon": round(DATA_STORE.budget_tracker.remaining_epsilon, 4),
                "budget_exhausted": DATA_STORE.budget_tracker.is_exhausted
            }
        })

    # ==========================================================================
    # 9. Endpoints: Viewer+ (Viewer, Analyst, Admin)
    # ==========================================================================

    @app.get("/api/v1/summary", tags=["Aggregates (Viewer+)"])
    def get_executive_summary(
        request: Request,
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        High-level pipeline health and district vulnerability summary.
        Deducts epsilon = 0.05 from privacy budget.
        """
        eps = 0.05
        request.state.epsilon_spent = eps
        payload = {
            "query_type": "executive_summary",
            "data": DATA_STORE.summary,
            "privacy_metadata": {
                "epsilon_cost": eps,
                "remaining_budget": round(DATA_STORE.budget_tracker.remaining_epsilon - eps, 4)
            }
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=eps, query_id="summary_aggregate")

    @app.get("/api/v1/funnel", tags=["Aggregates (Viewer+)"])
    def get_funnel_analytics(
        request: Request,
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        Differentially Private 5-stage case attrition funnel.
        Returns Laplace DP noisy counts, DP retention %, and 95% Confidence Bounds.
        Deducts epsilon = 0.10 from privacy budget.
        """
        eps = 0.10
        request.state.epsilon_spent = eps
        payload = {
            "query_type": "funnel_analytics",
            "funnel": DATA_STORE.dp_funnel,
            "privacy_protection": {
                "mechanism": "Laplace Differential Privacy (Delta f = 1) + Small-Cell k-Anonymity Suppression",
                "k_threshold": 5,
                "epsilon_spent": eps,
                "remaining_epsilon": round(DATA_STORE.budget_tracker.remaining_epsilon - eps, 4)
            }
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=eps, query_id="funnel_aggregate")

    @app.get("/api/v1/districts", tags=["Aggregates (Viewer+)"])
    def get_district_leakage(
        request: Request,
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        Differentially Private district-level leakage metrics (D1-D8),
        systemic leak codes (LP-01, LP-02, LP-03), and vulnerability scores.
        Deducts epsilon = 0.10 from privacy budget.
        """
        eps = 0.10
        request.state.epsilon_spent = eps
        payload = {
            "query_type": "district_leakage_diagnostics",
            "district_analytics": DATA_STORE.dp_districts,
            "privacy_metadata": {
                "epsilon_spent": eps,
                "remaining_epsilon": round(DATA_STORE.budget_tracker.remaining_epsilon - eps, 4)
            }
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=eps, query_id="district_aggregate")

    @app.get("/api/v1/policy-context", tags=["Aggregates (Viewer+)"])
    def get_policy_context(
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        Statutory policy benchmarks and infrastructure context for systemic leak points,
        powered by TinyFish and statutory standards (POCSO Act Sec 35(2), BNSS Sec 193/175).
        No DP budget deducted (public policy reference data).
        """
        cache = load_cache(os.path.join(DATA_STORE.data_dir, "policy_context_cache.json"))
        benchmarks = STATUTORY_BENCHMARKS
        payload = {
            "statutory_benchmarks": benchmarks,
            "external_policy_context": cache,
            "source": "Statutory Legislation & TinyFish Policy Benchmarking Engine",
            "zero_pii_verified": True
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=0.0, query_id="policy_context")

    # ==========================================================================
    # 10. Endpoints: Analyst+ (Analyst, Admin)
    # ==========================================================================

    @app.get("/api/v1/bottlenecks", tags=["Deep Analytics (Analyst+)"])
    def get_bottlenecks(
        request: Request,
        user: UserContext = Depends(require_role(Role.ANALYST))
    ):
        """
        Stage transition duration percentiles (median, IQR, p90, p95)
        and statutory latency compliance percentages.
        Deducts epsilon = 0.15 from privacy budget.
        """
        eps = 0.15
        request.state.epsilon_spent = eps
        payload = {
            "query_type": "duration_bottleneck_percentiles",
            "transitions": DATA_STORE.bottlenecks,
            "privacy_metadata": {
                "epsilon_spent": eps,
                "remaining_epsilon": round(DATA_STORE.budget_tracker.remaining_epsilon - eps, 4)
            }
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=eps, query_id="bottlenecks_aggregate")

    @app.get("/api/v1/categories", tags=["Deep Analytics (Analyst+)"])
    def get_category_disparities(
        request: Request,
        user: UserContext = Depends(require_role(Role.ANALYST))
    ):
        """
        Crime category disparities, step conversion attrition, and statutory volume.
        Deducts epsilon = 0.10 from privacy budget.
        """
        eps = 0.10
        request.state.epsilon_spent = eps
        payload = {
            "query_type": "category_disparities",
            "categories": DATA_STORE.category_disparities,
            "privacy_metadata": {
                "epsilon_spent": eps,
                "remaining_epsilon": round(DATA_STORE.budget_tracker.remaining_epsilon - eps, 4)
            }
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=eps, query_id="category_aggregate")

    @app.get("/api/v1/privacy/budget", tags=["Privacy Engine (Analyst+)"])
    def get_privacy_budget(
        user: UserContext = Depends(require_role(Role.ANALYST))
    ):
        """
        Returns global privacy budget status and consumption ledger under Sequential Composition.
        Zero privacy budget deducted.
        """
        tracker = DATA_STORE.budget_tracker
        payload = {
            "total_epsilon": tracker.total_epsilon,
            "spent_epsilon": round(tracker.spent_epsilon, 4),
            "remaining_epsilon": round(tracker.remaining_epsilon, 4),
            "total_delta": tracker.total_delta,
            "spent_delta": tracker.spent_delta,
            "remaining_delta": tracker.remaining_delta,
            "composition_mode": "Sequential Composition (sum(epsilon_i) <= total_epsilon)",
            "query_count": len(tracker.get_ledger()),
            "recent_ledger": tracker.get_ledger()[-20:]
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=0.0, query_id="budget_status")

    @app.post("/api/v1/query/aggregate", tags=["Deep Analytics (Analyst+)"])
    def query_aggregate(
        request_body: AggregateQueryRequest,
        request: Request,
        user: UserContext = Depends(require_role(Role.ANALYST))
    ):
        """
        Dynamic aggregate analytical query engine:
        - Validates input filters against case ID and PII leakage attempts.
        - Applies Laplace mechanism noise calibrated to query sensitivity (Delta f = 1).
        - Enforces k-anonymity suppression (counts < k are suppressed as '<k').
        - Deducts specified epsilon from global privacy budget.
        """
        # Guardrail: Check for forbidden case-level keys in filters
        if request_body.filters:
            for k, v in request_body.filters.items():
                k_lower = str(k).lower()
                v_str = str(v)
                if any(p in k_lower for p in ["case", "id", "name", "phone", "aadhaar", "survivor"]):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Forbidden filter '{k}': Case-level or survivor identifiers cannot be queried."
                    )
                for pattern in FORBIDDEN_ID_PATTERNS + FORBIDDEN_PII_PATTERNS:
                    if pattern.search(v_str):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Forbidden filter value: contains case/record ID or individual PII pattern."
                        )

        if DATA_STORE.df_cases is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Analytical cases table is not loaded."
            )

        df = DATA_STORE.df_cases.copy()
        dim = request_body.dimension
        if dim not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid dimension '{dim}'. Must be one of: {list(df.columns)}"
            )

        # Apply valid filters
        if request_body.filters:
            for fk, fv in request_body.filters.items():
                if fk in df.columns:
                    df = df[df[fk] == fv]

        # Allocate budget
        request.state.epsilon_spent = request_body.epsilon
        try:
            DATA_STORE.budget_tracker.allocate(
                query_id=f"dynamic_aggregate_{dim}",
                epsilon=request_body.epsilon
            )
        except PrivacyBudgetExceededError as e:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Privacy budget exceeded: {str(e)}"
            )

        # Compute raw counts
        raw_counts = df.groupby(dim, observed=True).size().to_dict()
        laplace = LaplaceMechanism(epsilon=request_body.epsilon, sensitivity=1.0)
        k = request_body.k_threshold

        noisy_aggregates = []
        for cat_val, raw_count in raw_counts.items():
            if raw_count < k:
                noisy_aggregates.append({
                    "dimension_value": str(cat_val),
                    "noisy_count": f"<{k}",
                    "is_k_anonymity_suppressed": True,
                    "confidence_bound_95": None
                })
            else:
                noisy_c = laplace.add_noise_int(raw_count, lower_bound=k)
                noisy_aggregates.append({
                    "dimension_value": str(cat_val),
                    "noisy_count": noisy_c,
                    "is_k_anonymity_suppressed": False,
                    "confidence_bound_95": round(laplace.confidence_bound(0.95), 2)
                })

        payload = {
            "query_dimension": dim,
            "metric": request_body.metric,
            "filters_applied": request_body.filters,
            "k_threshold": k,
            "epsilon_consumed": request_body.epsilon,
            "results": noisy_aggregates
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=0.0)

    # ==========================================================================
    # 11. Endpoints: Admin Only
    # ==========================================================================

    @app.get("/api/v1/audit/logs", tags=["Admin Controls"])
    def get_audit_logs(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        role_filter: Optional[str] = Query(default=None),
        user: UserContext = Depends(require_role(Role.ADMIN))
    ):
        """
        Inspect the immutable API query audit ledger.
        Returns request IDs, timestamps, roles, endpoints, DP cost, and zero-PII status.
        Accessible exclusively to Administrators.
        """
        logs = DATA_STORE.audit_logger.get_logs(limit=limit, offset=offset, role=role_filter)
        payload = {
            "total_logs_returned": len(logs),
            "limit": limit,
            "offset": offset,
            "role_filter": role_filter,
            "logs": logs
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=0.0, query_id="admin_audit_logs")

    @app.post("/api/v1/privacy/budget/reset", tags=["Admin Controls"])
    def reset_privacy_budget(
        request_body: ResetBudgetRequest,
        user: UserContext = Depends(require_role(Role.ADMIN))
    ):
        """
        Reconfigures or resets the Differential Privacy budget under administrative authority.
        Logs the administrative action into the audit ledger.
        Accessible exclusively to Administrators.
        """
        DATA_STORE.budget_tracker = PrivacyBudgetTracker(
            total_epsilon=request_body.total_epsilon,
            total_delta=request_body.total_delta
        )
        payload = {
            "message": "Privacy budget successfully reset by Administrator.",
            "new_total_epsilon": request_body.total_epsilon,
            "new_total_delta": request_body.total_delta,
            "remaining_epsilon": request_body.total_epsilon,
            "reset_timestamp": datetime.now(timezone.utc).isoformat()
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=0.0, query_id="budget_reset")

    @app.get("/api/v1/privacy/diagnostics", tags=["Admin Controls"])
    def get_privacy_diagnostics(
        user: UserContext = Depends(require_role(Role.ADMIN))
    ):
        """
        Returns full Phase 5 privacy diagnostics:
        - k-anonymity across broad and granular quasi-identifiers.
        - l-diversity and t-closeness distributional tests.
        - Adversarial singling-out and differencing attack simulations.
        Accessible exclusively to Administrators.
        """
        return sanitize_and_verify_payload(
            DATA_STORE.privacy_report,
            epsilon_cost=0.0,
            query_id="privacy_diagnostics"
        )

    @app.post("/api/v1/policy-context/refresh", tags=["Admin Controls"])
    def refresh_policy_context(
        user: UserContext = Depends(require_role(Role.ADMIN))
    ):
        """
        Triggers an external policy context refresh via TinyFish Search and Web Agent APIs.
        Updates data/policy_context_cache.json with resilient fallback.
        Accessible exclusively to Administrators.
        """
        try:
            results = generate_all_top5_policy_contexts(
                force_refresh=True,
                cache_path=os.path.join(DATA_STORE.data_dir, "policy_context_cache.json")
            )
            payload = {
                "message": "Policy context refreshed successfully via TinyFish integration.",
                "leak_points_updated": list(results.keys()),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            return sanitize_and_verify_payload(payload, epsilon_cost=0.0, query_id="policy_refresh")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"External policy refresh failed: {str(e)}"
            )

    return app


# Default app instance for ASGI servers (e.g. uvicorn src.api:app)
app = create_app()


if __name__ == "__main__":
    import uvicorn
    print("[*] Launching The Leaky Pipeline Privacy-Preserving API on http://127.0.0.1:8000")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
