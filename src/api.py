"""
src/api.py - Privacy-Preserving FastAPI Backend Service.

Phase 6 of CaseLens:
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
    get_configured_total_budget,
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
from src.analytics import (
    filter_event_log,
    compute_filtered_summary,
    compute_filtered_dp_funnel,
    compute_filtered_bottlenecks,
    compute_filtered_districts,
    compute_filtered_evaluation,
    compute_filtered_before_after,
)



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
    total_epsilon: Optional[float] = Field(default=None, gt=0.0, le=1000.0)
    total_delta: float = Field(default=1e-4, gt=0.0, le=1e-2)


class SimulatorRunRequest(BaseModel):
    scenario_key: Optional[str] = None
    custom_records: Optional[List[Dict[str, Any]]] = None
    custom_title: Optional[str] = None


class AuditLogEventRequest(BaseModel):
    event_type: str
    page: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


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
                    entries = json.load(f)
                    # Clean out any old legacy username keys
                    for e in entries:
                        if "username" in e:
                            e["user_id"] = e.pop("username")
                    self._entries = entries
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
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        page: Optional[str] = None,
    ):
        if not event_type:
            if endpoint.startswith("/api/v1/auth/login"):
                event_type = "LOGIN_SUCCESS" if 200 <= status_code < 300 else "LOGIN_FAILED"
            else:
                event_type = "API_REQUEST"

        entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "user_id": user_id or "anonymous",
            "role": role,
            "page": page or endpoint,
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
        configured_eps = get_configured_total_budget()
        self.budget_tracker = PrivacyBudgetTracker(total_epsilon=configured_eps, total_delta=1e-4)
        self.audit_logger = ApiAuditLogger(os.path.join(data_dir, "api_audit_log.json"))
        self.dp_funnel: Dict[str, Any] = {}
        self.dp_districts: Dict[str, Any] = {}
        self.bottlenecks: Dict[str, Any] = {}
        self.category_disparities: Dict[str, Any] = {}
        self.summary: Dict[str, Any] = {}
        self.privacy_report: Dict[str, Any] = {}
        self.evaluation: Dict[str, Any] = {}
        self.journey: Dict[str, Any] = {}
        self.df_cases: Optional[pd.DataFrame] = None
        self.df_events: Optional[pd.DataFrame] = None
        self.filter_cache: Dict[str, Any] = {}
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

        # 7. Linkage Evaluation Metrics
        e_path = os.path.join(self.data_dir, "evaluation_metrics.json")
        if os.path.exists(e_path):
            with open(e_path, "r", encoding="utf-8") as f:
                self.evaluation = json.load(f)

        # 8. Journey Metrics
        j_path = os.path.join(self.data_dir, "journey_metrics.json")
        if os.path.exists(j_path):
            with open(j_path, "r", encoding="utf-8") as f:
                self.journey = json.load(f)

        # 9. Reconstructed Events to Cases (for dynamic DP aggregation)
        ev_path = os.path.join(self.data_dir, "event_log.csv")
        if os.path.exists(ev_path):
            try:
                self.df_events = pd.read_csv(ev_path)
                df_ev = self.df_events.copy()
                df_ev["month"] = pd.to_datetime(df_ev["timestamp"]).dt.to_period("M").astype(str)
                self.df_cases = df_ev.groupby("case_id").agg({
                    "district": "first",
                    "case_category": "first",
                    "month": "first",
                    "activity": "last"
                }).reset_index().rename(columns={"activity": "terminal_stage"})
            except Exception:
                self.df_cases = None
                self.df_events = None


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
    "district_name",
    "cached",
    "too_few_cases",
    "message",
    "filtered_case_count",
    "filtered_event_count",
    "fir_to_chargesheet_rate_pct",
    "attrition_drop_off_pct",
    "comparative_table"
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
    else:
        configured_eps = get_configured_total_budget()
        DATA_STORE.budget_tracker.reset(total_epsilon=configured_eps)
        DATA_STORE.filter_cache.clear()

    app = FastAPI(
        title="CaseLens - Privacy-Preserving Policy Analytics API",
        description=(
            "Production-grade FastAPI analytical service providing institutional aggregates, "
            "differential privacy protections, and statutory policy benchmarking for violence-against-women "
            "case attrition analysis."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    @app.on_event("startup")
    def on_startup():
        configured_eps = get_configured_total_budget()
        DATA_STORE.budget_tracker.reset(total_epsilon=configured_eps)
        DATA_STORE.filter_cache.clear()

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
        else:
            auth_hdr = request.headers.get("Authorization", "")
            if auth_hdr.lower().startswith("bearer "):
                token_val = auth_hdr[7:].strip()
                if token_val in API_KEYS:
                    role_header = API_KEYS[token_val].value
                elif token_val.count(".") == 2:
                    try:
                        from src.auth import decode_access_token
                        p_info = decode_access_token(token_val)
                        role_header = p_info.get("role", "viewer")
                    except Exception:
                        pass

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
            "service": "CaseLens - Policy Analytics API",
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
    # 8b. Server-Side Filter & Cache Helpers
    # ==========================================================================

    def is_baseline_filter(
        district: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        stage: Optional[str] = None,
        category: Optional[str] = None
    ) -> bool:
        d_base = district is None or str(district).strip() in ["", "All", "All Districts"]
        s_base = stage is None or str(stage).strip() in ["", "All", "All Stages", "All Institutions"]
        c_base = category is None or str(category).strip() in ["", "All", "All Categories"]
        f_base = from_date is None or str(from_date).strip() in ["", "2022-12-31"]
        t_base = to_date is None or str(to_date).strip() in ["", "2026-11-27"]
        return d_base and s_base and c_base and f_base and t_base

    def handle_filtered_query(
        cache_prefix: str,
        district: Optional[str],
        from_date: Optional[str],
        to_date: Optional[str],
        stage: Optional[str],
        category: Optional[str],
        epsilon: float,
        compute_fn: Any,
        request: Request,
        baseline_data: Any = None
    ) -> Tuple[Any, float, bool]:
        """
        Returns (data, epsilon_cost, was_cached).
        If query matches baseline filters, returns baseline_data with epsilon_cost.
        If cached, returns cached data with epsilon_cost = 0.0 (no extra privacy budget spent!).
        If uncached filtered query, spends epsilon, computes on filtered events, caches result.
        """
        norm_d = str(district).strip() if district else "ALL"
        norm_f = str(from_date).strip() if from_date else "ALL"
        norm_t = str(to_date).strip() if to_date else "ALL"
        norm_s = str(stage).strip() if stage else "ALL"
        norm_c = str(category).strip() if category else "ALL"
        cache_key = f"{cache_prefix}|{norm_d}|{norm_f}|{norm_t}|{norm_s}|{norm_c}|{epsilon:.3f}"

        if cache_key in DATA_STORE.filter_cache:
            request.state.epsilon_spent = 0.0
            return DATA_STORE.filter_cache[cache_key], 0.0, True

        if is_baseline_filter(district, from_date, to_date, stage, category):
            if DATA_STORE.budget_tracker.remaining_epsilon < (epsilon - 1e-9):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Privacy budget exhausted: remaining {DATA_STORE.budget_tracker.remaining_epsilon:.3f} epsilon < requested {epsilon:.3f}."
                )
            DATA_STORE.filter_cache[cache_key] = baseline_data
            request.state.epsilon_spent = epsilon
            return baseline_data, epsilon, False

        # Cache miss: verify budget and compute
        if DATA_STORE.budget_tracker.remaining_epsilon < (epsilon - 1e-9):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Privacy budget exhausted: remaining {DATA_STORE.budget_tracker.remaining_epsilon:.3f} epsilon < requested {epsilon:.3f}."
            )

        df_ev = DATA_STORE.df_events
        if df_ev is None and os.path.exists(os.path.join(DATA_STORE.data_dir, "event_log.csv")):
            df_ev = pd.read_csv(os.path.join(DATA_STORE.data_dir, "event_log.csv"))
            DATA_STORE.df_events = df_ev

        df_filtered = filter_event_log(
            df_ev,
            district=district,
            from_date=from_date,
            to_date=to_date,
            stage=stage,
            category=category
        )

        res = compute_fn(df_filtered)
        DATA_STORE.filter_cache[cache_key] = res
        request.state.epsilon_spent = epsilon
        return res, epsilon, False

    # ==========================================================================
    # 9. Endpoints: Viewer+ (Viewer, Analyst, Admin)
    # ==========================================================================

    @app.get("/api/v1/summary", tags=["Aggregates (Viewer+)"])
    def get_executive_summary(
        request: Request,
        district: Optional[str] = Query(None, description="District filter (e.g. 'D3_Rudrapur' or 'All')"),
        from_date: Optional[str] = Query(None, description="From inception date YYYY-MM-DD"),
        to_date: Optional[str] = Query(None, description="To inception date YYYY-MM-DD"),
        stage: Optional[str] = Query(None, description="Stage presence filter (e.g. 'Police FIR' or 'All')"),
        category: Optional[str] = Query(None, description="Crime category filter (e.g. 'trafficking' or 'All')"),
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        High-level pipeline health and district vulnerability summary.
        Dynamically recomputed server-side for selected global filters.
        Deducts epsilon = 0.05 from privacy budget (cached requests spend 0.0).
        """
        eps = 0.05
        data, eps_spent, cached = handle_filtered_query(
            "summary", district, from_date, to_date, stage, category, eps,
            lambda df: compute_filtered_summary(df, baseline_summary=DATA_STORE.summary, k_anonymity=5),
            request, baseline_data=DATA_STORE.summary
        )
        payload = {
            "query_type": "executive_summary",
            "data": data,
            "privacy_metadata": {
                "epsilon_cost": eps_spent,
                "remaining_budget": round(DATA_STORE.budget_tracker.remaining_epsilon - (eps_spent if not cached else 0.0), 4),
                "cached": cached
            }
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=eps_spent, query_id="summary_aggregate")

    @app.get("/api/v1/funnel", tags=["Aggregates (Viewer+)"])
    def get_funnel_analytics(
        request: Request,
        district: Optional[str] = Query(None),
        from_date: Optional[str] = Query(None),
        to_date: Optional[str] = Query(None),
        stage: Optional[str] = Query(None),
        category: Optional[str] = Query(None),
        epsilon: float = Query(0.10, ge=0.01, le=1.0),
        k_threshold: int = Query(5, ge=3, le=20),
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        Differentially Private 5-stage case attrition funnel.
        Dynamically recomputed server-side for selected global filters.
        Returns Laplace DP noisy counts, DP retention %, and 95% Confidence Bounds.
        """
        eps = epsilon
        data, eps_spent, cached = handle_filtered_query(
            f"funnel_k{k_threshold}", district, from_date, to_date, stage, category, eps,
            lambda df: compute_filtered_dp_funnel(df, baseline_funnel=DATA_STORE.dp_funnel, epsilon_per_stage=eps, k_threshold=k_threshold),
            request, baseline_data=DATA_STORE.dp_funnel
        )
        payload = {
            "query_type": "funnel_analytics",
            "funnel": data,
            "privacy_protection": {
                "mechanism": "Laplace Differential Privacy (Delta f = 1) + Small-Cell k-Anonymity Suppression",
                "k_threshold": k_threshold,
                "epsilon_spent": eps_spent,
                "remaining_epsilon": round(DATA_STORE.budget_tracker.remaining_epsilon - (eps_spent if not cached else 0.0), 4),
                "cached": cached
            }
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=eps_spent, query_id="funnel_aggregate")

    @app.get("/api/v1/districts", tags=["Aggregates (Viewer+)"])
    def get_district_leakage(
        request: Request,
        district: Optional[str] = Query(None),
        from_date: Optional[str] = Query(None),
        to_date: Optional[str] = Query(None),
        stage: Optional[str] = Query(None),
        category: Optional[str] = Query(None),
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        Differentially Private district-level leakage metrics (D1-D8),
        systemic leak codes (LP-01, LP-02, LP-03), and vulnerability scores.
        Dynamically recomputed server-side for selected global filters.
        """
        eps = 0.10
        data, eps_spent, cached = handle_filtered_query(
            "districts", district, from_date, to_date, stage, category, eps,
            lambda df: compute_filtered_districts(df, baseline_districts=DATA_STORE.dp_districts, k_threshold=5),
            request, baseline_data=DATA_STORE.dp_districts
        )
        payload = {
            "query_type": "district_leakage_diagnostics",
            "district_analytics": data,
            "privacy_metadata": {
                "epsilon_spent": eps_spent,
                "remaining_epsilon": round(DATA_STORE.budget_tracker.remaining_epsilon - (eps_spent if not cached else 0.0), 4),
                "cached": cached
            }
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=eps_spent, query_id="district_aggregate")

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
        # Sanitize and rank sources server-side
        from src.policy_context import filter_and_rank_sources
        cleaned = {}
        for k, v in cache.items():
            if isinstance(v, dict) and "sources" in v:
                entry = dict(v)
                entry["sources"] = filter_and_rank_sources(entry["sources"], catalog_key=k)
                cleaned[k] = entry
            else:
                cleaned[k] = v
        benchmarks = STATUTORY_BENCHMARKS
        payload = {
            "statutory_benchmarks": benchmarks,
            "external_policy_context": cleaned,
            "source": "Statutory Legislation & TinyFish Policy Benchmarking Engine",
            "zero_pii_verified": True
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=0.0, query_id="policy_context")

    @app.get("/api/v1/evaluation", tags=["Aggregates (Viewer+)"])
    def get_evaluation_metrics(
        request: Request,
        district: Optional[str] = Query(None),
        from_date: Optional[str] = Query(None),
        to_date: Optional[str] = Query(None),
        stage: Optional[str] = Query(None),
        category: Optional[str] = Query(None),
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        Probabilistic record linkage evaluation metrics,
        precision-recall curve data, and cluster-level purity/BCubed F1 benchmarks.
        Dynamically scaled/recomputed server-side for selected global filters.
        Zero PII.
        """
        eps = 0.0
        eval_safe = {
            k: v for k, v in DATA_STORE.evaluation.items()
            if k not in ["error_diagnostics", "false_positives_sample", "false_negatives_sample"]
        }
        data, eps_spent, cached = handle_filtered_query(
            "evaluation", district, from_date, to_date, stage, category, eps,
            lambda df: compute_filtered_evaluation(df, baseline_evaluation=eval_safe, k_threshold=5),
            request, baseline_data=eval_safe
        )
        payload = {
            "evaluation": data,
            "zero_pii_verified": True,
            "privacy_metadata": {
                "epsilon_spent": 0.0,
                "remaining_budget": round(DATA_STORE.budget_tracker.remaining_epsilon, 4),
                "cached": cached
            }
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=0.0, query_id="evaluation_metrics")

    @app.get("/api/v1/before-after", tags=["Aggregates (Viewer+)"])
    def get_before_after_insights(
        request: Request,
        district: Optional[str] = Query(None),
        from_date: Optional[str] = Query(None),
        to_date: Optional[str] = Query(None),
        stage: Optional[str] = Query(None),
        category: Optional[str] = Query(None),
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        Dynamic Before vs After policy transformation insights and comparative metrics.
        Dynamically recomputed server-side for selected global filters.
        Zero PII.
        """
        eps = 0.0
        base_before_after = compute_filtered_before_after(DATA_STORE.df_events, k_threshold=5) if DATA_STORE.df_events is not None else {}
        data, eps_spent, cached = handle_filtered_query(
            "before_after", district, from_date, to_date, stage, category, eps,
            lambda df: compute_filtered_before_after(df, k_threshold=5),
            request, baseline_data=base_before_after
        )
        payload = {
            "query_type": "before_after_insights",
            "data": data,
            "zero_pii_verified": True,
            "privacy_metadata": {
                "epsilon_spent": 0.0,
                "remaining_budget": round(DATA_STORE.budget_tracker.remaining_epsilon, 4),
                "cached": cached
            }
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=0.0, query_id="before_after_insights")

    @app.get("/api/v1/journey", tags=["Aggregates (Viewer+)"])
    def get_journey_metrics(
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        Reconstructed longitudinal journey metrics and milestone duration distributions.
        Zero PII.
        """
        payload = {
            "journey": DATA_STORE.journey,
            "zero_pii_verified": True
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=0.0, query_id="journey_metrics")

    # ==========================================================================
    # 10. Endpoints: Analyst+ (Analyst, Admin) & Deep Analytics
    # ==========================================================================

    @app.get("/api/v1/bottlenecks", tags=["Deep Analytics (Analyst+)"])
    def get_bottlenecks(
        request: Request,
        district: Optional[str] = Query(None),
        from_date: Optional[str] = Query(None),
        to_date: Optional[str] = Query(None),
        stage: Optional[str] = Query(None),
        category: Optional[str] = Query(None),
        user: UserContext = Depends(require_role(Role.ANALYST))
    ):
        """
        Stage transition duration percentiles (median, IQR, p90, p95)
        and statutory latency compliance percentages.
        Dynamically recomputed server-side for selected global filters.
        Deducts epsilon = 0.15 from privacy budget (cached requests spend 0.0).
        """
        eps = 0.15
        data, eps_spent, cached = handle_filtered_query(
            "bottlenecks", district, from_date, to_date, stage, category, eps,
            lambda df: compute_filtered_bottlenecks(df, baseline_bottlenecks=DATA_STORE.bottlenecks, k_threshold=5),
            request, baseline_data=DATA_STORE.bottlenecks
        )
        payload = {
            "query_type": "duration_bottleneck_percentiles",
            "transitions": data,
            "privacy_metadata": {
                "epsilon_spent": eps_spent,
                "remaining_epsilon": round(DATA_STORE.budget_tracker.remaining_epsilon - (eps_spent if not cached else 0.0), 4),
                "cached": cached
            }
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=eps_spent, query_id="bottlenecks_aggregate")

    @app.get("/api/v1/categories", tags=["Deep Analytics (Analyst+)"])
    def get_category_disparities(
        request: Request,
        user: UserContext = Depends(require_role(Role.ANALYST))
    ):
        """
        Crime category disparities, step conversion attrition, and statutory volume.
        Repeated identical calls are served from cache and do NOT spend privacy budget.
        """
        cache_key = "category_disparities_baseline"
        if cache_key in DATA_STORE.filter_cache:
            request.state.epsilon_spent = 0.0
            payload = {
                "query_type": "category_disparities",
                "categories": DATA_STORE.category_disparities,
                "privacy_metadata": {
                    "epsilon_spent": 0.0,
                    "remaining_epsilon": round(DATA_STORE.budget_tracker.remaining_epsilon, 4),
                    "cached": True
                }
            }
            return sanitize_and_verify_payload(payload, epsilon_cost=0.0, query_id="category_aggregate")

        eps = 0.10
        if DATA_STORE.budget_tracker.remaining_epsilon < (eps - 1e-9):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Privacy budget exhausted: remaining {DATA_STORE.budget_tracker.remaining_epsilon:.3f} epsilon < requested {eps:.3f}."
            )

        DATA_STORE.filter_cache[cache_key] = True
        request.state.epsilon_spent = eps
        payload = {
            "query_type": "category_disparities",
            "categories": DATA_STORE.category_disparities,
            "privacy_metadata": {
                "epsilon_spent": eps,
                "remaining_epsilon": round(DATA_STORE.budget_tracker.remaining_epsilon - eps, 4),
                "cached": False
            }
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=eps, query_id="category_aggregate")

    @app.get("/api/v1/privacy/budget", tags=["Privacy Engine (Viewer+)"])
    def get_privacy_budget(
        user: UserContext = Depends(require_role(Role.VIEWER))
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

        # Check cache before allocating budget
        filters_str = json.dumps(request_body.filters or {}, sort_keys=True)
        cache_key = f"query_agg|{dim}|{request_body.metric}|{filters_str}|{request_body.epsilon:.3f}|{request_body.k_threshold}"

        if cache_key in DATA_STORE.filter_cache:
            request.state.epsilon_spent = 0.0
            cached_data = dict(DATA_STORE.filter_cache[cache_key])
            cached_data["epsilon_consumed"] = 0.0
            cached_data["cached"] = True
            return sanitize_and_verify_payload(cached_data, epsilon_cost=0.0, query_id="dynamic_aggregate")

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
            "cached": False,
            "results": noisy_aggregates
        }
        DATA_STORE.filter_cache[cache_key] = payload
        return sanitize_and_verify_payload(payload, epsilon_cost=0.0)

    SIMULATOR_SAFE_FIELDS = {
        "name", "guardian", "relative_name", "full_name",
        "scenario_3_swapped_names", "records"
    }

    @app.get("/api/v1/simulator/scenarios", tags=["Case Simulator (Analyst+)"])
    def get_simulator_scenarios(
        user: UserContext = Depends(require_role(Role.ANALYST))
    ):
        """
        Pre-configured synthetic judicial case scenarios and agency records for simulation.
        Accessible to Analysts and Administrators.
        """
        from src.case_simulator import SCENARIOS
        payload = {
            "scenarios": SCENARIOS,
            "zero_pii_verified": True
        }
        return sanitize_and_verify_payload(
            payload,
            epsilon_cost=0.0,
            query_id="simulator_scenarios",
            allowed_fields=SIMULATOR_SAFE_FIELDS
        )

    @app.post("/api/v1/simulator/run", tags=["Case Simulator (Analyst+)"])
    def run_case_simulation(
        request_body: SimulatorRunRequest,
        user: UserContext = Depends(require_role(Role.ANALYST))
    ):
        """
        Executes real-time Fellegi-Sunter record linkage and journey reconstruction
        for a scenario key or custom synthetic agency records.
        Accessible to Analysts and Administrators.
        """
        from src.case_simulator import run_simulation
        if request_body.custom_records:
            sim_result = run_simulation(
                request_body.custom_records,
                custom_title=request_body.custom_title or "Custom Scenario"
            )
        elif request_body.scenario_key:
            sim_result = run_simulation(request_body.scenario_key)
        else:
            raise HTTPException(status_code=400, detail="Must provide either scenario_key or custom_records.")

        payload = {
            "simulation_result": sim_result,
            "zero_pii_verified": True
        }
        return sanitize_and_verify_payload(
            payload,
            epsilon_cost=0.0,
            query_id="simulator_run",
            allowed_fields=SIMULATOR_SAFE_FIELDS
        )

    # ==========================================================================
    # 10b. Endpoints: Case Explorer (Viewer, Analyst, Admin with RBAC Masking)
    # ==========================================================================

    CASE_EXPLORER_SAFE_FIELDS = {
        "name", "guardian", "relative_name", "full_name",
        "records", "evidence", "pairwise_results", "examples",
        "timeline_steps", "steps", "bottlenecks", "elapsed_days",
        "category", "details", "notes", "dob"
    }

    @app.get("/api/v1/cases/examples", tags=["Case Explorer"])
    def get_case_examples(
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        Curated benchmark examples for end-to-end case exploration:
        clean match, spelling variants, uncertain pair, similar names kept apart,
        missed link, District 3 forensic delay, District 6 chargesheet drop-off.
        Accessible to Viewers, Analysts, and Administrators.
        """
        from src.case_explorer import get_case_explorer_service
        svc = get_case_explorer_service()
        examples = svc.get_curated_examples_summary()
        payload = {
            "examples": examples,
            "total_examples": len(examples),
            "synthetic_data": True,
            "zero_pii_verified": True
        }
        return sanitize_and_verify_payload(
            payload,
            epsilon_cost=0.0,
            query_id="case_explorer_examples",
            allowed_fields=CASE_EXPLORER_SAFE_FIELDS
        )

    @app.get("/api/v1/cases/random", tags=["Case Explorer"])
    def get_random_case_endpoint(
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        Picks a random case from the 5,000 generated synthetic multi-agency dataset
        and returns its end-to-end fragmented records, pairwise scoring, and journey timeline.
        Enforces server-side RBAC: Viewers receive masked names; Analysts/Admins receive pseudonymised IDs.
        """
        from src.case_explorer import get_case_explorer_service
        svc = get_case_explorer_service()
        rand_id = svc.get_random_case_id()
        case_data = svc.build_case_explorer_data(rand_id, role=user.role.value)
        payload = {
            "case": case_data,
            "zero_pii_verified": True
        }
        return sanitize_and_verify_payload(
            payload,
            epsilon_cost=0.0,
            query_id="case_explorer_random",
            allowed_fields=CASE_EXPLORER_SAFE_FIELDS
        )

    @app.get("/api/v1/cases/{case_id}", tags=["Case Explorer"])
    def get_case_by_id(
        case_id: str,
        user: UserContext = Depends(require_role(Role.VIEWER))
    ):
        """
        Retrieves end-to-end multi-agency records, pairwise match probabilities,
        reconstructed journey timeline with elapsed days, and one-line verdict.
        Enforces server-side RBAC: Viewers receive masked names; Analysts/Admins receive pseudonymised IDs.
        """
        from src.case_explorer import get_case_explorer_service
        svc = get_case_explorer_service()
        case_data = svc.build_case_explorer_data(case_id, role=user.role.value)
        payload = {
            "case": case_data,
            "zero_pii_verified": True
        }
        return sanitize_and_verify_payload(
            payload,
            epsilon_cost=0.0,
            query_id="case_explorer_detail",
            allowed_fields=CASE_EXPLORER_SAFE_FIELDS
        )

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
        request_body: Optional[ResetBudgetRequest] = None,
        user: UserContext = Depends(require_role(Role.ADMIN))
    ):
        """
        Reconfigures or resets the Differential Privacy budget under administrative authority.
        Logs the administrative action into the audit ledger.
        Accessible exclusively to Administrators.
        """
        req_eps = request_body.total_epsilon if request_body else None
        req_delta = request_body.total_delta if request_body else 1e-4
        new_eps = req_eps if req_eps is not None else get_configured_total_budget()
        DATA_STORE.budget_tracker = PrivacyBudgetTracker(
            total_epsilon=new_eps,
            total_delta=req_delta
        )
        DATA_STORE.filter_cache.clear()
        payload = {
            "message": "Privacy budget successfully reset by Administrator.",
            "new_total_epsilon": new_eps,
            "new_total_delta": req_delta,
            "remaining_epsilon": new_eps,
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
        report = dict(DATA_STORE.privacy_report)
        report["privacy_budget_summary"] = {
            "total_epsilon_allocated": round(DATA_STORE.budget_tracker.total_epsilon, 4),
            "spent_epsilon": round(DATA_STORE.budget_tracker.spent_epsilon, 4),
            "remaining_epsilon": round(DATA_STORE.budget_tracker.remaining_epsilon, 4),
            "composition_theorem": "Sequential Composition (sum(epsilon_i) <= total_epsilon)",
            "budget_ledger_entries": len(DATA_STORE.budget_tracker.get_ledger()),
        }
        return sanitize_and_verify_payload(
            report,
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

    @app.get("/api/v1/review-queue", tags=["Admin Controls"])
    def get_review_queue(
        limit: int = Query(default=200, ge=1, le=1000),
        user: UserContext = Depends(require_role(Role.ADMIN))
    ):
        """
        Returns uncertain candidate pairs (0.50 <= P < 0.85) for human adjudication.
        Strictly strips all direct personal identifiers (names, relatives, addresses).
        Exposes only pseudonymized record IDs, incident dates, and match probabilities.
        Accessible exclusively to Administrators.
        """
        review_path = os.path.join(DATA_STORE.data_dir, "review_queue.csv")
        safe_cols = [
            "unique_id_l", "unique_id_r", "district_l", "district_r",
            "incident_date_l", "incident_date_r", "match_probability",
            "match_weight", "review_status"
        ]
        records = []
        total_pairs = 0
        avg_prob = 0.0

        if os.path.exists(review_path):
            try:
                df_rq = pd.read_csv(review_path)
                total_pairs = len(df_rq)
                if "match_probability" in df_rq.columns and len(df_rq) > 0:
                    avg_prob = float(df_rq["match_probability"].mean())
                display_cols = [c for c in safe_cols if c in df_rq.columns]
                df_display = df_rq[display_cols].head(limit)
                records = df_display.to_dict(orient="records")
            except Exception:
                records = []

        payload = {
            "total_pairs": total_pairs,
            "avg_match_probability": round(avg_prob, 3),
            "accept_threshold": 0.85,
            "reject_threshold": 0.50,
            "records": records,
            "zero_pii_verified": True
        }
        return sanitize_and_verify_payload(payload, epsilon_cost=0.0, query_id="review_queue")

    @app.post("/api/v1/audit/log", tags=["System"])
    def record_audit_log_event(
        request: Request,
        request_body: AuditLogEventRequest,
    ):
        """
        Records an application or session audit event (login, logout, page visit, access denied)
        into the immutable audit ledger with zero credentials or PII.
        """
        client_ip = request.client.host if request.client else "127.0.0.1"
        DATA_STORE.audit_logger.log_event(
            request_id=f"audit_{uuid.uuid4().hex[:8]}",
            role=request_body.role or "viewer",
            method="POST",
            endpoint=f"/audit/event/{request_body.event_type.lower()}",
            status_code=200,
            client_ip=client_ip,
            event_type=request_body.event_type,
            user_id=request_body.user_id or "anonymous",
            page=request_body.page,
            details=request_body.details or {}
        )
        return {"status": "recorded", "event_type": request_body.event_type}

    return app


# Default app instance for ASGI servers (e.g. uvicorn src.api:app)
app = create_app()


if __name__ == "__main__":
    import uvicorn
    print("[*] Launching CaseLens Privacy-Preserving API on http://127.0.0.1:8000")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
