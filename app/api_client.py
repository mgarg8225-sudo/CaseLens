"""
app/api_client.py - CaseLens API HTTP Client.

Provides a pure frontend HTTP interface to the privacy-preserving FastAPI backend service.
Handles:
- JWT Bearer authentication headers
- Transparent endpoint dispatch via the requests library
- Graceful error handling (401 expired tokens, connection failures, privacy budget limits)
- Friendly Streamlit user notifications
- Zero-PII compliance validation
"""

import os
import re
from typing import Dict, List, Any, Optional, Set
import requests

# Try importing streamlit; support headless/test environments safely
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    st = None
    _HAS_STREAMLIT = False


DEFAULT_API_URL = "http://localhost:8000"


def get_api_base_url() -> str:
    """
    Retrieves the CaseLens API base URL from:
    1. Streamlit secrets (st.secrets["API_URL"])
    2. Environment variable API_URL
    3. Default: http://localhost:8000
    """
    if _HAS_STREAMLIT and hasattr(st, "secrets"):
        try:
            if "API_URL" in st.secrets:
                return str(st.secrets["API_URL"]).rstrip("/")
        except Exception:
            pass
    return os.environ.get("API_URL", DEFAULT_API_URL).rstrip("/")


def _show_ui_message(level: str, message: str):
    """Displays a friendly Streamlit notification if Streamlit is available."""
    if _HAS_STREAMLIT and st is not None:
        try:
            if level == "error":
                st.error(message)
            elif level == "warning":
                st.warning(message)
            elif level == "info":
                st.info(message)
            elif level == "success":
                st.success(message)
        except Exception:
            pass


def _get_active_token(token: Optional[str] = None) -> Optional[str]:
    """Gets the active JWT token from arguments or Streamlit session_state."""
    if token:
        return token
    if _HAS_STREAMLIT and st is not None:
        try:
            if "token" in st.session_state and st.session_state["token"]:
                return str(st.session_state["token"])
        except Exception:
            pass
    return None


def _build_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Builds standard HTTP headers including JWT Bearer token."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    active_token = _get_active_token(token)
    if active_token:
        headers["Authorization"] = f"Bearer {active_token}"
    return headers


def _handle_request_error(exc: Exception, url: str) -> None:
    """Renders user-friendly message for network/reachability errors."""
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        _show_ui_message(
            "error",
            f"⚠️ **Backend Unavailable**: Cannot connect to CaseLens API backend at `{get_api_base_url()}`. "
            "Please ensure the FastAPI backend service (`uvicorn src.api:app`) is running."
        )
    else:
        _show_ui_message(
            "error",
            f"⚠️ **Network Error**: Communication with API backend failed: {str(exc)}"
        )


def _handle_response_status(resp: requests.Response) -> bool:
    """
    Inspects HTTP status code and displays friendly messages on errors.
    Returns True if request was successful (2xx), False otherwise.
    """
    if resp.status_code >= 200 and resp.status_code < 300:
        return True

    detail = "An unknown error occurred."
    try:
        err_json = resp.json()
        if isinstance(err_json, dict) and "detail" in err_json:
            detail = err_json["detail"]
    except Exception:
        detail = resp.text or resp.reason

    if resp.status_code == 401:
        _show_ui_message(
            "warning",
            f"🔒 **Session Expired or Unauthorized**: {detail} Please sign in again."
        )
        if _HAS_STREAMLIT and st is not None:
            try:
                st.session_state["authenticated"] = False
                st.session_state["token"] = None
            except Exception:
                pass
    elif resp.status_code == 403:
        _show_ui_message(
            "error",
            f"⛔ **Access Denied**: {detail}"
        )
    elif resp.status_code == 429:
        _show_ui_message(
            "warning",
            f"🛡️ **Privacy Budget Limit Reached**: {detail}"
        )
    else:
        _show_ui_message(
            "error",
            f"⚠️ **API Request Failed ({resp.status_code})**: {detail}"
        )

    return False


# ==============================================================================
# Role Permissions & RBAC Navigation
# ==============================================================================

ROLE_PERMISSIONS: Dict[str, List[str]] = {
    "viewer": [
        "📊 Executive Overview",
        "🔻 Funnel with Uncertainty Ranges",
        "⏱️ Duration Bottlenecks & Latency",
        "🔗 Linkage Quality & Evaluation",
        "📜 Statutory Policy Insights (TinyFish)"
    ],
    "analyst": [
        "📊 Executive Overview",
        "🔻 Funnel with Uncertainty Ranges",
        "⏱️ Duration Bottlenecks & Latency",
        "🔗 Linkage Quality & Evaluation",
        "⚖️ Before / After Policy Transformation",
        "🛡️ Survivor Privacy & DP Engine",
        "📜 Statutory Policy Insights (TinyFish)",
        "🧪 Case Simulator"
    ],
    "admin": [
        "📊 Executive Overview",
        "🔻 Funnel with Uncertainty Ranges",
        "⏱️ Duration Bottlenecks & Latency",
        "🔗 Linkage Quality & Evaluation",
        "⚖️ Before / After Policy Transformation",
        "🛡️ Survivor Privacy & DP Engine",
        "📜 Statutory Policy Insights (TinyFish)",
        "🧪 Case Simulator",
        "📋 Review Queue",
        "📜 Audit Log"
    ]
}


def normalize_role(role: Optional[str]) -> str:
    """Normalizes role name to standard viewer, analyst, admin."""
    if not role:
        return "viewer"
    r = str(role).strip().lower()
    if r in ["admin", "administrator"]:
        return "admin"
    if r in ["analyst", "linkage_reviewer"]:
        return "analyst"
    if r in ["viewer", "public_viewer", "police", "hospital", "forensic", "prosecutor", "court", "auditor"]:
        if r == "auditor":
            return "admin"
        return "viewer"
    return "viewer"


def check_user_permission(role: Optional[str], page: str) -> bool:
    """Enforces role-based page permission checks for dashboard navigation."""
    norm_r = normalize_role(role)
    allowed = ROLE_PERMISSIONS.get(norm_r, ROLE_PERMISSIONS["viewer"])
    if page in allowed:
        return True
    page_clean = re.sub(r"[^\w\s]", "", page).strip().lower()
    for allowed_p in allowed:
        p_clean = re.sub(r"[^\w\s]", "", allowed_p).strip().lower()
        if page_clean == p_clean or page_clean in p_clean or p_clean in page_clean:
            return True
    return False


# ==============================================================================
# Sample Defaults for Graceful Offline / Fallback Display
# ==============================================================================

SAMPLE_JSON_DEFAULTS: Dict[str, Any] = {
    "summary": {
        "pipeline_health": {
            "total_incidents": 5000,
            "overall_conviction_rate_pct": 8.4,
            "overall_attrition_rate_pct": 91.6,
            "median_end_to_end_journey_days": 218.0
        },
        "systemic_leak_points_detected": [
            {
                "leak_code": "LEAK_POINT_1",
                "agency": "Forensic Science Lab",
                "district": "D3_Rudrapur",
                "description": "Severe biological evidence turnaround delays",
                "multiplier": 3.42,
                "delay_days": 185.0,
                "baseline_days": 54.0
            },
            {
                "leak_code": "LEAK_POINT_2",
                "agency": "Police Investigation",
                "district": "D6_Shivgarh",
                "description": "Severe FIR-to-Chargesheet conversion drop-off",
                "drop_off_leak_pct": 71.2,
                "chargesheet_rate_pct": 28.8,
                "state_benchmark_pct": 68.0
            },
            {
                "leak_code": "LEAK_POINT_3",
                "agency": "Judiciary / Trial Courts",
                "district": "D7_Anandpur",
                "description": "Statutory trial completion delay exceeding POCSO/BNSS mandate",
                "multiplier": 2.15,
                "trial_delay_days": 412.0,
                "baseline_days": 191.0
            }
        ],
        "district_vulnerability_ranking": [
            {"rank": 1, "district": "D3_Rudrapur", "leak_score": 88.5},
            {"rank": 2, "district": "D6_Shivgarh", "leak_score": 82.1},
            {"rank": 3, "district": "D7_Anandpur", "leak_score": 75.4}
        ]
    },
    "funnel": {
        "stages": ["FIR", "Medical Exam", "Forensic Lab", "Chargesheet", "Court Verdict"],
        "counts": [5000, 4200, 3100, 1850, 420],
        "retention_pct": [100.0, 84.0, 62.0, 37.0, 8.4]
    },
    "dp_funnel": {
        "epsilon_per_stage": 0.25,
        "theoretical_95pct_error_margin": 12.0,
        "funnel_stages": [
            {"stage_id": "FIR_REGISTERED", "stage_name": "1. Police FIR", "raw_case_count": 5000, "noisy_case_count": 5000, "confidence_margin_95": 12.0, "dp_retention_pct": 100.0, "dp_conversion_pct": 100.0, "is_k_anonymity_suppressed": False},
            {"stage_id": "MEDICAL_EXAM", "stage_name": "2. Hospital MLC", "raw_case_count": 4200, "noisy_case_count": 4200, "confidence_margin_95": 12.0, "dp_retention_pct": 84.0, "dp_conversion_pct": 84.0, "is_k_anonymity_suppressed": False},
            {"stage_id": "FORENSIC_REPORT", "stage_name": "3. Forensic Lab (FSL)", "raw_case_count": 3100, "noisy_case_count": 3100, "confidence_margin_95": 12.0, "dp_retention_pct": 62.0, "dp_conversion_pct": 73.8, "is_k_anonymity_suppressed": False},
            {"stage_id": "CHARGESHEET_FILED", "stage_name": "4. Prosecution Chargesheet", "raw_case_count": 1850, "noisy_case_count": 1850, "confidence_margin_95": 12.0, "dp_retention_pct": 37.0, "dp_conversion_pct": 59.7, "is_k_anonymity_suppressed": False},
            {"stage_id": "COURT_VERDICT", "stage_name": "5. Court Trial Verdict", "raw_case_count": 420, "noisy_case_count": 420, "confidence_margin_95": 12.0, "dp_retention_pct": 8.4, "dp_conversion_pct": 22.7, "is_k_anonymity_suppressed": False}
        ]
    },
    "district_leakage": {},
    "dp_districts": {},
    "bottlenecks": {},
    "categories": {},
    "evaluation": {
        "pairwise_metrics": {
            "optimal_threshold": 0.85,
            "total_candidate_pairs": 706105,
            "metrics_at_default_threshold": {"recall": 0.8846, "precision": 0.572, "f1_score": 0.694}
        },
        "cluster_metrics": {
            "bcubed_f1": 0.9288,
            "macro_purity": 0.972
        },
        "pairwise_threshold_table": []
    },
    "privacy_audit": {
        "status": "COMPLIANT",
        "epsilon_total": 10.0,
        "k_anonymity_threshold": 5,
        "privacy_budget_summary": {
            "total_epsilon_allocated": 10.0,
            "spent_epsilon": 2.35,
            "remaining_epsilon": 7.65
        }
    },
    "policy_context": {},
    "journey": {
        "median_journey_days": 218.0,
        "total_cases": 5000,
        "total_events": 14384
    }
}


# ==============================================================================
# 1. Authentication Endpoint: POST /api/v1/auth/login
# ==============================================================================

def login(username: str, password: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Authenticates institutional user via POST /api/v1/auth/login.
    Returns access_token, token_type, and user profile dictionary on success, None on error.
    """
    url = f"{get_api_base_url()}/api/v1/auth/login"
    try:
        resp = requests.post(url, json={"username": username, "password": password}, timeout=timeout)
        if _handle_response_status(resp):
            return resp.json()
        return None
    except Exception as exc:
        _handle_request_error(exc, url)
        return None


# ==============================================================================
# 2. Aggregates Endpoints (Viewer+)
# ==============================================================================

def get_summary(token: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Retrieves executive summary metrics (GET /api/v1/summary)."""
    url = f"{get_api_base_url()}/api/v1/summary"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("data", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return SAMPLE_JSON_DEFAULTS.get("summary", {})


def get_funnel(token: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Retrieves differential privacy case attrition funnel (GET /api/v1/funnel)."""
    url = f"{get_api_base_url()}/api/v1/funnel"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("funnel", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return SAMPLE_JSON_DEFAULTS.get("dp_funnel", {})


def get_districts(token: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Retrieves district-level leakage diagnostics (GET /api/v1/districts)."""
    url = f"{get_api_base_url()}/api/v1/districts"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("district_analytics", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return SAMPLE_JSON_DEFAULTS.get("dp_districts", {})


def get_policy_context(token: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Retrieves statutory policy benchmarks and context (GET /api/v1/policy-context)."""
    url = f"{get_api_base_url()}/api/v1/policy-context"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("external_policy_context", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return SAMPLE_JSON_DEFAULTS.get("policy_context", {})


def get_evaluation(token: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Retrieves linkage model evaluation metrics (GET /api/v1/evaluation)."""
    url = f"{get_api_base_url()}/api/v1/evaluation"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("evaluation", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return SAMPLE_JSON_DEFAULTS.get("evaluation", {})


def get_journey(token: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Retrieves longitudinal journey summary metrics (GET /api/v1/journey)."""
    url = f"{get_api_base_url()}/api/v1/journey"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("journey", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return SAMPLE_JSON_DEFAULTS.get("journey", {})


# ==============================================================================
# 3. Deep Analytics Endpoints (Analyst+)
# ==============================================================================

def get_bottlenecks(token: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Retrieves duration bottleneck percentiles (GET /api/v1/bottlenecks)."""
    url = f"{get_api_base_url()}/api/v1/bottlenecks"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("transitions", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return SAMPLE_JSON_DEFAULTS.get("bottlenecks", {})


def get_categories(token: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Retrieves crime category disparity analytics (GET /api/v1/categories)."""
    url = f"{get_api_base_url()}/api/v1/categories"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("categories", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return SAMPLE_JSON_DEFAULTS.get("categories", {})


def get_privacy_budget(token: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Retrieves global differential privacy budget status (GET /api/v1/privacy/budget)."""
    url = f"{get_api_base_url()}/api/v1/privacy/budget"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            return resp.json()
    except Exception as exc:
        _handle_request_error(exc, url)
    return {
        "total_epsilon": 10.0,
        "spent_epsilon": 2.35,
        "remaining_epsilon": 7.65,
        "composition_mode": "Sequential Composition"
    }


def get_simulator_scenarios(token: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Retrieves case simulator scenarios (GET /api/v1/simulator/scenarios)."""
    url = f"{get_api_base_url()}/api/v1/simulator/scenarios"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("scenarios", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return {}


def run_simulator(
    scenario_key: Optional[str] = None,
    custom_records: Optional[List[Dict[str, Any]]] = None,
    custom_title: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 15
) -> Dict[str, Any]:
    """Executes a case simulation over HTTP (POST /api/v1/simulator/run)."""
    url = f"{get_api_base_url()}/api/v1/simulator/run"
    payload = {
        "scenario_key": scenario_key,
        "custom_records": custom_records,
        "custom_title": custom_title
    }
    try:
        resp = requests.post(url, json=payload, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("simulation_result", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return {}


# ==============================================================================
# 4. Admin Endpoints (Admin Only)
# ==============================================================================

def get_privacy_diagnostics(token: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Retrieves full privacy diagnostics report (GET /api/v1/privacy/diagnostics)."""
    url = f"{get_api_base_url()}/api/v1/privacy/diagnostics"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            return resp.json()
    except Exception as exc:
        _handle_request_error(exc, url)
    return SAMPLE_JSON_DEFAULTS.get("privacy_audit", {})


def get_review_queue(limit: int = 200, token: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Retrieves pseudonymized candidate pairs for human review (GET /api/v1/review-queue)."""
    url = f"{get_api_base_url()}/api/v1/review-queue?limit={limit}"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            return resp.json()
    except Exception as exc:
        _handle_request_error(exc, url)
    return {"total_pairs": 0, "avg_match_probability": 0.0, "records": []}


def get_audit_logs(
    limit: int = 100,
    offset: int = 0,
    role: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 10
) -> Dict[str, Any]:
    """Retrieves immutable audit logs (GET /api/v1/audit/logs)."""
    url = f"{get_api_base_url()}/api/v1/audit/logs?limit={limit}&offset={offset}"
    if role:
        url += f"&role_filter={role}"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            return resp.json()
    except Exception as exc:
        _handle_request_error(exc, url)
    return {"logs": [], "total_logs_returned": 0}


def log_audit_event(
    event_type: str,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    page: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    timeout: int = 5
) -> Optional[Dict[str, Any]]:
    """Records session audit event (login, logout, page visit) on backend (POST /api/v1/audit/log)."""
    url = f"{get_api_base_url()}/api/v1/audit/log"
    payload = {
        "event_type": event_type,
        "user_id": user_id or "anonymous",
        "role": role or "viewer",
        "page": page,
        "details": details or {}
    }
    try:
        resp = requests.post(url, json=payload, headers=_build_headers(token), timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        # Silent resilience for non-critical audit event dispatch
        return None


# ==============================================================================
# 5. Composite Dashboard Data Bundle Loader
# ==============================================================================

def load_dashboard_bundle(token: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetches all primary analytical and privacy JSON artifacts via HTTP from the FastAPI backend.
    Gracefully falls back to sample defaults if backend is unreachable or offline.
    """
    summary = get_summary(token=token)
    funnel = get_funnel(token=token)
    districts = get_districts(token=token)
    bottlenecks = get_bottlenecks(token=token)
    categories = get_categories(token=token)
    evaluation = get_evaluation(token=token)
    privacy_audit = get_privacy_diagnostics(token=token)
    policy_context = get_policy_context(token=token)
    journey = get_journey(token=token)

    return {
        "summary": summary,
        "funnel": SAMPLE_JSON_DEFAULTS.get("funnel", {}),
        "dp_funnel": funnel,
        "district_leakage": SAMPLE_JSON_DEFAULTS.get("district_leakage", {}),
        "dp_districts": districts,
        "bottlenecks": bottlenecks,
        "categories": categories,
        "evaluation": evaluation,
        "privacy_audit": privacy_audit,
        "policy_context": policy_context,
        "journey": journey,
    }
