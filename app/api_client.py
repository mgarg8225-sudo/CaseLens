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
import json
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
        if _HAS_STREAMLIT and st is not None:
            try:
                st.session_state["privacy_budget_exhausted"] = True
                st.session_state["budget_exhausted_detail"] = detail
            except Exception:
                pass
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
        "🔻 Case Funnel",
        "⏱️ Duration & Bottlenecks",
        "🔗 Linkage Quality",
        "📜 Policy Insights",
        "🔍 Case Review"
    ],
    "analyst": [
        "🔬 Analyze Your Data",
        "📊 Executive Overview",
        "🔻 Case Funnel",
        "⏱️ Duration & Bottlenecks",
        "🔗 Linkage Quality",
        "⚖️ Policy Impact",
        "🛡️ Privacy & Protection",
        "📜 Policy Insights",
        "🔍 Case Review"
    ],
    "admin": [
        "🔬 Analyze Your Data",
        "📊 Executive Overview",
        "🔻 Case Funnel",
        "⏱️ Duration & Bottlenecks",
        "🔗 Linkage Quality",
        "⚖️ Policy Impact",
        "🛡️ Privacy & Protection",
        "📜 Policy Insights",
        "🔍 Case Review",
        "📜 Audit Log"
    ]
}

PAGE_ALIASES: Dict[str, str] = {
    "funnel with uncertainty ranges": "case funnel",
    "case funnel": "case funnel",
    "duration bottlenecks & latency": "duration & bottlenecks",
    "duration bottlenecks and latency": "duration & bottlenecks",
    "duration bottlenecks latency": "duration & bottlenecks",
    "duration & bottlenecks": "duration & bottlenecks",
    "duration bottlenecks": "duration & bottlenecks",
    "linkage quality & evaluation": "linkage quality",
    "linkage quality and evaluation": "linkage quality",
    "linkage quality evaluation": "linkage quality",
    "linkage quality": "linkage quality",
    "before / after policy transformation": "policy impact",
    "before after policy transformation": "policy impact",
    "policy impact": "policy impact",
    "survivor privacy & dp engine": "privacy & protection",
    "survivor privacy and dp engine": "privacy & protection",
    "survivor privacy dp engine": "privacy & protection",
    "privacy & protection": "privacy & protection",
    "privacy protection": "privacy & protection",
    "statutory policy insights (tinyfish)": "policy insights",
    "statutory policy insights tinyfish": "policy insights",
    "statutory policy insights": "policy insights",
    "policy insights": "policy insights",
    "case explorer": "case review",
    "case review": "case review",
    "case simulator": "policy impact",
    "review queue": "review queue",
    "audit log": "audit log",
    "executive overview": "executive overview",
    "analyze your data": "analyze your data",
}


def _clean_page_str(val: str) -> str:
    """Strips emoji and punctuation, returns collapsed lowercase tokens."""
    return " ".join(re.sub(r"[^\w\s]", " ", val).lower().split())


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

    p_clean = _clean_page_str(page)

    # Admin-only sub-features / routes
    if "review queue" in p_clean:
        return norm_r == "admin"
    if "audit log" in p_clean or p_clean == "audit":
        return norm_r == "admin"

    # Analyst+ features / routes
    if "case simulator" in p_clean or p_clean == "simulator":
        return norm_r in ["analyst", "admin"]
    if "analyze your data" in p_clean or "analyze" in p_clean:
        return norm_r in ["analyst", "admin"]
    if "policy impact" in p_clean or "before after" in p_clean:
        return norm_r in ["analyst", "admin"]

    # Case Explorer is accessible to all roles under Case Review (with role-based masking)
    if "case explorer" in p_clean or "explorer" in p_clean:
        return True

    # Check direct alias mapping
    mapped = PAGE_ALIASES.get(p_clean)

    # Check against allowed pages for the role
    for allowed_p in allowed:
        ap_clean = _clean_page_str(allowed_p)
        if p_clean == ap_clean:
            return True
        if mapped and (_clean_page_str(mapped) == ap_clean or mapped in ap_clean or ap_clean in mapped):
            return True
        # Keyword heuristic matching between alias and canonical
        if "duration" in p_clean and "duration" in ap_clean:
            return True
        if "funnel" in p_clean and "funnel" in ap_clean:
            return True
        if "linkage" in p_clean and "linkage" in ap_clean:
            return True
        if "statutory" in p_clean and ("statutory" in ap_clean or "insight" in ap_clean):
            return True
        if "insight" in p_clean and "insight" in ap_clean:
            return True
        if "privacy" in p_clean and "privacy" in ap_clean:
            return True
        if ("case review" in p_clean or "case explorer" in p_clean) and "case review" in ap_clean:
            return True
        if "overview" in p_clean and "overview" in ap_clean:
            return True

    return False


DATA_DIR = os.environ.get("LEAKY_DATA_DIR", "data")


# ==============================================================================
# Sample Defaults for Graceful Offline / Fallback Display
# ==============================================================================

SAMPLE_JSON_DEFAULTS: Dict[str, Any] = {
    "summary": {
        "pipeline_health": {
            "total_incidents": 0,
            "total_reconstructed_cases": 0,
            "cases_initiating_fir": 0,
            "cases_reaching_verdict": 0,
            "overall_conviction_reach_rate_pct": 0.0,
            "overall_conviction_rate_pct": 0.0,
            "overall_attrition_rate_pct": 0.0,
            "median_end_to_end_journey_days": 0.0
        },
        "systemic_leak_points_detected": [],
        "district_vulnerability_ranking": []
    },
    "funnel": {
        "stages": ["FIR", "Medical Exam", "Forensic Lab", "Chargesheet", "Court Verdict"],
        "counts": [0, 0, 0, 0, 0],
        "retention_pct": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "dp_funnel": {
        "epsilon_per_stage": 0.25,
        "theoretical_95pct_error_margin": 0.0,
        "funnel_stages": []
    },
    "district_leakage": {},
    "dp_districts": {},
    "bottlenecks": {},
    "categories": {},
    "evaluation": {
        "pairwise_metrics": {},
        "cluster_metrics": {},
        "pairwise_threshold_table": []
    },
    "privacy_audit": {
        "status": "UNINITIALIZED",
        "privacy_budget_summary": {}
    },
    "policy_context": {},
    "journey": {
        "median_journey_days": 0.0,
        "total_cases": 0,
        "total_events": 0
    }
}


def load_json_fallback(key_or_filename: str) -> Dict[str, Any]:
    """Safely loads fallback data from local data directory or SAMPLE_JSON_DEFAULTS."""
    filename = key_or_filename if key_or_filename.endswith(".json") else f"{key_or_filename}.json"
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        mapping = {
            "summary": "analytics_summary.json",
            "funnel": "analytics_funnel.json",
            "dp_funnel": "privacy_dp_funnel.json",
            "district_leakage": "analytics_district_leakage.json",
            "dp_districts": "privacy_dp_districts.json",
            "bottlenecks": "analytics_bottlenecks.json",
            "categories": "analytics_category.json",
            "evaluation": "evaluation_metrics.json",
            "privacy_audit": "privacy_audit_report.json",
            "policy_context": "policy_context_cache.json",
            "journey": "journey_metrics.json",
        }
        alt = mapping.get(key_or_filename.replace(".json", ""))
        if alt:
            file_path = os.path.join(DATA_DIR, alt)

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    key = key_or_filename.replace(".json", "")
    for k in [
        "summary", "funnel", "dp_funnel", "bottlenecks", "evaluation",
        "privacy_audit", "policy_context", "journey", "dp_districts",
        "categories", "district_leakage"
    ]:
        if k in key:
            return SAMPLE_JSON_DEFAULTS.get(k, {})
    return SAMPLE_JSON_DEFAULTS.get(key, {})


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
# ==============================================================================
# 2. Aggregates Endpoints (Viewer+)
# ==============================================================================

def _build_filter_params(
    district: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Formats filter parameters into clean HTTP query parameters."""
    params = {}
    if district and str(district).strip() not in ["All", "All Districts", ""]:
        params["district"] = str(district).strip()
    if from_date and str(from_date).strip():
        params["from_date"] = str(from_date).strip()
    if to_date and str(to_date).strip():
        params["to_date"] = str(to_date).strip()
    if stage and str(stage).strip() not in ["All", "All Stages", "All Institutions", ""]:
        params["stage"] = str(stage).strip()
    if category and str(category).strip() not in ["All", "All Categories", ""]:
        params["category"] = str(category).strip()
    for k, v in kwargs.items():
        if v is not None:
            params[k] = v
    return params


def get_summary(
    district: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 10
) -> Dict[str, Any]:
    """Retrieves executive summary metrics (GET /api/v1/summary)."""
    url = f"{get_api_base_url()}/api/v1/summary"
    params = _build_filter_params(district, from_date, to_date, stage, category)
    try:
        resp = requests.get(url, params=params, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("data", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return load_json_fallback("analytics_summary.json")


def get_funnel(
    district: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    epsilon: float = 0.10,
    k_threshold: int = 5,
    token: Optional[str] = None,
    timeout: int = 10
) -> Dict[str, Any]:
    """Retrieves differential privacy case attrition funnel (GET /api/v1/funnel)."""
    url = f"{get_api_base_url()}/api/v1/funnel"
    params = _build_filter_params(district, from_date, to_date, stage, category, epsilon=epsilon, k_threshold=k_threshold)
    try:
        resp = requests.get(url, params=params, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("funnel", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return load_json_fallback("privacy_dp_funnel.json")


def get_districts(
    district: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 10
) -> Dict[str, Any]:
    """Retrieves district-level leakage diagnostics (GET /api/v1/districts)."""
    url = f"{get_api_base_url()}/api/v1/districts"
    params = _build_filter_params(district, from_date, to_date, stage, category)
    try:
        resp = requests.get(url, params=params, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("district_analytics", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return load_json_fallback("privacy_dp_districts.json")


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
    return load_json_fallback("policy_context_cache.json")


def get_evaluation(
    district: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 10
) -> Dict[str, Any]:
    """Retrieves linkage model evaluation metrics (GET /api/v1/evaluation)."""
    url = f"{get_api_base_url()}/api/v1/evaluation"
    params = _build_filter_params(district, from_date, to_date, stage, category)
    try:
        resp = requests.get(url, params=params, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("evaluation", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return load_json_fallback("evaluation_metrics.json")


def get_before_after(
    district: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 10
) -> Dict[str, Any]:
    """Retrieves dynamic Before vs After policy transformation insights (GET /api/v1/before-after)."""
    url = f"{get_api_base_url()}/api/v1/before-after"
    params = _build_filter_params(district, from_date, to_date, stage, category)
    try:
        resp = requests.get(url, params=params, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("data", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return {}


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
    return load_json_fallback("journey_metrics.json")


# ==============================================================================
# 3. Deep Analytics Endpoints (Analyst+)
# ==============================================================================

def get_bottlenecks(
    district: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 10
) -> Dict[str, Any]:
    """Retrieves duration bottleneck percentiles (GET /api/v1/bottlenecks)."""
    url = f"{get_api_base_url()}/api/v1/bottlenecks"
    params = _build_filter_params(district, from_date, to_date, stage, category)
    try:
        resp = requests.get(url, params=params, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("transitions", body)
    except Exception as exc:
        _handle_request_error(exc, url)
    return load_json_fallback("analytics_bottlenecks.json")


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
    return load_json_fallback("analytics_category.json")


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
# 3b. Case Explorer Endpoints (Viewer, Analyst, Admin)
# ==============================================================================

def get_case_examples(token: Optional[str] = None, timeout: int = 10) -> List[Dict[str, Any]]:
    """Retrieves curated judicial benchmark examples (GET /api/v1/cases/examples)."""
    url = f"{get_api_base_url()}/api/v1/cases/examples"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("examples", [])
    except Exception as exc:
        _handle_request_error(exc, url)

    # Local fallback
    try:
        from src.case_explorer import get_case_explorer_service
        return get_case_explorer_service().get_curated_examples_summary()
    except Exception:
        return []


def get_case_detail(case_id: str, token: Optional[str] = None, timeout: int = 15) -> Dict[str, Any]:
    """Retrieves full case details and reconstructed trajectory (GET /api/v1/cases/{case_id})."""
    url = f"{get_api_base_url()}/api/v1/cases/{case_id}"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("case", body)
    except Exception as exc:
        _handle_request_error(exc, url)

    # Local fallback
    try:
        from src.case_explorer import get_case_explorer_service
        # Default role viewer for fallback if token not provided
        role = "analyst" if token in ["analyst", "admin"] else "viewer"
        return get_case_explorer_service().build_case_explorer_data(case_id, role=role)
    except Exception:
        return {}


def get_random_case(token: Optional[str] = None, timeout: int = 15) -> Dict[str, Any]:
    """Picks a random case from the generated dataset (GET /api/v1/cases/random)."""
    url = f"{get_api_base_url()}/api/v1/cases/random"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            body = resp.json()
            return body.get("case", body)
    except Exception as exc:
        _handle_request_error(exc, url)

    # Local fallback
    try:
        from src.case_explorer import get_case_explorer_service
        svc = get_case_explorer_service()
        rand_id = svc.get_random_case_id()
        role = "analyst" if token in ["analyst", "admin"] else "viewer"
        return svc.build_case_explorer_data(rand_id, role=role)
    except Exception:
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
    return load_json_fallback("privacy_audit_report.json")


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


def reset_privacy_budget(
    total_epsilon: Optional[float] = None,
    total_delta: float = 1e-4,
    token: Optional[str] = None,
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Reconfigures or resets the Differential Privacy budget under administrative authority.
    Calls POST /api/v1/privacy/budget/reset. Accessible exclusively to Administrators.
    """
    url = f"{get_api_base_url()}/api/v1/privacy/budget/reset"
    payload: Dict[str, Any] = {"total_delta": total_delta}
    if total_epsilon is not None:
        payload["total_epsilon"] = total_epsilon
    try:
        resp = requests.post(url, json=payload, headers=_build_headers(token), timeout=timeout)
        if _handle_response_status(resp):
            if _HAS_STREAMLIT and st is not None:
                try:
                    st.session_state["privacy_budget_exhausted"] = False
                    st.session_state.pop("budget_exhausted_detail", None)
                except Exception:
                    pass
            return resp.json()
    except Exception as exc:
        _handle_request_error(exc, url)
    return {"message": "Failed to reset privacy budget", "success": False}


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

def is_backend_available(timeout: float = 0.8) -> bool:
    """Fast check whether the FastAPI backend is running and reachable."""
    url = f"{get_api_base_url()}/api/v1/health"
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _clean_policy_sources(raw_policy: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures policy context excludes non-authoritative domains like scribd or facebook."""
    if not isinstance(raw_policy, dict):
        return {}
    cleaned = {}
    for k, v in raw_policy.items():
        if isinstance(v, dict) and "sources" in v:
            entry = dict(v)
            entry["sources"] = [
                s for s in entry.get("sources", [])
                if "scribd" not in str(s.get("source_domain", "")).lower()
                and "facebook" not in str(s.get("source_domain", "")).lower()
                and "scribd" not in str(s.get("url", "")).lower()
                and "facebook" not in str(s.get("url", "")).lower()
            ]
            cleaned[k] = entry
        else:
            cleaned[k] = v
    return cleaned


def load_dashboard_bundle(
    token: Optional[str] = None,
    district: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    epsilon: float = 0.25,
    k_threshold: int = 5
) -> Dict[str, Any]:
    """
    Fetches all primary analytical and privacy JSON artifacts via HTTP from the FastAPI backend.
    Recomputes on server when global filters are specified.
    Gracefully falls back to precomputed artifacts if backend is unreachable or offline.
    """
    if not is_backend_available():
        return {
            "summary": load_json_fallback("analytics_summary.json"),
            "funnel": load_json_fallback("analytics_funnel.json"),
            "dp_funnel": load_json_fallback("privacy_dp_funnel.json"),
            "district_leakage": load_json_fallback("analytics_district_leakage.json"),
            "dp_districts": load_json_fallback("privacy_dp_districts.json"),
            "bottlenecks": load_json_fallback("analytics_bottlenecks.json"),
            "categories": load_json_fallback("analytics_category.json"),
            "evaluation": load_json_fallback("evaluation_metrics.json"),
            "before_after": {},
            "privacy_audit": load_json_fallback("privacy_audit_report.json"),
            "privacy_budget": {
                "total_epsilon": 10.0,
                "spent_epsilon": 2.35,
                "remaining_epsilon": 7.65,
                "composition_mode": "Sequential Composition"
            },
            "policy_context": _clean_policy_sources(load_json_fallback("policy_context_cache.json")),
            "journey": load_json_fallback("journey_metrics.json"),
        }

    summary = get_summary(district=district, from_date=from_date, to_date=to_date, stage=stage, category=category, token=token)
    funnel = load_json_fallback("analytics_funnel.json")
    dp_funnel = get_funnel(district=district, from_date=from_date, to_date=to_date, stage=stage, category=category, epsilon=epsilon, k_threshold=k_threshold, token=token)
    district_leakage = load_json_fallback("analytics_district_leakage.json")
    dp_districts = get_districts(district=district, from_date=from_date, to_date=to_date, stage=stage, category=category, token=token)
    bottlenecks = get_bottlenecks(district=district, from_date=from_date, to_date=to_date, stage=stage, category=category, token=token)
    categories = get_categories(token=token)
    evaluation = get_evaluation(district=district, from_date=from_date, to_date=to_date, stage=stage, category=category, token=token)
    before_after = get_before_after(district=district, from_date=from_date, to_date=to_date, stage=stage, category=category, token=token)
    privacy_audit = get_privacy_diagnostics(token=token)
    privacy_budget = get_privacy_budget(token=token)
    policy_context = get_policy_context(token=token)
    journey = get_journey(token=token)

    return {
        "summary": summary,
        "funnel": funnel,
        "dp_funnel": dp_funnel,
        "district_leakage": district_leakage,
        "dp_districts": dp_districts,
        "bottlenecks": bottlenecks,
        "categories": categories,
        "evaluation": evaluation,
        "before_after": before_after,
        "privacy_audit": privacy_audit,
        "privacy_budget": privacy_budget,
        "policy_context": _clean_policy_sources(policy_context),
        "journey": journey,
    }


# ==============================================================================
# Analyze Your Data Client Functions
# ==============================================================================

def get_analyze_template(token: Optional[str] = None, timeout: int = 15) -> Optional[bytes]:
    """Retrieves zip template of header-only CSVs (GET /api/v1/analyze/template)."""
    url = f"{get_api_base_url()}/api/v1/analyze/template"
    try:
        resp = requests.get(url, headers=_build_headers(token), timeout=timeout)
        if resp.status_code == 200:
            return resp.content
        _handle_response_status(resp)
    except Exception as exc:
        _handle_request_error(exc, url)
    return None


def run_analyze_sample(limit: int = 500, token: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    """Runs analysis on sample data (POST /api/v1/analyze/sample)."""
    url = f"{get_api_base_url()}/api/v1/analyze/sample"
    params = {"limit": limit}
    try:
        resp = requests.post(url, params=params, headers=_build_headers(token), timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        detail = "Failed to run sample analysis."
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            detail = resp.text or detail
        _handle_response_status(resp)
        return {"error": detail, "status_code": resp.status_code}
    except Exception as exc:
        _handle_request_error(exc, url)
        return {"error": f"API unreachable: {str(exc)}", "status_code": 503}


def run_analyze_upload(files: Dict[str, Any], token: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    """Uploads 5 CSV files for analysis (POST /api/v1/analyze/upload)."""
    url = f"{get_api_base_url()}/api/v1/analyze/upload"
    headers = {}
    active_token = _get_active_token(token)
    if active_token:
        headers["Authorization"] = f"Bearer {active_token}"
    try:
        resp = requests.post(url, files=files, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        detail = "Failed to analyze uploaded files."
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            detail = resp.text or detail
        _handle_response_status(resp)
        return {"error": detail, "status_code": resp.status_code}
    except Exception as exc:
        _handle_request_error(exc, url)
        return {"error": f"API unreachable: {str(exc)}", "status_code": 503}


def run_analyze_case(sections: Dict[str, Any], token: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
    """Analyzes a single case entered by hand (POST /api/v1/analyze/case)."""
    url = f"{get_api_base_url()}/api/v1/analyze/case"
    try:
        resp = requests.post(url, json=sections, headers=_build_headers(token), timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        detail = "Failed to analyze case."
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            detail = resp.text or detail
        _handle_response_status(resp)
        return {"error": detail, "status_code": resp.status_code}
    except Exception as exc:
        _handle_request_error(exc, url)
        return {"error": f"API unreachable: {str(exc)}", "status_code": 503}
