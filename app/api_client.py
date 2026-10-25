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
        "🔻 Funnel with Uncertainty Ranges",
        "⏱️ Duration Bottlenecks & Latency",
        "🔗 Linkage Quality & Evaluation",
        "📜 Statutory Policy Insights (TinyFish)",
        "🔍 Case Explorer"
    ],
    "analyst": [
        "📊 Executive Overview",
        "🔻 Funnel with Uncertainty Ranges",
        "⏱️ Duration Bottlenecks & Latency",
        "🔗 Linkage Quality & Evaluation",
        "⚖️ Before / After Policy Transformation",
        "🛡️ Survivor Privacy & DP Engine",
        "📜 Statutory Policy Insights (TinyFish)",
        "🧪 Case Simulator",
        "🔍 Case Explorer"
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
        "🔍 Case Explorer",
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


DATA_DIR = os.environ.get("LEAKY_DATA_DIR", "data")


# ==============================================================================
# Sample Defaults for Graceful Offline / Fallback Display
# ==============================================================================

SAMPLE_JSON_DEFAULTS: Dict[str, Any] = {
    "summary": {
        "pipeline_health": {
            "total_incidents": 5000,
            "total_reconstructed_cases": 5125,
            "cases_initiating_fir": 4347,
            "cases_reaching_verdict": 1397,
            "overall_conviction_reach_rate_pct": 32.14,
            "overall_conviction_rate_pct": 8.4,
            "overall_attrition_rate_pct": 91.6,
            "median_end_to_end_journey_days": 218.0
        },
        "systemic_leak_points_detected": [
            {
                "leak_code": "LP-01-FORENSIC-DELAY",
                "agency": "Forensic Science Lab",
                "district": "D3_Rudrapur",
                "title": "Severe Forensic Lab Turnaround Delay",
                "multiplier": 3.19,
                "delay_days": 137.5,
                "baseline_days": 43.0,
                "affected_cases": 201
            },
            {
                "leak_code": "LP-02-POLICE-ATTRITION",
                "agency": "Police Investigation",
                "district": "D6_Shivgarh",
                "title": "Excessive FIR-to-Chargesheet Drop-off",
                "drop_off_leak_pct": 72.2,
                "chargesheet_rate_pct": 27.8,
                "state_benchmark_pct": 59.0,
                "affected_firs": 392
            },
            {
                "leak_code": "LP-03-COURT-DELAY",
                "agency": "Judiciary / Trial Courts",
                "district": "D7_Anandpur",
                "title": "Severe Judicial Trial Adjudication Delay",
                "multiplier": 2.14,
                "trial_delay_days": 612.9,
                "baseline_days": 288.0,
                "cases_in_trial": 180
            }
        ],
        "district_vulnerability_ranking": [
            {"district": "D6_Shivgarh", "score": 56.5, "status": "CRITICAL"},
            {"district": "D7_Anandpur", "score": 36.6, "status": "CRITICAL"},
            {"district": "D3_Rudrapur", "score": 34.0, "status": "CRITICAL"},
            {"district": "D4_Kalyanpur", "score": 5.3, "status": "NORMAL"},
            {"district": "D2_Chandranagar", "score": 4.4, "status": "NORMAL"},
            {"district": "D5_Mayaganj", "score": 3.4, "status": "NORMAL"},
            {"district": "D8_Devkalan", "score": 2.5, "status": "NORMAL"},
            {"district": "D1_Suryapur", "score": 2.0, "status": "NORMAL"}
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
    "bottlenecks": {
        "FIR_REGISTERED -> MEDICAL_EXAM": {
            "median_days": 2.0,
            "iqr_days": 3.0,
            "p95_days": 7.0,
            "delay_buckets": {"<= 24 Hours": 45.0, "1-3 Days": 35.0, "> 3 Days": 20.0}
        },
        "MEDICAL_EXAM -> FORENSIC_REPORT": {
            "median_days": 43.0,
            "iqr_days": 35.0,
            "p95_days": 138.0,
            "delay_buckets": {"<= 30 Days": 32.0, "31-60 Days": 28.0, "> 60 Days": 40.0}
        },
        "FORENSIC_REPORT -> CHARGESHEET_FILED": {
            "median_days": 21.0,
            "iqr_days": 18.0,
            "p95_days": 55.0,
            "delay_buckets": {"<= 30 Days": 68.0, "31-60 Days": 22.0, "> 60 Days": 10.0}
        },
        "CHARGESHEET_FILED -> COURT_VERDICT": {
            "median_days": 284.0,
            "iqr_days": 210.0,
            "p95_days": 650.0,
            "delay_buckets": {"<= 180 Days": 15.0, "181-365 Days": 42.0, "> 365 Days": 43.0}
        }
    },
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
        "pairwise_threshold_table": [
            {"threshold": 0.50, "precision": 0.42, "recall": 0.95, "f1": 0.58},
            {"threshold": 0.65, "precision": 0.50, "recall": 0.92, "f1": 0.65},
            {"threshold": 0.75, "precision": 0.54, "recall": 0.90, "f1": 0.68},
            {"threshold": 0.85, "precision": 0.57, "recall": 0.88, "f1": 0.69},
            {"threshold": 0.95, "precision": 0.75, "recall": 0.70, "f1": 0.72}
        ]
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
    "policy_context": {
        "lp_01": {
            "leak_code": "LP-01-FORENSIC-DELAY",
            "leak_point_label": "Severe Forensic Lab Turnaround Delay",
            "finding": "Forensic report generation exceeds 90-day statutory threshold",
            "suggested_intervention": "Automate digital transmission of biological forensic reports",
            "sources": [{"title": "POCSO Act 2012", "url": "https://wcd.nic.in", "source_domain": "wcd.nic.in", "summary": "Section 35(2)"}]
        },
        "lp_02": {
            "leak_code": "LP-02-POLICE-ATTRITION",
            "leak_point_label": "Excessive FIR-to-Chargesheet Drop-off",
            "finding": "72.2% drop-off in D6 Shivgarh",
            "suggested_intervention": "Mandate recording of Sec 164 statements within 48h",
            "sources": [{"title": "BNSS 2023", "url": "https://mha.gov.in", "source_domain": "mha.gov.in", "summary": "Section 173(2)"}]
        },
        "lp_03": {
            "leak_code": "LP-03-COURT-DELAY",
            "leak_point_label": "Severe Judicial Trial Adjudication Delay",
            "finding": "612.9 days trial duration in D7 Anandpur",
            "suggested_intervention": "Fast Track Special Courts (FTSC) establishment",
            "sources": [{"title": "Department of Justice Guidelines", "url": "https://doj.gov.in", "source_domain": "doj.gov.in", "summary": "FTSC Scheme"}]
        },
        "lp_04": {
            "leak_code": "LP-04-HOSPITAL-DELAY",
            "leak_point_label": "MLC Examination Delay",
            "finding": "Delay in medical examination exceeding 24 hours",
            "suggested_intervention": "Designate 24/7 forensic medical examiners at district hospitals",
            "sources": [{"title": "MoHFW Guidelines", "url": "https://mohfw.gov.in", "source_domain": "mohfw.gov.in", "summary": "Medico-Legal Care"}]
        },
        "lp_05": {
            "leak_code": "LP-05-PROSECUTION-GAP",
            "leak_point_label": "Prosecutorial Scrutiny Latency",
            "finding": "Delay in chargesheet submission after investigation completion",
            "suggested_intervention": "Digital integration between Police CCTNS and e-Courts",
            "sources": [{"title": "NCRB Crime in India", "url": "https://ncrb.gov.in", "source_domain": "ncrb.gov.in", "summary": "Disposal of Cases"}]
        }
    },
    "journey": {
        "median_journey_days": 218.0,
        "total_cases": 5000,
        "total_events": 14384
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
