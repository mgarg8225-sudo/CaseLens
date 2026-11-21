"""
tests/test_nan_sanitization.py - Regression & Unit Tests for NaN/Inf Sanitization.

Validates that:
1. clean_nan recursively converts float NaN, +inf, -inf to None for dict, list, tuple,
   numpy scalars (numpy.floating, numpy.integer, numpy.bool_) and pandas values.
2. SafeJSONResponse central wrapper prevents starlette json.dumps ValueError.
3. POST /api/v1/analyze/sample?limit=500 returns HTTP 200 with strictly RFC 8259-compliant JSON.
"""

import json
import math
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from src.utils import clean_nan, SafeJSONResponse
from src.api import create_app, DATA_STORE


# ==============================================================================
# 1. clean_nan Unit Tests
# ==============================================================================

def test_clean_nan_scalars():
    """Verifies that clean_nan converts nan, inf, -inf and numpy scalars correctly."""
    # Standard float specials -> None
    assert clean_nan(float("nan")) is None
    assert clean_nan(float("inf")) is None
    assert clean_nan(float("-inf")) is None

    # NumPy float specials -> None
    assert clean_nan(np.nan) is None
    assert clean_nan(np.inf) is None
    assert clean_nan(-np.inf) is None
    assert clean_nan(np.float64(np.nan)) is None
    assert clean_nan(np.float32(np.inf)) is None

    # Normal numeric values preserved
    assert clean_nan(0) == 0
    assert clean_nan(0.0) == 0.0
    assert clean_nan(3.14159) == 3.14159
    assert clean_nan(-42.5) == -42.5

    # NumPy integers and booleans preserved
    assert clean_nan(np.int64(42)) == 42
    assert isinstance(clean_nan(np.int64(42)), int)
    assert clean_nan(np.bool_(True)) is True
    assert clean_nan(np.bool_(False)) is False
    assert clean_nan(True) is True
    assert clean_nan(False) is False

    # Strings, None, bytes preserved
    assert clean_nan("case_123") == "case_123"
    assert clean_nan(None) is None
    assert clean_nan(b"bytes_data") == b"bytes_data"


def test_clean_nan_pandas_types():
    """Verifies that clean_nan handles pandas NA, NaT, Series, and DataFrames."""
    assert clean_nan(pd.NA) is None
    assert clean_nan(pd.NaT) is None

    # Series
    s = pd.Series([10.5, np.nan, float("inf"), 20.0])
    cleaned_s = clean_nan(s)
    assert cleaned_s == [10.5, None, None, 20.0]

    # DataFrame
    df = pd.DataFrame([
        {"id": 1, "rate": np.nan, "label": "A"},
        {"id": 2, "rate": float("-inf"), "label": None}
    ])
    cleaned_df = clean_nan(df)
    assert cleaned_df == [
        {"id": 1, "rate": None, "label": "A"},
        {"id": 2, "rate": None, "label": None}
    ]


def test_clean_nan_nested_structures():
    """Verifies that nested dicts, lists, and tuples are recursively cleaned and JSON serializable."""
    nested = {
        "summary": {
            "mean": float("nan"),
            "ratio": float("inf"),
            "valid_stat": 95.5
        },
        "records": [
            {"name": "test_1", "duration": np.float64(np.nan)},
            {"name": "test_2", "duration": 14.0, "tuple_field": (1, float("-inf"), 3)}
        ],
        "tags": {"tag1", float("nan")},
        "empty_cell": None
    }

    cleaned = clean_nan(nested)

    assert cleaned["summary"]["mean"] is None
    assert cleaned["summary"]["ratio"] is None
    assert cleaned["summary"]["valid_stat"] == 95.5
    assert cleaned["records"][0]["duration"] is None
    assert cleaned["records"][1]["duration"] == 14.0
    assert cleaned["records"][1]["tuple_field"] == (1, None, 3)
    assert cleaned["empty_cell"] is None

    # Guaranteed strictly JSON-compliant with allow_nan=False
    dumped = json.dumps(cleaned, allow_nan=False)
    assert "NaN" not in dumped
    assert "Infinity" not in dumped


def test_safe_json_response_render():
    """Verifies that SafeJSONResponse renders payloads containing NaN/Inf without ValueError."""
    resp = SafeJSONResponse(content={"status": "ok", "value": float("nan"), "inf": float("inf")})
    rendered = resp.body.decode("utf-8")
    parsed = json.loads(rendered)
    assert parsed == {"status": "ok", "value": None, "inf": None}


# ==============================================================================
# 2. API Endpoint Regression Tests: POST /api/v1/analyze/sample
# ==============================================================================

@pytest.fixture(scope="module")
def api_test_client():
    """Creates TestClient with initialized test database."""
    app = create_app(data_dir="data")
    DATA_STORE.budget_tracker.reset()
    with TestClient(app) as client:
        yield client


def test_analyze_sample_returns_200_valid_json(api_test_client):
    """
    Regression Test for Render live bug:
    POST /api/v1/analyze/sample?limit=500 returned HTTP 500 due to NaN in legal_sections.
    Verifies that limit=500 now returns HTTP 200 with valid, parseable JSON containing no NaNs.
    """
    headers = {"X-API-Key": "analyst-key-leaky-2026"}
    response = api_test_client.post("/api/v1/analyze/sample?limit=500", headers=headers)

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"

    data = response.json()
    assert data.get("status") == "success"
    assert "summary_kpis" in data
    assert "events" in data

    # Verify no raw NaN floats remain in the response body text
    body_text = response.text
    # In RFC 8259 JSON, valid null is 'null', not unquoted 'NaN' or 'Infinity'
    assert ": NaN" not in body_text
    assert ": -NaN" not in body_text
    assert ": Infinity" not in body_text
    assert ": -Infinity" not in body_text

    # Re-verify strict non-nan JSON dumping
    json.dumps(data, allow_nan=False)


def test_analyze_sample_viewer_forbidden(api_test_client):
    """Verifies that role enforcement remains intact (Viewer receives HTTP 403)."""
    headers = {"X-API-Key": "viewer-key-leaky-2026"}
    response = api_test_client.post("/api/v1/analyze/sample?limit=500", headers=headers)
    assert response.status_code == 403
