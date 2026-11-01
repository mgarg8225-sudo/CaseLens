"""
tests/test_runtime_registry.py - Verifies runtime behavior, resilience, and immutability of .active_analysis_registry.json.

Requirements tested:
1. (a) Registry file missing on startup -> created automatically with the sample dataset published & active.
2. (b) Registry file corrupted/empty on startup -> recreated automatically with safe default state instead of crashing.
3. (c) Viewer can read the published sample dataset right after a fresh start.
4. (d) Pipeline data/*.json analytics result files remain 100% byte-for-byte identical.
5. Robustness: data/ directory unwritable -> falls back gracefully to in-memory state without crashing.
"""

import os
import json
import glob
import hashlib
import pytest
from starlette.testclient import TestClient

from src.api import create_app, DataStore, Role
import src.api as api_mod

VIEWER_HEADERS = {"X-API-Key": "viewer-key-leaky-2026"}
ANALYST_HEADERS = {"X-API-Key": "analyst-key-leaky-2026"}
REGISTRY_PATH = os.path.join("data", ".active_analysis_registry.json")


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry file is cleaned before and after each test."""
    if os.path.exists(REGISTRY_PATH):
        try:
            os.remove(REGISTRY_PATH)
        except Exception:
            pass
    yield
    if os.path.exists(REGISTRY_PATH):
        try:
            os.remove(REGISTRY_PATH)
        except Exception:
            pass


def test_registry_file_missing_created_with_sample_dataset_published():
    """
    (a) On startup, if data/.active_analysis_registry.json is missing,
    it is created automatically with the sample dataset published and active.
    """
    assert not os.path.exists(REGISTRY_PATH)

    # Initialize DataStore on missing registry
    store = DataStore(data_dir="data")

    # Verify file was created automatically
    assert os.path.exists(REGISTRY_PATH), "Expected .active_analysis_registry.json to be created on startup"

    # Verify in-memory state
    assert store.active_dataset_id == "sample"
    assert "Benchmark Sample Dataset" in (store.active_dataset_label or "")
    assert store.active_analysis_result is not None
    assert store.registered_datasets.get("sample", {}).get("is_published") is True

    # Verify persisted JSON contents on disk
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert disk_data["active_dataset_id"] == "sample"
    assert "Benchmark Sample Dataset" in disk_data["active_dataset_label"]
    assert disk_data["registered_datasets"]["sample"]["is_published"] is True


def test_registry_file_corrupted_or_empty_recreated():
    """
    (b) On startup, if data/.active_analysis_registry.json is empty or contains corrupted JSON,
    it is recreated automatically with the safe default state instead of crashing.
    """
    # 1. Test empty file (0 bytes)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        f.write("")
    assert os.path.getsize(REGISTRY_PATH) == 0

    store_empty = DataStore(data_dir="data")
    assert store_empty.active_dataset_id == "sample"
    assert os.path.getsize(REGISTRY_PATH) > 0
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert disk_data["active_dataset_id"] == "sample"

    # 2. Test corrupted JSON
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        f.write("{invalid_json_corrupted: [true, missing_bracket")

    store_corrupt = DataStore(data_dir="data")
    assert store_corrupt.active_dataset_id == "sample"
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        recreated_data = json.load(f)
    assert recreated_data["active_dataset_id"] == "sample"
    assert recreated_data["registered_datasets"]["sample"]["is_published"] is True


def test_viewer_reads_published_sample_on_fresh_start():
    """
    (c) Viewer can read the published sample dataset right after a fresh start.
    Simulates fresh server boot on ephemeral disk (no preexisting registry file).
    """
    assert not os.path.exists(REGISTRY_PATH)

    # Re-initialize global DATA_STORE and create app
    api_mod.DATA_STORE = DataStore(data_dir="data")
    app = create_app(data_dir="data")
    client = TestClient(app)

    # Viewer queries active dataset
    resp = client.get("/api/v1/datasets/active", headers=VIEWER_HEADERS)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["dataset_id"] == "sample"
    assert "Benchmark Sample Dataset" in data["dataset_label"]
    assert data["analysis_result"] is not None
    assert data["analysis_result"]["summary_kpis"]["total_cases"] > 0
    assert len(data["analysis_result"]["funnel_stages"]) > 0

    # Viewer queries list of available datasets
    list_resp = client.get("/api/v1/datasets", headers=VIEWER_HEADERS)
    assert list_resp.status_code == 200
    datasets = list_resp.json().get("datasets", [])
    assert len(datasets) >= 1
    sample_entry = next((d for d in datasets if d["dataset_id"] == "sample"), None)
    assert sample_entry is not None
    assert sample_entry["is_active"] is True
    assert sample_entry["is_published"] is True


def test_data_json_files_remain_byte_identical_during_registry_ops():
    """
    (d) Pipeline data/*.json analytics result files remain 100% byte-for-byte identical
    across registry creation, corruption recovery, and dataset access.
    """
    json_files = [
        f for f in glob.glob("data/*.json")
        if not f.endswith("audit_log.json") and "active_analysis_registry" not in f
    ]
    assert len(json_files) >= 10, f"Expected at least 10 pipeline JSON files, found {len(json_files)}"

    def get_hashes():
        return {fpath: hashlib.sha256(open(fpath, "rb").read()).hexdigest() for fpath in json_files}

    initial_hashes = get_hashes()

    # 1. Trigger missing file creation
    store1 = DataStore(data_dir="data")
    assert store1.active_dataset_id == "sample"

    # 2. Trigger corrupted file recreation
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        f.write("{corrupted: true")
    store2 = DataStore(data_dir="data")
    assert store2.active_dataset_id == "sample"

    # 3. Trigger Viewer access
    api_mod.DATA_STORE = store2
    app = create_app(data_dir="data")
    client = TestClient(app)
    resp = client.get("/api/v1/datasets/active", headers=VIEWER_HEADERS)
    assert resp.status_code == 200

    # Verify all analytical files are byte-for-byte identical
    after_hashes = get_hashes()
    for fpath, h_before in initial_hashes.items():
        assert after_hashes[fpath] == h_before, f"Immutability violation: {fpath} was modified!"


def test_unwritable_directory_falls_back_to_in_memory(monkeypatch):
    """
    (Requirement 3) If data directory is not writable, DataStore falls back to in-memory state
    without crashing, and can still serve datasets.
    """
    real_open = open

    def mock_open(file, mode="r", *args, **kwargs):
        if ".active_analysis_registry.json" in str(file) and "w" in mode:
            raise PermissionError("Simulated read-only filesystem / permission denied")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)

    store = DataStore(data_dir="data")
    # In-memory state is valid despite disk write failure
    assert store.active_dataset_id == "sample"
    assert store.active_analysis_result is not None
