"""
tests/demo_api.py - Script to execute and print live API outputs.
"""

import json
from starlette.testclient import TestClient
from src.api import create_app, DATA_STORE

def main():
    DATA_STORE.budget_tracker.reset()
    DATA_STORE.audit_logger.clear()
    client = TestClient(create_app(data_dir="data"))

    viewer_headers = {"X-API-Key": "viewer-key-leaky-2026"}
    analyst_headers = {"X-API-Key": "analyst-key-leaky-2026"}
    admin_headers = {"X-API-Key": "admin-key-leaky-2026"}

    print("=" * 70)
    print("LIVE API DEMONSTRATION: CASELENS PRIVACY-PRESERVING SERVICE")
    print("=" * 70)

    print("\n[+] 1. HEALTH & PRIVACY ENGINE STATUS (GET /api/v1/health)")
    r = client.get("/api/v1/health")
    print(json.dumps(r.json(), indent=2))

    print("\n[+] 2. DIFFERONENTIALLY PRIVATE FUNNEL AGGREGATE (GET /api/v1/funnel - Role: Viewer)")
    r = client.get("/api/v1/funnel", headers=viewer_headers)
    data = r.json()
    print(f"Privacy Mechanism: {data['privacy_protection']['mechanism']}")
    print(f"Epsilon Cost: {data['privacy_protection']['epsilon_spent']}")
    print("Stages:")
    for s in data["funnel"]["funnel_stages"]:
        print(f"  * {s['stage_name']}: raw={s['raw_case_count']} -> noisy={s['noisy_case_count']} (+/-{s['confidence_margin_95']}), retention={s['dp_retention_pct']}%, suppressed={s['is_k_anonymity_suppressed']}")

    print("\n[+] 3. DISTRICT LEAKAGE AGGREGATES & VULNERABILITIES (GET /api/v1/districts - Role: Viewer)")
    r = client.get("/api/v1/districts", headers=viewer_headers)
    dist_data = r.json()["district_analytics"]["districts"]
    for dist, info in dist_data.items():
        print(f"  * {dist}: status={info['status']}, raw={info['raw_cases']} -> noisy={info['noisy_cases']}, score={info['vulnerability_score']}, primary_leak={info['primary_policy_leak']}")

    print("\n[+] 4. DURATION BOTTLENECKS & STATUTORY COMPLIANCE (GET /api/v1/bottlenecks - Role: Analyst)")
    r = client.get("/api/v1/bottlenecks", headers=analyst_headers)
    b_data = r.json()["transitions"]
    for k in list(b_data.keys())[:3]:
        t = b_data[k]
        print(f"  * Transition: {k}")
        print(f"    Sample: {t['sample_count']}, Median: {t['median_days']} days, IQR: {t['iqr_days']} days, P95: {t['p95_days']} days")

    print("\n[+] 5. DYNAMIC DP AGGREGATE QUERY (POST /api/v1/query/aggregate - Role: Analyst)")
    req_body = {"dimension": "case_category", "epsilon": 0.25, "k_threshold": 5}
    r = client.post("/api/v1/query/aggregate", json=req_body, headers=analyst_headers)
    print(json.dumps(r.json(), indent=2))

    print("\n[+] 6. STATUTORY POLICY BENCHMARKS & TINYFISH INTEGRATION (GET /api/v1/policy-context - Role: Viewer)")
    r = client.get("/api/v1/policy-context", headers=viewer_headers)
    pol_data = r.json()
    print("Statutory Benchmarks Available:")
    for k, v in pol_data["statutory_benchmarks"].items():
        print(f"  * [{k}] {v['title']}: {v['statutory_reference']} - {v['mandate_summary']}")

    print("\n[+] 7. RBAC RESTRICTIONS & PRIVACY GUARDRAILS ENFORCEMENT")
    # Unauthenticated
    r_unauth = client.get("/api/v1/funnel")
    print(f"  * Unauthenticated call to /funnel -> Status {r_unauth.status_code}: {r_unauth.json()['detail']}")
    # Viewer forbidden from analyst
    r_view_bot = client.get("/api/v1/bottlenecks", headers=viewer_headers)
    print(f"  * Viewer call to /bottlenecks (Analyst only) -> Status {r_view_bot.status_code}: {r_view_bot.json()['detail']}")
    # Viewer forbidden from admin
    r_view_audit = client.get("/api/v1/audit/logs", headers=viewer_headers)
    print(f"  * Viewer call to /audit/logs (Admin only) -> Status {r_view_audit.status_code}: {r_view_audit.json()['detail']}")
    # Forbidden PII probe
    r_pii = client.post("/api/v1/query/aggregate", json={"dimension": "district", "filters": {"case_id": "ENT-001"}}, headers=analyst_headers)
    print(f"  * Probing forbidden case_id filter -> Status {r_pii.status_code}: {r_pii.json()['detail']}")

    print("\n[+] 8. SEQUENTIAL COMPOSITION PRIVACY BUDGET STATUS (GET /api/v1/privacy/budget - Role: Analyst)")
    r = client.get("/api/v1/privacy/budget", headers=analyst_headers)
    b_stat = r.json()
    print(f"  * Total Epsilon: {b_stat['total_epsilon']}")
    print(f"  * Spent Epsilon: {b_stat['spent_epsilon']}")
    print(f"  * Remaining Epsilon: {b_stat['remaining_epsilon']}")
    print(f"  * Budget Ledger Queries: {b_stat['query_count']}")

    print("\n[+] 9. IMMUTABLE QUERY AUDIT LOG (GET /api/v1/audit/logs?limit=4 - Role: Admin)")
    r = client.get("/api/v1/audit/logs?limit=4", headers=admin_headers)
    print(json.dumps(r.json(), indent=2))
    print("=" * 70)

if __name__ == "__main__":
    main()
