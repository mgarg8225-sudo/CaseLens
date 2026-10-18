# Project Progress: The Leaky Pipeline

## Status Overview
- **Phase 1: Synthetic Data Generation (`src/generate_data.py`)** - COMPLETED & COMMITTED.
- **Phase 2: Privacy-Preserving Record Linkage (`src/linkage.py`)** - COMPLETED & COMMITTED.
- **Phase 3: Linkage Evaluation & Error Diagnostics (`src/evaluate.py`)** - COMPLETED & COMMITTED.
- **Phase 4: Case Journey Reconstruction & Policy Leakage Analytics (`src/journey.py`, `src/analytics.py`)** - COMPLETED.
  - Event log reconstruction (14,384 events across 5,125 cases)
  - Process mining with PM4Py and Directly-Follows Graph (DFG)
  - District policy leak point detection (D3 forensic delay, D6 chargesheet attrition, D7 court delay)
  - DuckDB OLAP analytical SQL queries & privacy k-anonymity suppression
- **Phase 5: Survivor Privacy Protection & Differential Privacy Engine (`src/privacy.py`)** - COMPLETED.
  - Pure $\epsilon$-Differential Privacy via Laplace Mechanism ($\Delta f = 1$) & Gaussian Mechanism $(\epsilon, \delta)$
  - Global Privacy Budget accounting under Sequential Composition ($\sum \epsilon_i \le \epsilon_{\text{total}}$)
  - Quasi-identifier equivalence class auditing ($k$-anonymity, $l$-diversity, $t$-closeness)
  - Adversarial attack simulation (Marksmanship / Singling-Out, Prosecutor, Journalist risk)
  - Differencing / membership inference attack neutralization with plausible deniability
  - Differentially private analytical artifacts generated (`privacy_dp_funnel.json`, `privacy_dp_districts.json`, `privacy_audit_report.json`)
  - Zero-PII verification guaranteeing zero survivor identifying fields in published tables
  - 38/38 unit tests passing across all test suites
- **Phase 5.5: External Policy Context Integration (`src/policy_context.py`)** - COMPLETED.
  - Official TinyFish Search, Fetch, and Web Agent APIs integrated for public policy & infrastructure benchmarking.
  - Zero-dependency `.env` reader for `TINYFISH_API_KEY` (no external `python-dotenv` package installed/required; key never hardcoded or printed).
  - Non-negotiable Privacy Guardrail (`sanitize_query`): Strictly blocks case/record IDs (`ENT-*`, `FIR-*`, `FSL-*`, `CRT-*`), synthetic district tokens (`D1`-`D8`, `Rudrapur`, `Shivgarh`, etc.), survivor PII (names, phone, Aadhaar, DOB), and micro-cohort counts.
  - Top 5 leak points benchmarked with real-world statutory and infrastructure sources:
    1. Forensic Lab Turnaround Delay (`LP-01-FORENSIC-DELAY`) - Sources: `ijlmh.com`, `ignited.in`, `scribd.com`
    2. FIR to Chargesheet Drop-off (`LP-02-POLICE-ATTRITION`) - Sources: `vidhikarya.com`, `bprd.nic.in`
    3. Court Trial Adjudication Pendency (`LP-03-COURT-DELAY`) - Sources: `pib.gov.in` (Fast Track Special Courts)
    4. Medico-Legal Examination Delay (`LP-04-MEDICAL-EXAM-DELAY`) - Sources: `who.int`, `arogyakeralam.gov.in` (24h MLC protocol)
    5. Victim Compensation Lag (`LP-05-VICTIM-COMPENSATION-DELAY`) - Sources: `nalsa.gov.in` (interim relief disbursal)
  - Autonomous TinyFish Web Agent automation task visiting public statutory portal extracting POCSO Act Section 35(2) 1-year trial completion mandate.
  - Local offline cache fallback (`data/policy_context_cache.json`) with timeouts and max 2 retries, guaranteeing 100% offline demo resilience.
  - External API audit ledger (`data/external_api_audit_log.json`) tracking timestamp, endpoint, sanitized query, and status without leaking any case data.
- **Phase 6: Privacy-Preserving FastAPI Backend Service (`src/api.py`)** - COMPLETED.
  - Production-grade FastAPI service exposing analytical aggregates exclusively (Zero Microdata / Zero PII guarantee).
  - Role-Based Access Control (RBAC) with hierarchical permissions across three institutional tiers:
    * `admin`: Full analytical access, immutable audit ledger inspection, privacy budget reconfiguration, Phase 5 privacy diagnostics, and TinyFish policy context refresh.
    * `analyst`: Deep duration bottlenecks, crime category disparities, privacy budget accounting, and dynamic DP aggregate queries.
    * `viewer`: High-level differentially private funnel aggregates, district leakage rankings, and statutory benchmark contexts.
  - Multi-method authentication supporting `X-API-Key`, standard `Authorization: Bearer <token>`, and `X-User-Role` headers.
  - Privacy Layer Enforcement:
    * Mathematical Differential Privacy accounting under Sequential Composition ($\sum \epsilon_i \le \epsilon_{\text{total}}$).
    * Dynamic Laplace Mechanism noise injection ($\Delta f = 1$) on count queries with 95% Confidence Bounds ($\pm 2.996 \cdot b$).
    * Strict $k$-anonymity small-cell suppression ($k \ge 5$) masking micro-cohorts as `"<5"`.
    * Zero-PII Response Validator running `verify_zero_pii_leakage()` on every outbound payload, blocking survivor names, phones, Aadhaar, DOB, and addresses.
    * Query sanitization strictly rejecting case ID probing (`ENT-*`, `FIR-*`, `CASE-*`) and PII filters with HTTP 400.
    * Global privacy budget exhaustion protection: returns HTTP 429 Too Many Requests when allocated $\epsilon$ is exhausted.
  - Immutable Audit Ledger (`data/api_audit_log.json`): logs every API request with UUID, UTC timestamp, caller role, endpoint, HTTP status, client IP, $\epsilon$ consumed, Zero-PII verification flag, and latency.
  - Comprehensive unit & integration tests (`tests/test_api.py`) passing with 100% success rate (90/90 passing across entire project repository).
- **Next Step**: Phase 7 (Streamlit Interactive Frontend Dashboard for Multi-Stakeholder Policy Analytics & Live Leak Point Interventions).
