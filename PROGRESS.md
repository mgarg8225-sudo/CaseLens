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
- **Phase 7: Multi-Stakeholder Streamlit Policy Dashboard (`app/dashboard.py`, `dashboard.py`)** - COMPLETED.
  - Interactive multi-page Streamlit web application providing high-resolution policy intelligence with mathematical Differential Privacy protection:
    1. **Overview Page**: Headline KPIs (5,125 reconstructed cases, 4,347 FIRs, 1,397 verdicts), composite District Vulnerability Index ranking (D6 Shivgarh 56.5, D7 Anandpur 36.6, D3 Rudrapur 34.0), and systemic leak point detection cards.
    2. **Funnel with Uncertainty Ranges**: 5-stage case attrition funnel (FIR -> MLC -> Forensic -> Chargesheet -> Verdict) featuring Laplace Differential Privacy error bars ($\pm 95\%$ Confidence Intervals), raw-vs-noisy overlay toggles, and $k$-anonymity small-cell suppression ($< 5$).
    3. **Duration Bottlenecks & Latency**: Inter-stage duration percentiles (Median, IQR, P90, P95) and statutory latency compliance tracking (CrPC 173 / BNSS 193 60-day mandate, 24h medical exam, 30-day forensic reporting) binned into latency delay buckets.
    4. **Linkage Quality & Evaluation**: Phase 3 Splink probabilistic record linkage metrics (B-Cubed F1: 0.929, Macro Purity: 97.2%, Pairwise Recall: 88.5%), interactive Precision-Recall trade-off curve across probability thresholds (0.10 to 0.95), and false-positive diagnostics.
    5. **Before / After Policy Transformation**: Side-by-side comparison illustrating how isolated institutional silos conceal critical drop-offs whereas longitudinal record linkage exposes exact root causes (D3 forensic delay, D6 police attrition, D7 court delay).
    6. **Survivor Privacy & Differential Privacy Engine**: Live interactive Laplace mechanism simulator with $\epsilon$ slider and 95% error margin visualizer, Sequential Composition privacy budget accounting ledger, $k$-anonymity, $l$-diversity, $t$-closeness diagnostics, adversarial singling-out / prosecutor risk evaluation, and Zero-PII verification.
    7. **Statutory Policy Insights (TinyFish)**: Official government policy benchmarks prioritizing `.gov.in` and `.nic.in` sources (`dfss.gov.in`, `bprd.nic.in`, `pib.gov.in`, `mohfw.gov.in`, `nalsa.gov.in`), autonomous TinyFish Web Agent legal fact extraction (POCSO Sec 35(2) 1-year trial mandate), and actionable institutional interventions.
  - 96/96 unit and integration tests passing across all 8 test suites.
- **Next Step**: Project Wrap-up, Final Review, and Deployment Packaging.
