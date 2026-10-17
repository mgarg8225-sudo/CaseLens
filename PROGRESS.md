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
  - 38 new unit tests in `tests/test_policy_context.py` (76/76 passing across entire repository).
- **Next Step**: Phase 6 (`src/api.py` - FastAPI Backend Service for Privacy-Preserving Analytical Endpoints and Policy Insights Dashboard Integration).
