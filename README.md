# CaseLens: The Leaky Pipeline
### Tracing Gaps in the Justice System • Reforged '26 (Open Source Track)

> **Syntax Squad**: Mehak Garg • Naina Shukla • Aarna Bansal • Sakshi Sharma  
> **Mission**: Diagnosing systemic attrition in violence-against-women cases across fragmented institutional silos without compromising survivor confidentiality.  
> **Global Alignment**: **UN SDG 5** (Gender Equality) & **UN SDG 16** (Peace, Justice and Strong Institutions).

---

## Reforged '26 Core Empirical Highlights

CaseLens reconstructs longitudinal case journeys across 5 disconnected institutional silos (Police FIR, Hospital MLC, Forensic Science Laboratories, Prosecution, and District Courts). On synthetic data calibrated to public judicial statistics, CaseLens delivers:

- **$\approx 3.2\times$ Forensic Latency Signal (District 3)**: Uncovers extreme lab turnaround bottlenecks (137.5 days vs 43.0 day baseline; 201 affected cases).
- **Chargesheet Drop-Off Leak (District 6)**: Detects severe police-to-prosecution attrition where $72.2\%$ of cases leak out (only $27.8\%$ reaching court vs $59.0\%$ state benchmark).
- **High-Precision Record Linkage (Splink)**: Achieves **$88.5\%$ Pairwise Recall** and **$92.9\%$ B-Cubed $F_1$** across 700K+ candidate pair evaluations.
- **Survivor Confidentiality**: Enforces Pure Laplace Differential Privacy ($\epsilon$-DP), small-cell suppression ($k \ge 5$), and zero-PII security ledgers.

---

## Technical Architecture & Workflow

In accordance with the Reforged '26 architecture:
1. **Ingest**: Approved records & metadata via FastAPI ingestion endpoints.
2. **Normalize**: Standardization of names, transliteration variations, dates, and institutional schemas in DuckDB.
3. **Match**: Probabilistic Fellegi-Sunter scoring (Splink) with Jaro-Winkler string distance and soundex phonetics.
4. **Review**: Low-confidence candidate pairs ($0.70 \le \text{prob} < 0.85$) quarantined to the human review queue.
5. **Map**: Chronological milestone reconstruction and differential-privacy aggregate reporting.


## Quickstart & How to Run

### 1. Environment Setup

CaseLens requires **Python 3.10+**.

Clone the repository and install dependencies:
```bash
git clone https://github.com/mgarg8225-sudo/CaseLens.git
cd CaseLens
python -m pip install -r requirements.txt
```

### 2. Configure Environment Variables (`.env`)

Create a `.env` file in the project root. This file is git-ignored and never committed to version control:

```bash
# JWT Secret Key (required for access token issuance, 32+ characters)
JWT_SECRET_KEY=your-secure-random-32-byte-secret-key-here

# Database Path (defaults to data/pipeline.duckdb)
LEAKY_DB_PATH=data/pipeline.duckdb

# Demo User Passwords (read into memory on startup)
DEMO_VIEWER_PASSWORD=Viewer@CaseLens2026!
DEMO_ANALYST_PASSWORD=Analyst@CaseLens2026!
DEMO_ADMIN_PASSWORD=Admin@CaseLens2026!

# Optional: TinyFish Statutory Policy Benchmarking API Key
TINYFISH_API_KEY=your-tinyfish-api-key
```

> **Security Guarantee**: All credentials are **set via `.env`** or securely generated in memory on first run. Plaintext passwords, hashes, and tokens are never hardcoded or committed to version control.

### 3. Launch the Policy Dashboard

Run the Streamlit application:
```bash
streamlit run app/dashboard.py
```
*(Alternatively: `streamlit run dashboard.py` from the root directory).*

Navigate to `http://localhost:8501` in your browser.

### 4. Run the Automated Test Suite

Execute the full suite of unit, integration, and security tests:
```bash
python -m pytest
```
*(Or simply `pytest` if pytest is on your PATH).*

---

## Role-Based Access Control (RBAC)

CaseLens enforces server-side role-based access control across three user tiers:

| Tier / Role | Permitted Analytical Views | Restrictions & Security Boundaries |
|---|---|---|
| **viewer** | 📊 Executive Overview<br/>🔻 Funnel with Uncertainty Ranges<br/>⏱️ Duration Bottlenecks & Latency<br/>🔗 Linkage Quality & Diagnostics<br/>📜 Statutory Policy Insights (TinyFish) | **Aggregates only**. Strictly restricted to privatized, noise-injected macro statistics. Cannot access the Case Simulator, Human Review Queue, or Security Audit Log. |
| **analyst** | All **viewer** views<br/>⚖️ Before / After Policy Transformation<br/>🛡️ Survivor Privacy & DP Engine<br/>🧪 Case Simulator | Authorized to evaluate policy counterfactuals, explore privacy-budget trade-offs, and run scenarios in the Case Simulator. Cannot view the Review Queue or Security Audit Log. |
| **admin** | Full access to all views<br/>🧪 Case Simulator<br/>📋 Review Queue<br/>📜 Security Audit Log | Access to the human review queue (pseudonymized candidate pairs) and the immutable zero-PII security audit ledger. |

### Institutional Agency Roles

For multi-agency federation, CaseLens defines agency-specific institutional profiles:
- `police`: FIR records and investigation timelines
- `hospital`: Medico-legal examination records and evidence collection
- `forensic`: Forensic science lab analysis, DNA matching, and report dispatch
- `prosecutor`: Chargesheet filings and legal scrutiny
- `court`: District court proceedings, trial hearings, and verdict recording
- `linkage_reviewer`: Quarantined candidate-pair adjudication
- `auditor`: Independent inspection of Differential Privacy budgets and audit logs
- `public_viewer`: Public transparency portal with aggregate-only visibility

### Security Engine Specifications

- **Argon2id Password Hashing**: Memory-hard parameters ($m=64\text{ MiB}, t=2, p=1$). Plaintext passwords are never stored or logged.
- **Brute-Force Lockout Defense**: Enforces an automatic 15-minute lockout after 5 consecutive failed login attempts.
- **Short-Lived Signed JWTs**: Tokens expire in 30 minutes and are signed using HMAC-SHA256 with the secret loaded from `JWT_SECRET_KEY`.
- **Zero-Credential Audit Ledger**: Every login, logout, failed attempt, and page navigation is recorded in `data/app_audit_log.json`. Passwords, hashes, and tokens are cryptographically forbidden from entering the ledger.

---

## Judicial Case Simulator: "What Goes In vs What Comes Out"

The **Case Simulator** is an interactive experimentation environment designed for judges, prosecutors, and policy analysts to inspect the exact mathematical entity resolution pipeline on multi-agency records.

### How It Works

1. **Input (What Goes In)**: A bundle of institutional records arriving from police stations, hospitals, forensic labs, and courts with noisy real-world discrepancies (transliteration differences, phonetic typos, mismatched family names, missing records, delays).
2. **Entity Resolution Engine**: Records are compared pairwise using the Fellegi-Sunter model with Jaro-Winkler string similarity, soundex phonetics, and temporal proximity.
3. **Scoring & Classification**:
   - $\text{Match Probability} \ge 0.85$ ($\text{Weight} \ge +5.0$): **AUTO_ACCEPT** (records automatically linked into a single case).
   - $0.70 \le \text{Match Probability} < 0.85$: **HUMAN_REVIEW** (quarantined to the review queue for human verification).
   - $\text{Match Probability} < 0.70$: **REJECT** (treated as distinct cases).
4. **Output (What Comes Out)**: A unified case trajectory with total calendar duration, milestone status, delay flags, and detailed per-field log-odds evidence.

### Pre-Configured Synthetic Scenarios

| Scenario | Scenario Name | Agency Records | Challenge / Real-World Condition | Expected Pipeline Verdict |
|---|---|---|---|---|
| **A** | Perfect Clean Journey | Police + Hospital + FSL + Court | Clean matching names, identifiers, and dates | Auto-Accepted (Prob: 0.999) |
| **B** | Transliteration Variant | Police + Hospital + FSL | Hindi/English phonetic variations ("Sunita" vs "Suneeta") | Auto-Accepted (Prob: 0.985) |
| **C** | Minor Typo in Name | Police + Hospital | OCR / manual data entry typographical errors | Auto-Accepted (Prob: 0.970) |
| **D** | Father Name Discrepancy | Police + Court | Discrepant patronymics ("Ram Lal" vs "Ramesh Kumar") | Quarantined for Human Review |
| **E** | Missing Hospital Record | Police + FSL + Court | Institutional silo gap (medical exam bypassed/unrecorded) | Partial Journey (Flagged Silo Gap) |
| **F** | Long Delayed Hospital Record | Police + Hospital | Severe 9-month delay exceeding statutory guidelines | Delayed Journey (Flagged Bottleneck) |
| **G** | Different Individuals (Colliding Names) | Police + Police | Common names in same district with distinct ages & parentage | Correctly Rejected (Prob: 0.050) |
| **H** | Out-of-Order Agency Records | Court + Hospital + Police | Chronologically inverted data entry across agency databases | Correctly Ordered via Incident Date |

---

## Critical System Limitations & Guardrails

To prevent misuse and ensure responsible application in judicial and policy contexts, CaseLens operates under four fundamental constraints:

1. **Synthetic Data Only**
   All case records, victim and suspect names, dates, hospital MLC reports, and FIR numbers utilized in this application are **100% synthetic**. Data was generated using Faker and empirical distributions calibrated to public judicial statistics. No actual survivor or victim records are ever ingested, processed, or stored.

2. **Linkage Can Be Wrong**
   Probabilistic record linkage is fundamentally **statistical, not deterministic**. While Fellegi-Sunter with Jaro-Winkler indexing achieves high accuracy ($F_1 > 0.92$), false positives and false negatives can occur—particularly with sparse records, very common patronymics, or extreme data entry errors. The system is an analytical diagnostic tool, not an automated conviction engine.

3. **Aggregates Only for Policy & Public Views**
   Non-privileged users, public viewers, and policy analysts **never see individual survivor trajectories**. All reporting is strictly aggregated at the district or milestone level and masked with Laplace Differential Privacy noise ($\epsilon \in [0.1, 2.0]$) and $k$-anonymity suppression ($k \ge 5$).

4. **Human Review Required for Uncertain Cases**
   Automated linkage decisions are never made for borderline cases. Any candidate pair with a match probability between $0.70$ and $0.85$ is quarantined into a **Human Review Queue** (`data/review_queue.csv`). Authorized reviewers adjudicate these cases using pseudonymized IDs (with all direct identifiers stripped) before records can be linked into canonical journeys.

---

## Security & Secrets Management

- **Zero Plaintext Secrets**: `.env`, passwords, private keys, and session tokens are explicitly excluded in `.gitignore` and are strictly forbidden from git commits.
- **Runtime Integrity**: Passwords are read exclusively from environment variables or randomly generated in memory on startup.
- **Zero-PII Audit Ledger**: The application audit trail records actions (logins, lockouts, view navigations) without logging passwords, tokens, or personal identifiers.
