# CaseLens: The Leaky Pipeline
### Survivor Confidentiality & Justice System Policy Intelligence

CaseLens reconstructs the longitudinal journey of violence-against-women cases across fragmented institutional silos (Police FIR, Hospital MLC, Forensic Science Laboratories, Prosecution, and District Courts) to diagnose systemic attrition points without compromising survivor confidentiality.

---

## Quickstart

### 1. Environment Setup
CaseLens requires Python 3.10+ and uses DuckDB with Argon2id password hashing and short-lived JWT tokens.
Configure your `.env` file (which is git-ignored and never committed):

```bash
# JWT Secret Key (required for access token issuance)
JWT_SECRET_KEY=your-secure-random-32-byte-secret-key-here

# Demo User Passwords (read from environment on startup)
DEMO_VIEWER_PASSWORD=set via .env
DEMO_ANALYST_PASSWORD=set via .env
DEMO_ADMIN_PASSWORD=set via .env

# Optional: TinyFish Statutory Policy Benchmarking Key
TINYFISH_API_KEY=your-tinyfish-api-key
```

> **Security Guarantee**: Demo user credentials are **set via `.env`** or securely generated in memory on first run and printed once. Passwords, hashes, and tokens are never hardcoded or committed to version control.

### 2. Launch the Policy Dashboard
```bash
streamlit run app/dashboard.py
```

### 3. Run the Automated Test Suite
```bash
python -m pytest
```

---

## Role-Based Access Control (RBAC)

CaseLens enforces server-side role-based access control across three tiers:

| Role | Permitted Analytical Views | Restrictions |
|---|---|---|
| **viewer** | 📊 Executive Overview<br/>🔻 Funnel with Uncertainty Ranges<br/>⏱️ Duration Bottlenecks & Latency<br/>🔗 Linkage Quality & Evaluation<br/>📜 Statutory Policy Insights (TinyFish) | Strictly restricted to aggregate statistics. Cannot access Case Simulator, Review Queue, or Audit Log. |
| **analyst** | All **viewer** pages<br/>⚖️ Before / After Policy Transformation<br/>🛡️ Survivor Privacy & DP Engine<br/>🧪 Case Simulator | Cannot access admin Review Queue or Audit Log. |
| **admin** | Full access to all analytical pages<br/>🧪 Case Simulator<br/>📋 Review Queue<br/>📜 Security Audit Log | Access to human adjudication review queue (pseudonymized IDs only) and immutable zero-PII security audit ledger. |

---

## Security & Privacy Architecture

- **Argon2id Hashing**: Passwords hashed with RFC-compliant memory-hard parameters.
- **Account Lockout**: Automatically locks an account for 15 minutes after 5 consecutive failed login attempts to prevent brute-force attacks.
- **Short-Lived JWT Tokens**: Signed with `HS256` loaded from `JWT_SECRET_KEY` with 30-minute expiration.
- **Zero-Credential Audit Ledger**: Every login, logout, failed attempt, and page visit is logged to `data/app_audit_log.json`. Passwords, hashes, and tokens are cryptographically forbidden from all log entries.
- **Differential Privacy & k-Anonymity**: All published tables are protected with Laplace Mechanism noise and small-cell suppression ($k \ge 5$).
- **Pseudonymized Review Queue**: Uncertain candidate pairs in `data/review_queue.csv` are strictly stripped of names, relatives, and addresses.
