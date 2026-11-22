-- Supabase Migration: 001_initial_schema.sql
-- CaseLens Security & Persistent Storage Schema
-- RLS Policy: Row Level Security is ENABLED on ALL tables with NO public/anon/authenticated policies.
-- Access is strictly restricted to the FastAPI backend running with the service_role key.
-- Direct client access through Supabase anon key is completely blocked (default-deny).

-- ==============================================================================
-- 1. Profiles Table (Server-side RBAC)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS public.profiles (
    user_id UUID PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('viewer', 'analyst', 'admin')),
    full_name TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
-- No public/anon policies created. Only service_role can access.

-- ==============================================================================
-- 2. Datasets Table (Draft & Published Lifecycle)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS public.datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft', 'published')),
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    published_by TEXT,
    published_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ
);

ALTER TABLE public.datasets ENABLE ROW LEVEL SECURITY;
-- No public/anon policies created. Only service_role can access.

-- ==============================================================================
-- 3. Dataset Cases Table (Encrypted PII + Non-identifying Analysis Columns)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS public.dataset_cases (
    id BIGSERIAL PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES public.datasets(id) ON DELETE CASCADE,
    case_ref TEXT NOT NULL,
    payload_encrypted BYTEA NOT NULL,
    district TEXT,
    category TEXT,
    incident_date TEXT,
    fir_date TEXT,
    status TEXT,
    stage TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    CONSTRAINT uq_dataset_case_ref UNIQUE (dataset_id, case_ref)
);

CREATE INDEX IF NOT EXISTS idx_dataset_cases_dataset_id ON public.dataset_cases(dataset_id);
CREATE INDEX IF NOT EXISTS idx_dataset_cases_district ON public.dataset_cases(district);
CREATE INDEX IF NOT EXISTS idx_dataset_cases_case_ref ON public.dataset_cases(case_ref);

ALTER TABLE public.dataset_cases ENABLE ROW LEVEL SECURITY;
-- No public/anon policies created. Only service_role can access.

-- ==============================================================================
-- 4. Analysis Results Table (Linked Pairs, Timelines, Aggregates)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS public.analysis_results (
    dataset_id TEXT PRIMARY KEY REFERENCES public.datasets(id) ON DELETE CASCADE,
    linked_pairs JSONB,
    timelines JSONB,
    aggregates JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

ALTER TABLE public.analysis_results ENABLE ROW LEVEL SECURITY;
-- No public/anon policies created. Only service_role can access.

-- ==============================================================================
-- 5. Audit Log Table (Immutable Security & Access Ledger)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS public.audit_log (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    actor TEXT NOT NULL,
    role TEXT,
    action TEXT NOT NULL,
    detail JSONB
);

CREATE INDEX IF NOT EXISTS idx_audit_log_ts ON public.audit_log(ts DESC);

ALTER TABLE public.audit_log ENABLE ROW LEVEL SECURITY;
-- No public/anon policies created. Only service_role can access.

-- ==============================================================================
-- 6. Privacy Budget Table (Differential Privacy State & Atomic Allocation)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS public.privacy_budget (
    id TEXT PRIMARY KEY,
    epsilon_total DOUBLE PRECISION NOT NULL DEFAULT 10.0,
    epsilon_spent DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

ALTER TABLE public.privacy_budget ENABLE ROW LEVEL SECURITY;
-- No public/anon policies created. Only service_role can access.

-- Initial default global privacy budget record
INSERT INTO public.privacy_budget (id, epsilon_total, epsilon_spent, updated_at)
VALUES ('global', 10.0, 0.0, timezone('utc'::text, now()))
ON CONFLICT (id) DO NOTHING;

-- ==============================================================================
-- 7. Privacy Budget Ledger Table (Sequential Composition Audit Trail)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS public.privacy_budget_ledger (
    id BIGSERIAL PRIMARY KEY,
    budget_id TEXT NOT NULL REFERENCES public.privacy_budget(id) ON DELETE CASCADE,
    query_id TEXT NOT NULL,
    epsilon_cost DOUBLE PRECISION NOT NULL,
    delta_cost DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    actor TEXT,
    role TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_budget_ledger_timestamp ON public.privacy_budget_ledger(timestamp DESC);

ALTER TABLE public.privacy_budget_ledger ENABLE ROW LEVEL SECURITY;
-- No public/anon policies created. Only service_role can access.
