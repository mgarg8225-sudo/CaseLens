-- Supabase Migration: 003_privacy_budget_integrity.sql
-- CaseLens Phase 3B Stage 3: Privacy Budget Integrity & Persistent Delta Accounting
-- RLS Policy: Row Level Security is ENABLED on ALL tables with NO public/anon/authenticated policies.
-- Access is strictly restricted to the FastAPI backend running with the service_role key.
-- Direct client access through Supabase anon key is completely blocked (default-deny).

-- ==============================================================================
-- 1. Delta Accounting Columns on Privacy Budget
-- ==============================================================================
ALTER TABLE public.privacy_budget
    ADD COLUMN IF NOT EXISTS delta_total DOUBLE PRECISION NOT NULL DEFAULT 0.0001,
    ADD COLUMN IF NOT EXISTS delta_spent DOUBLE PRECISION NOT NULL DEFAULT 0.0;

-- ==============================================================================
-- 2. Idempotency Constraint on Privacy Budget Ledger
-- ==============================================================================
ALTER TABLE public.privacy_budget_ledger
    ADD COLUMN IF NOT EXISTS idempotency_key TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_idx_budget_ledger_idempotency
    ON public.privacy_budget_ledger (budget_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- ==============================================================================
-- 3. Non-Destructive Ledger Foreign Key Protection (Prevent Cascading Deletions)
-- ==============================================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'privacy_budget_ledger_budget_id_fkey'
          AND table_name = 'privacy_budget_ledger'
    ) THEN
        ALTER TABLE public.privacy_budget_ledger
            DROP CONSTRAINT privacy_budget_ledger_budget_id_fkey;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'fk_privacy_budget_ledger_budget_id'
          AND table_name = 'privacy_budget_ledger'
    ) THEN
        ALTER TABLE public.privacy_budget_ledger
            ADD CONSTRAINT fk_privacy_budget_ledger_budget_id
            FOREIGN KEY (budget_id) REFERENCES public.privacy_budget(id)
            ON DELETE RESTRICT;
    END IF;
END $$;

-- ==============================================================================
-- 4. Persistent Administrative Reset History Table
-- ==============================================================================
CREATE TABLE IF NOT EXISTS public.privacy_budget_resets (
    id BIGSERIAL PRIMARY KEY,
    budget_id TEXT NOT NULL REFERENCES public.privacy_budget(id) ON DELETE RESTRICT,
    actor TEXT NOT NULL,
    role TEXT NOT NULL,
    previous_epsilon_spent DOUBLE PRECISION NOT NULL,
    previous_delta_spent DOUBLE PRECISION NOT NULL,
    new_epsilon_total DOUBLE PRECISION NOT NULL,
    new_delta_total DOUBLE PRECISION NOT NULL,
    reason TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_budget_resets_timestamp
    ON public.privacy_budget_resets (timestamp DESC);

ALTER TABLE public.privacy_budget_resets ENABLE ROW LEVEL SECURITY;
-- No public/anon policies created. Only service_role can access.

-- ==============================================================================
-- 5. Immutability Protection for Privacy Budget Ledger (Prevent UPDATE/DELETE)
-- ==============================================================================
CREATE OR REPLACE RULE privacy_budget_ledger_no_update AS
    ON UPDATE TO public.privacy_budget_ledger DO INSTEAD NOTHING;

CREATE OR REPLACE RULE privacy_budget_ledger_no_delete AS
    ON DELETE TO public.privacy_budget_ledger DO INSTEAD NOTHING;
