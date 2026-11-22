-- Supabase Migration: 002_schema_migrations.sql
-- Tracking table for applied database migrations with checksums and timestamps.
-- RLS Policy: Row Level Security is ENABLED with NO public access (default-deny).

CREATE TABLE IF NOT EXISTS public.schema_migrations (
    version TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    checksum TEXT
);

ALTER TABLE public.schema_migrations ENABLE ROW LEVEL SECURITY;
-- No public/anon policies created. Only service_role can access.
