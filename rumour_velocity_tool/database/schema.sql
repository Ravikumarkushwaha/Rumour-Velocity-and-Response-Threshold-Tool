-- SQLite schema for the Rumour Velocity and Response Threshold Tool.
-- Referential integrity is enforced (PRAGMA foreign_keys = ON is set per connection).

CREATE TABLE IF NOT EXISTS claims (
    claim_id                 TEXT PRIMARY KEY,
    description              TEXT NOT NULL CHECK (length(trim(description)) > 0),
    category                 TEXT NOT NULL CHECK (category IN
        ('Recruitment notification','Incident','Unit-related claim','Family-related claim','Other')),
    first_observed           TEXT NOT NULL,
    verification_status      TEXT NOT NULL CHECK (verification_status IN
        ('Verified true','Verified false','Partly true','Unverified','Conflicting evidence')),
    created_at               TEXT NOT NULL,
    -- additional analyst-supplied context used by the decision safeguards
    public_impact            TEXT NOT NULL DEFAULT 'Low' CHECK (public_impact IN ('Low','Medium','High')),
    public_safety_flag       INTEGER NOT NULL DEFAULT 0 CHECK (public_safety_flag IN (0,1)),
    credible_allegation_flag INTEGER NOT NULL DEFAULT 0 CHECK (credible_allegation_flag IN (0,1)),
    analyst_remarks          TEXT,
    is_synthetic             INTEGER NOT NULL DEFAULT 0 CHECK (is_synthetic IN (0,1)),
    review_due_at            TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id             TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    timestamp            TEXT NOT NULL,
    mention_count        INTEGER NOT NULL CHECK (mention_count >= 0),   -- cumulative, OBSERVED
    engagement_count     INTEGER CHECK (engagement_count IS NULL OR engagement_count >= 0),
    estimated_reach      REAL    CHECK (estimated_reach IS NULL OR estimated_reach >= 0), -- ESTIMATED, may be NULL
    source_type          TEXT,
    independent_sources  INTEGER CHECK (independent_sources IS NULL OR independent_sources >= 0),
    reach_basis          TEXT CHECK (reach_basis IS NULL OR reach_basis IN ('estimated','observed')),
    source_reference     TEXT,
    source_reliability   TEXT NOT NULL DEFAULT 'Unassessed' CHECK
        (source_reliability IN ('Unassessed','Low','Moderate','High')),
    UNIQUE (claim_id, timestamp)          -- duplicate observations are not independent evidence
);

CREATE TABLE IF NOT EXISTS assessments (
    assessment_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id                 TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    velocity                 REAL,
    growth                   REAL,
    acceleration             REAL,
    reach                    REAL,
    amplification_risk       TEXT,
    response_pressure_score  REAL CHECK (response_pressure_score IS NULL OR
                                         (response_pressure_score >= 0 AND response_pressure_score <= 1)),
    recommendation           TEXT NOT NULL CHECK (recommendation IN
        ('REMAIN SILENT','MONITOR','RECOMMEND RESPONSE REVIEW','INSUFFICIENT DATA / HUMAN VERIFICATION REQUIRED')),
    justification            TEXT,
    assessed_at              TEXT NOT NULL,
    -- additional audit fields
    trend_label              TEXT,
    lifecycle_stage          TEXT,
    escalation               INTEGER NOT NULL DEFAULT 0 CHECK (escalation IN (0,1)),
    human_review_status      TEXT,
    observations_used        INTEGER,
    config_snapshot          TEXT,
    hardening_index          REAL,
    hardening_level          TEXT
);

CREATE TABLE IF NOT EXISTS decision_history (
    decision_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id                 TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    previous_recommendation  TEXT,
    new_recommendation       TEXT NOT NULL,
    reason_for_change        TEXT,
    decision_timestamp       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_obs_claim_ts ON observations(claim_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_assess_claim ON assessments(claim_id, assessed_at);
CREATE INDEX IF NOT EXISTS idx_hist_claim ON decision_history(claim_id, decision_timestamp);

CREATE TABLE IF NOT EXISTS audit_events (
    audit_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id       TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    event_type     TEXT NOT NULL,
    record_type    TEXT NOT NULL,
    record_id      TEXT,
    field_names    TEXT NOT NULL DEFAULT '[]',
    detail         TEXT NOT NULL DEFAULT '',
    occurred_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_claim_time ON audit_events(claim_id, occurred_at);
