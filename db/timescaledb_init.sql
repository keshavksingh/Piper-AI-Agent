-- Piper AI Agent — TimescaleDB Schema
-- Immutable time-series store for episodic memory and session audit trails.
-- All tables are append-only hypertables. No UPDATE or DELETE permitted.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================
-- Episodic Memories (immutable hypertable)
-- ============================================
-- Summarised snapshots of completed sessions.
-- Immutable: once a session summary is written it is never modified.
-- Partitioned by created_at (7-day chunks).

CREATE TABLE episodic_memories (
    id              UUID        NOT NULL DEFAULT uuid_generate_v4(),
    customer_id     VARCHAR(255) NOT NULL,
    session_id      UUID,
    event_type      VARCHAR(50) NOT NULL DEFAULT 'session_summary'
                        CHECK (event_type IN (
                            'session_summary',
                            'topic_snapshot',
                            'resolution_record',
                            'preference_learned',
                            'reflexion_insight',
                            'evaluation_record'
                        )),
    summary         TEXT        NOT NULL,
    key_topics      TEXT[]      DEFAULT '{}',
    resolution_status VARCHAR(20) DEFAULT 'resolved'
                        CHECK (resolution_status IN ('resolved', 'unresolved', 'partial')),
    metadata        JSONB       DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable('episodic_memories', 'created_at',
    chunk_time_interval => INTERVAL '7 days'
);

CREATE INDEX idx_episodic_customer_time
    ON episodic_memories (customer_id, created_at DESC);
CREATE INDEX idx_episodic_session
    ON episodic_memories (session_id, created_at DESC);
CREATE INDEX idx_episodic_event_type
    ON episodic_memories (event_type, created_at DESC);

-- ============================================
-- Session Audit Trail (immutable hypertable)
-- ============================================
-- Every significant event during a session is recorded here.
-- Enables full session replay and compliance auditing.
-- Immutable: events are never modified or deleted.
-- Partitioned by event_time (1-day chunks).

CREATE TABLE session_audit_trail (
    id              UUID        NOT NULL DEFAULT uuid_generate_v4(),
    session_id      UUID        NOT NULL,
    customer_id     VARCHAR(255) NOT NULL,
    event_type      VARCHAR(50) NOT NULL
                        CHECK (event_type IN (
                            'session_created',
                            'session_resumed',
                            'turn_user',
                            'turn_assistant',
                            'intent_classified',
                            'react_thought',
                            'react_action',
                            'react_observation',
                            'tool_executed',
                            'clarification_sent',
                            'clarification_received',
                            'response_completed',
                            'recommendation_served',
                            'error_occurred',
                            'session_closed'
                        )),
    event_data      JSONB       NOT NULL DEFAULT '{}',
    event_time      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, event_time)
);

SELECT create_hypertable('session_audit_trail', 'event_time',
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX idx_audit_session_time
    ON session_audit_trail (session_id, event_time ASC);
CREATE INDEX idx_audit_customer_time
    ON session_audit_trail (customer_id, event_time DESC);
CREATE INDEX idx_audit_event_type
    ON session_audit_trail (event_type, event_time DESC);

-- ============================================
-- Immutability enforcement
-- ============================================
-- Trigger functions that prevent UPDATE and DELETE on both tables.

CREATE OR REPLACE FUNCTION prevent_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Table % is immutable. UPDATE and DELETE are not permitted.', TG_TABLE_NAME;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER immutable_episodic_memories
    BEFORE UPDATE OR DELETE ON episodic_memories
    FOR EACH ROW EXECUTE FUNCTION prevent_mutation();

CREATE TRIGGER immutable_session_audit_trail
    BEFORE UPDATE OR DELETE ON session_audit_trail
    FOR EACH ROW EXECUTE FUNCTION prevent_mutation();

-- ============================================
-- Continuous aggregate: daily session stats
-- ============================================
-- Materialised view refreshed automatically for dashboards / analytics.

CREATE MATERIALIZED VIEW daily_session_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', event_time)    AS day,
    customer_id,
    COUNT(*) FILTER (WHERE event_type = 'session_created')       AS sessions_started,
    COUNT(*) FILTER (WHERE event_type = 'turn_user')             AS user_messages,
    COUNT(*) FILTER (WHERE event_type = 'tool_executed')         AS tool_executions,
    COUNT(*) FILTER (WHERE event_type = 'clarification_sent')    AS clarifications,
    COUNT(*) FILTER (WHERE event_type = 'error_occurred')        AS errors
FROM session_audit_trail
GROUP BY day, customer_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('daily_session_stats',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- ============================================
-- Retention policy (optional, 90-day default)
-- ============================================
-- Audit trail chunks older than 90 days are automatically dropped.
-- Episodic memories are kept indefinitely (no retention policy).

SELECT add_retention_policy('session_audit_trail', INTERVAL '90 days');
