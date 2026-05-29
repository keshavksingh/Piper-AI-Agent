"""Centralized configuration for all Piper services."""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

_config_log = logging.getLogger("piper.config")


def _safe_int(env_var: str, default: int) -> int:
    """Parse an env var as int, returning default on failure."""
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        _config_log.warning("Invalid integer for %s=%r, using default %d", env_var, raw, default)
        return default


def _safe_float(env_var: str, default: float) -> float:
    """Parse an env var as float, returning default on failure."""
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        _config_log.warning("Invalid float for %s=%r, using default %s", env_var, raw, default)
        return default


class Config:
    # Anthropic Claude
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    # Voyage AI (Anthropic-recommended embedding provider)
    VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "voyage-3")
    EMBEDDING_DIMENSIONS: int = _safe_int("EMBEDDING_DIMENSIONS", 1024)

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    SESSION_TTL_SECONDS: int = _safe_int("SESSION_TTL_SECONDS", 1800)

    # PostgreSQL (relational + pgvector)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://piper:piper@postgres:5432/piper"
    )

    # TimescaleDB (immutable episodic memory + audit trail)
    TIMESCALEDB_URL: str = os.getenv(
        "TIMESCALEDB_URL", "postgresql://piper:piper@timescaledb:5432/piper_ts"
    )

    # Agent
    REACT_MAX_ITERATIONS: int = _safe_int("REACT_MAX_ITERATIONS", 8)
    REACT_TIMEOUT_SECONDS: int = _safe_int("REACT_TIMEOUT_SECONDS", 120)
    INTENT_CONFIDENCE_THRESHOLD: float = _safe_float("INTENT_CONFIDENCE_THRESHOLD", 0.8)
    DOMAIN_RELEVANCE_THRESHOLD: float = _safe_float("DOMAIN_RELEVANCE_THRESHOLD", 0.5)

    # Memory Context
    MEMORY_CONTEXT_MAX_TURNS: int = _safe_int("MEMORY_CONTEXT_MAX_TURNS", 10)
    MEMORY_CONTEXT_TRUNCATE_LENGTH: int = _safe_int("MEMORY_CONTEXT_TRUNCATE_LENGTH", 200)

    # Query Rewriting (conversational context resolution)
    QUERY_REWRITE_ENABLED: bool = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() == "true"
    QUERY_REWRITE_TEMPERATURE: float = _safe_float("QUERY_REWRITE_TEMPERATURE", 0.1)
    QUERY_REWRITE_MAX_TOKENS: int = _safe_int("QUERY_REWRITE_MAX_TOKENS", 128)

    # Reflection Agent
    REFLECTION_ENABLED: bool = os.getenv("REFLECTION_ENABLED", "true").lower() == "true"
    REFLECTION_MAX_ITERATIONS: int = _safe_int("REFLECTION_MAX_ITERATIONS", 2)
    REFLECTION_QUALITY_THRESHOLD: float = _safe_float("REFLECTION_QUALITY_THRESHOLD", 0.75)
    REFLECTION_TEMPERATURE: float = _safe_float("REFLECTION_TEMPERATURE", 0.2)
    REFLECTION_MAX_TOKENS: int = _safe_int("REFLECTION_MAX_TOKENS", 512)

    # Reflexion Agent (persistent learning)
    REFLEXION_ENABLED: bool = os.getenv("REFLEXION_ENABLED", "true").lower() == "true"
    REFLEXION_INSIGHT_THRESHOLD: float = _safe_float("REFLEXION_INSIGHT_THRESHOLD", 0.7)
    REFLEXION_MAX_INSIGHTS_PER_QUERY: int = _safe_int("REFLEXION_MAX_INSIGHTS_PER_QUERY", 3)

    # Tool Validation
    TOOL_VALIDATION_ENABLED: bool = os.getenv("TOOL_VALIDATION_ENABLED", "true").lower() == "true"

    # Planning Layer
    PLANNING_ENABLED: bool = os.getenv("PLANNING_ENABLED", "true").lower() == "true"
    PLANNING_TEMPERATURE: float = _safe_float("PLANNING_TEMPERATURE", 0.2)
    PLANNING_MAX_TOKENS: int = _safe_int("PLANNING_MAX_TOKENS", 512)

    # Multi-Agent Orchestration
    MULTI_AGENT_ENABLED: bool = os.getenv("MULTI_AGENT_ENABLED", "true").lower() == "true"
    MULTI_AGENT_MAX_AGENTS: int = _safe_int("MULTI_AGENT_MAX_AGENTS", 3)

    # Guardrails (Input + Output)
    GUARDRAILS_ENABLED: bool = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
    GUARDRAILS_MAX_QUERY_LENGTH: int = _safe_int("GUARDRAILS_MAX_QUERY_LENGTH", 2000)

    # Evaluation Storage
    EVALUATION_STORAGE_ENABLED: bool = os.getenv("EVALUATION_STORAGE_ENABLED", "true").lower() == "true"

    # Service addresses
    AGENT_SERVICE_ADDR: str = os.getenv("AGENT_SERVICE_ADDR", "agent_service:50054")
    MEMORY_SERVICE_ADDR: str = os.getenv("MEMORY_SERVICE_ADDR", "memory_service:50055")
    LLM_SERVICE_ADDR: str = os.getenv("LLM_SERVICE_ADDR", "llm_service:50053")
    KNOWLEDGE_SERVICE_ADDR: str = os.getenv(
        "KNOWLEDGE_SERVICE_ADDR", "knowledge_service:50052"
    )
    TOOL_SERVICE_ADDR: str = os.getenv("TOOL_SERVICE_ADDR", "tool_service:50056")
    RECOMMENDATION_SERVICE_ADDR: str = os.getenv(
        "RECOMMENDATION_SERVICE_ADDR", "recommendation_service:50057"
    )

    # JWT / Authentication
    JWT_SECRET: str = os.getenv("JWT_SECRET", "piper-dev-secret-change-in-prod")
    JWT_EXPIRY_HOURS: int = _safe_int("JWT_EXPIRY_HOURS", 24)

    # TLS certificates
    TLS_CA_CERT: str = os.getenv("TLS_CA_CERT", "/app/certs/ca.pem")
    TLS_SERVER_CERT: str = os.getenv("TLS_SERVER_CERT", "/app/certs/server.pem")
    TLS_SERVER_KEY: str = os.getenv("TLS_SERVER_KEY", "/app/certs/server-key.pem")

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = _safe_int("RATE_LIMIT_PER_MINUTE", 30)

    # LLM defaults
    DEFAULT_TEMPERATURE: float = 0.3
    DEFAULT_MAX_TOKENS: int = 1024
    INTENT_TEMPERATURE: float = 0.1
    INTENT_MAX_TOKENS: int = 256
