"""Shared pytest fixtures for Piper AI Agent test suite."""

import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Mock PostgreSQL ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_pg_conn():
    """Mock psycopg2 connection with cursor context manager.

    The cursor supports both `with conn.cursor() as cur:` (context manager)
    and `conn.cursor()` (direct) usage patterns.
    """
    conn = MagicMock()
    cursor = MagicMock()
    # Support context manager pattern: with conn.cursor() as cur:
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return conn, cursor


@pytest.fixture
def mock_pg_conn_factory(mock_pg_conn):
    """Return a factory that always yields the same mock conn."""
    conn, cursor = mock_pg_conn

    def factory():
        return conn

    return factory, conn, cursor


# ── Mock Redis ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_redis():
    """Mock Redis client with common operations."""
    r = MagicMock()
    r.get.return_value = None
    r.setex.return_value = True
    r.expire.return_value = True
    r.rpush.return_value = 1
    r.lrange.return_value = []
    r.delete.return_value = 1
    return r


# ── Mock gRPC Context ────────────────────────────────────────────────────────


@pytest.fixture
def mock_grpc_context():
    """Mock gRPC context with set_code and set_details."""
    ctx = MagicMock()
    ctx.set_code = MagicMock()
    ctx.set_details = MagicMock()
    return ctx


# ── Config Overrides ─────────────────────────────────────────────────────────


@pytest.fixture
def disable_all_features(monkeypatch):
    """Disable all optional agent features for isolated testing."""
    monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", False)
    monkeypatch.setattr("shared.config.Config.REFLEXION_ENABLED", False)
    monkeypatch.setattr("shared.config.Config.PLANNING_ENABLED", False)
    monkeypatch.setattr("shared.config.Config.MULTI_AGENT_ENABLED", False)
    monkeypatch.setattr("shared.config.Config.GUARDRAILS_ENABLED", False)
    monkeypatch.setattr("shared.config.Config.EVALUATION_STORAGE_ENABLED", False)
    monkeypatch.setattr("shared.config.Config.TOOL_VALIDATION_ENABLED", False)


# ── Sample Data ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_tool_definitions():
    """Realistic tool definition data matching DB schema."""
    return [
        {
            "name": "product_search",
            "description": "Search products by semantic similarity",
            "parameter_schema": json.dumps({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results"},
                },
                "required": ["query"],
            }),
        },
        {
            "name": "price_lookup",
            "description": "Look up prices by product name or price range",
            "parameter_schema": json.dumps({
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "min_price": {"type": "number"},
                    "max_price": {"type": "number"},
                },
                "required": [],
            }),
        },
        {
            "name": "warranty_check",
            "description": "Check warranty info for a product",
            "parameter_schema": json.dumps({
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Product name"},
                },
                "required": ["product_name"],
            }),
        },
        {
            "name": "product_compare",
            "description": "Compare products side by side",
            "parameter_schema": json.dumps({
                "type": "object",
                "properties": {
                    "product_names": {"type": "array", "description": "List of product names"},
                },
                "required": ["product_names"],
            }),
        },
    ]


@pytest.fixture
def sample_products():
    """Product dicts matching DB schema."""
    return [
        {
            "id": "prod-001",
            "product_name": "UltraWidget Pro",
            "description": "A premium widget with advanced features",
            "price": 299.99,
            "manufacturing_date": "2024-01-15",
            "warranty_months": 24,
        },
        {
            "id": "prod-002",
            "product_name": "BasicWidget",
            "description": "An affordable entry-level widget",
            "price": 49.99,
            "manufacturing_date": "2024-03-01",
            "warranty_months": 12,
        },
        {
            "id": "prod-003",
            "product_name": "MegaWidget X",
            "description": "Industrial-grade widget for professional use",
            "price": 899.99,
            "manufacturing_date": "2024-06-15",
            "warranty_months": 36,
        },
    ]
