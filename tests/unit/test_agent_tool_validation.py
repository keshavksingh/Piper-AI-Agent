"""Tests for agent_service tool parameter and result validation."""

import json
from unittest.mock import patch

import pytest

from agent_service.server import AgentServiceServicer


@pytest.fixture
def servicer():
    return AgentServiceServicer()


@pytest.fixture
def product_search_schema():
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results"},
        },
        "required": ["query"],
    }


@pytest.fixture
def compare_schema():
    return {
        "type": "object",
        "properties": {
            "product_names": {"type": "array", "description": "List of product names"},
        },
        "required": ["product_names"],
    }


class TestValidateToolParams:
    """Tests for _validate_tool_params."""

    def test_valid_params(self, servicer, product_search_schema):
        params = json.dumps({"query": "wireless headphones", "top_k": 5})
        is_valid, validated, error = servicer._validate_tool_params(
            "product_search", params, product_search_schema
        )
        assert is_valid is True
        assert error == ""

    def test_invalid_json(self, servicer, product_search_schema):
        params = "not valid json {"
        is_valid, validated, error = servicer._validate_tool_params(
            "product_search", params, product_search_schema
        )
        assert is_valid is False
        assert "Invalid JSON" in error

    def test_missing_required_field(self, servicer, product_search_schema):
        params = json.dumps({"top_k": 5})
        is_valid, validated, error = servicer._validate_tool_params(
            "product_search", params, product_search_schema
        )
        assert is_valid is False
        assert "Missing required field" in error
        assert "query" in error

    def test_empty_required_field(self, servicer, product_search_schema):
        params = json.dumps({"query": "   "})
        is_valid, validated, error = servicer._validate_tool_params(
            "product_search", params, product_search_schema
        )
        assert is_valid is False
        assert "empty" in error

    def test_wrong_type_string(self, servicer, product_search_schema):
        params = json.dumps({"query": 123})
        is_valid, validated, error = servicer._validate_tool_params(
            "product_search", params, product_search_schema
        )
        assert is_valid is False
        assert "should be a string" in error

    def test_wrong_type_integer(self, servicer, product_search_schema):
        params = json.dumps({"query": "test", "top_k": "five"})
        is_valid, validated, error = servicer._validate_tool_params(
            "product_search", params, product_search_schema
        )
        assert is_valid is False
        assert "should be an integer" in error

    def test_wrong_type_array(self, servicer, compare_schema):
        params = json.dumps({"product_names": "Widget A"})
        is_valid, validated, error = servicer._validate_tool_params(
            "product_compare", params, compare_schema
        )
        assert is_valid is False
        assert "should be an array" in error

    def test_no_schema_passes(self, servicer):
        params = json.dumps({"anything": "goes"})
        is_valid, validated, error = servicer._validate_tool_params(
            "unknown_tool", params, {}
        )
        assert is_valid is True

    def test_disabled_flag_skips_validation(self, servicer, product_search_schema, monkeypatch):
        monkeypatch.setattr("shared.config.Config.TOOL_VALIDATION_ENABLED", False)
        params = "not json at all"
        is_valid, validated, error = servicer._validate_tool_params(
            "product_search", params, product_search_schema
        )
        assert is_valid is True


class TestValidateToolResult:
    """Tests for _validate_tool_result."""

    def test_error_key_detected(self, servicer):
        result = json.dumps({"error": "Product not found"})
        is_valid, enriched = servicer._validate_tool_result("product_search", result)
        assert is_valid is False
        assert "tool returned an error" in enriched

    def test_empty_results(self, servicer):
        result = json.dumps({"results": [], "count": 0})
        is_valid, enriched = servicer._validate_tool_result("product_search", result)
        assert is_valid is False
        assert "No results found" in enriched

    def test_valid_result_passes(self, servicer):
        result = json.dumps({
            "results": [{"product_name": "Widget", "price": 9.99}],
            "count": 1,
        })
        is_valid, enriched = servicer._validate_tool_result("product_search", result)
        assert is_valid is True
        assert enriched == result

    def test_non_json_passes_through(self, servicer):
        result = "plain text result"
        is_valid, enriched = servicer._validate_tool_result("some_tool", result)
        assert is_valid is True
        assert enriched == result

    def test_disabled_flag_skips(self, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.TOOL_VALIDATION_ENABLED", False)
        result = json.dumps({"error": "bad"})
        is_valid, enriched = servicer._validate_tool_result("tool", result)
        assert is_valid is True
