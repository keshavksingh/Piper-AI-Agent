"""Tests for shared.config — Config class with environment variable overrides."""

import importlib
import os
from unittest.mock import patch

import pytest


class TestConfigDefaults:
    """Verify default values when no env vars are set."""

    def test_default_llm_model(self):
        from shared.config import Config
        # Default should be claude-sonnet-4-20250514 unless env var overrides it
        assert isinstance(Config.LLM_MODEL, str)
        assert len(Config.LLM_MODEL) > 0

    def test_default_react_max_iterations(self):
        from shared.config import Config
        assert isinstance(Config.REACT_MAX_ITERATIONS, int)
        assert Config.REACT_MAX_ITERATIONS > 0

    def test_default_rate_limit(self):
        from shared.config import Config
        assert isinstance(Config.RATE_LIMIT_PER_MINUTE, int)
        assert Config.RATE_LIMIT_PER_MINUTE > 0

    def test_boolean_parsing(self):
        from shared.config import Config
        # These should be booleans, not strings
        assert isinstance(Config.REFLECTION_ENABLED, bool)
        assert isinstance(Config.GUARDRAILS_ENABLED, bool)
        assert isinstance(Config.PLANNING_ENABLED, bool)
        assert isinstance(Config.MULTI_AGENT_ENABLED, bool)
        assert isinstance(Config.REFLEXION_ENABLED, bool)
        assert isinstance(Config.TOOL_VALIDATION_ENABLED, bool)
        assert isinstance(Config.EVALUATION_STORAGE_ENABLED, bool)

    def test_numeric_types(self):
        from shared.config import Config
        assert isinstance(Config.EMBEDDING_DIMENSIONS, int)
        assert isinstance(Config.SESSION_TTL_SECONDS, int)
        assert isinstance(Config.INTENT_CONFIDENCE_THRESHOLD, float)
        assert isinstance(Config.REFLECTION_QUALITY_THRESHOLD, float)
        assert isinstance(Config.REFLEXION_INSIGHT_THRESHOLD, float)

    def test_service_addresses_are_strings(self):
        from shared.config import Config
        assert isinstance(Config.AGENT_SERVICE_ADDR, str)
        assert isinstance(Config.MEMORY_SERVICE_ADDR, str)
        assert isinstance(Config.LLM_SERVICE_ADDR, str)
        assert isinstance(Config.KNOWLEDGE_SERVICE_ADDR, str)
        assert isinstance(Config.TOOL_SERVICE_ADDR, str)
        assert isinstance(Config.RECOMMENDATION_SERVICE_ADDR, str)


class TestConfigEnvOverrides:
    """Verify env vars override defaults by re-importing the module.

    Config evaluates class attributes at import time from os.getenv(),
    so we must reload the module for env changes to take effect.
    """

    @patch.dict(os.environ, {"REACT_MAX_ITERATIONS": "15"})
    def test_numeric_override(self):
        import shared.config
        importlib.reload(shared.config)
        assert shared.config.Config.REACT_MAX_ITERATIONS == 15
        # Reload with original env to not affect other tests
        importlib.reload(shared.config)

    @patch.dict(os.environ, {"REFLECTION_ENABLED": "false"})
    def test_bool_override_false(self):
        import shared.config
        importlib.reload(shared.config)
        assert shared.config.Config.REFLECTION_ENABLED is False
        importlib.reload(shared.config)

    @patch.dict(os.environ, {"REFLECTION_ENABLED": "true"})
    def test_bool_override_true(self):
        import shared.config
        importlib.reload(shared.config)
        assert shared.config.Config.REFLECTION_ENABLED is True
        importlib.reload(shared.config)

    @patch.dict(os.environ, {"EMBEDDING_DIMENSIONS": "512"})
    def test_embedding_dimensions_override(self):
        import shared.config
        importlib.reload(shared.config)
        assert shared.config.Config.EMBEDDING_DIMENSIONS == 512
        importlib.reload(shared.config)

    @patch.dict(os.environ, {"JWT_EXPIRY_HOURS": "48"})
    def test_jwt_expiry_override(self):
        import shared.config
        importlib.reload(shared.config)
        assert shared.config.Config.JWT_EXPIRY_HOURS == 48
        importlib.reload(shared.config)


class TestConfigSafeParsing:
    """Verify _safe_int and _safe_float handle invalid env var values gracefully."""

    @patch.dict(os.environ, {"REACT_MAX_ITERATIONS": "not_a_number"})
    def test_invalid_int_falls_back_to_default(self):
        import shared.config
        importlib.reload(shared.config)
        # Should fall back to default (8) instead of crashing
        assert shared.config.Config.REACT_MAX_ITERATIONS == 8
        importlib.reload(shared.config)

    @patch.dict(os.environ, {"INTENT_CONFIDENCE_THRESHOLD": "abc"})
    def test_invalid_float_falls_back_to_default(self):
        import shared.config
        importlib.reload(shared.config)
        # Should fall back to default (0.8) instead of crashing
        assert shared.config.Config.INTENT_CONFIDENCE_THRESHOLD == 0.8
        importlib.reload(shared.config)

    @patch.dict(os.environ, {"RATE_LIMIT_PER_MINUTE": ""})
    def test_empty_string_int_falls_back(self):
        import shared.config
        importlib.reload(shared.config)
        assert shared.config.Config.RATE_LIMIT_PER_MINUTE == 30
        importlib.reload(shared.config)


class TestConfigBooleanEdgeCases:
    """Verify boolean parsing handles various string inputs."""

    @patch.dict(os.environ, {"GUARDRAILS_ENABLED": "True"})
    def test_capitalized_true(self):
        import shared.config
        importlib.reload(shared.config)
        assert shared.config.Config.GUARDRAILS_ENABLED is True
        importlib.reload(shared.config)

    @patch.dict(os.environ, {"GUARDRAILS_ENABLED": "FALSE"})
    def test_uppercase_false(self):
        import shared.config
        importlib.reload(shared.config)
        assert shared.config.Config.GUARDRAILS_ENABLED is False
        importlib.reload(shared.config)

    @patch.dict(os.environ, {"GUARDRAILS_ENABLED": "yes"})
    def test_non_true_string_is_false(self):
        """Any value other than 'true' (case-insensitive) should be False."""
        import shared.config
        importlib.reload(shared.config)
        assert shared.config.Config.GUARDRAILS_ENABLED is False
        importlib.reload(shared.config)
