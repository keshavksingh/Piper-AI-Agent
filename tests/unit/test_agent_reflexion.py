"""Tests for agent_service reflexion — persistent learning fetch and store."""

import json
from unittest.mock import patch, MagicMock

import pytest

from agent_service.server import AgentServiceServicer


@pytest.fixture
def servicer():
    return AgentServiceServicer()


class TestGetReflexionInsights:
    """Tests for _get_reflexion_insights."""

    @patch("agent_service.server.get_memory_stub")
    def test_returns_formatted_context(self, mock_get_mem, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLEXION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLEXION_MAX_INSIGHTS_PER_QUERY", 3)

        stub = MagicMock()
        mock_get_mem.return_value = stub

        memory = MagicMock()
        memory.summary = "Always include price when answering product queries"
        memory.key_topics = ["price", "product"]
        memory.metadata = json.dumps({"intent": "product_inquiry"})

        stub.GetEpisodicMemories.return_value = MagicMock(memories=[memory])

        result = servicer._get_reflexion_insights("cust-1", "product_inquiry", "show me products")
        assert "Learnings from past interactions" in result
        assert "Always include price" in result

    @patch("agent_service.server.get_memory_stub")
    def test_intent_filter(self, mock_get_mem, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLEXION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLEXION_MAX_INSIGHTS_PER_QUERY", 3)

        stub = MagicMock()
        mock_get_mem.return_value = stub

        # Memory with matching intent
        mem1 = MagicMock()
        mem1.summary = "Relevant insight"
        mem1.key_topics = ["unrelated"]
        mem1.metadata = json.dumps({"intent": "price_check"})

        # Memory with non-matching intent
        mem2 = MagicMock()
        mem2.summary = "Other insight"
        mem2.key_topics = ["other"]
        mem2.metadata = json.dumps({"intent": "warranty_question"})

        stub.GetEpisodicMemories.return_value = MagicMock(memories=[mem1, mem2])

        result = servicer._get_reflexion_insights("cust-1", "price_check", "how much")
        assert "Relevant insight" in result

    @patch("agent_service.server.get_memory_stub")
    def test_topic_filter(self, mock_get_mem, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLEXION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLEXION_MAX_INSIGHTS_PER_QUERY", 3)

        stub = MagicMock()
        mock_get_mem.return_value = stub

        mem = MagicMock()
        mem.summary = "Topic-matched insight"
        mem.key_topics = ["warranty"]
        mem.metadata = json.dumps({"intent": "other"})

        stub.GetEpisodicMemories.return_value = MagicMock(memories=[mem])

        # "warranty" appears both in key_topics and query
        result = servicer._get_reflexion_insights("cust-1", "general", "what about warranty")
        assert "Topic-matched insight" in result

    @patch("agent_service.server.get_memory_stub")
    def test_fallback_to_recent(self, mock_get_mem, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLEXION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLEXION_MAX_INSIGHTS_PER_QUERY", 2)

        stub = MagicMock()
        mock_get_mem.return_value = stub

        # Neither intent nor topic matches
        mem = MagicMock()
        mem.summary = "Fallback insight"
        mem.key_topics = ["xyz"]
        mem.metadata = json.dumps({"intent": "abc"})

        stub.GetEpisodicMemories.return_value = MagicMock(memories=[mem])

        result = servicer._get_reflexion_insights("cust-1", "none", "unrelated query")
        assert "Fallback insight" in result

    @patch("agent_service.server.get_memory_stub")
    def test_empty_memories(self, mock_get_mem, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLEXION_ENABLED", True)

        stub = MagicMock()
        mock_get_mem.return_value = stub
        stub.GetEpisodicMemories.return_value = MagicMock(memories=[])

        result = servicer._get_reflexion_insights("cust-1", "product_inquiry", "test")
        assert result == ""

    def test_disabled_returns_empty(self, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLEXION_ENABLED", False)
        result = servicer._get_reflexion_insights("cust-1", "product_inquiry", "test")
        assert result == ""


class TestMaybeStoreReflexionInsight:
    """Tests for _maybe_store_reflexion_insight."""

    def test_below_threshold_triggers_store(self, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLEXION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLEXION_INSIGHT_THRESHOLD", 0.7)

        with patch("agent_service.server.get_llm_stub") as mock_llm, \
             patch("agent_service.server.get_memory_stub") as mock_mem:
            llm_stub = MagicMock()
            mock_llm.return_value = llm_stub
            llm_stub.GenerateAnswer.return_value = MagicMock(
                completion=json.dumps({
                    "query_pattern": "price query",
                    "failure_reason": "Missing details",
                    "suggested_improvement": "Include product name",
                    "key_topics": ["price"],
                })
            )
            mem_stub = MagicMock()
            mock_mem.return_value = mem_stub

            events = servicer._maybe_store_reflexion_insight(
                "sess-1", "cust-1", "how much?", "price_check",
                ["price_lookup"], {"issues": ["incomplete"]},
                0.4, 0.8, "response text"
            )
            assert len(events) == 1
            assert events[0].type == "reflexion_learning"
            mem_stub.StoreEpisodicMemory.assert_called_once()

    def test_above_threshold_skips(self, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLEXION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLEXION_INSIGHT_THRESHOLD", 0.7)

        events = servicer._maybe_store_reflexion_insight(
            "sess-1", "cust-1", "q", "intent",
            [], {}, 0.8, 0.9, "text"
        )
        assert events == []

    def test_disabled_returns_empty(self, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLEXION_ENABLED", False)
        events = servicer._maybe_store_reflexion_insight(
            "sess-1", "cust-1", "q", "intent",
            [], {}, 0.3, 0.5, "text"
        )
        assert events == []
