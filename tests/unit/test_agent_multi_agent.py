"""Tests for agent_service multi-agent orchestration — sub-loops and synthesis."""

import json
from unittest.mock import patch, MagicMock

import pytest

from agent_service.server import AgentServiceServicer, AGENT_REGISTRY


@pytest.fixture
def servicer():
    return AgentServiceServicer()


class TestRunAgentSubLoop:
    """Tests for _run_agent_sub_loop."""

    @patch("agent_service.server.get_llm_stub")
    def test_answer_found(self, mock_get_llm, servicer):
        stub = MagicMock()
        mock_get_llm.return_value = stub

        # LLM returns an answer immediately
        stub.GenerateAnswer.return_value = MagicMock(
            completion="Thought: I can answer directly\nAnswer: The Widget costs $50."
        )

        result_text, tools_used, steps = servicer._run_agent_sub_loop(
            "product_specialist", "how much is widget",
            "No previous conversation.",
            "- product_search: Search products",
            ["product_search", "price_lookup"],
            {},
        )
        assert "Widget costs $50" in result_text
        assert tools_used == []

    @patch("agent_service.server.get_llm_stub")
    @patch("agent_service.server.get_tool_stub")
    def test_preferred_tools(self, mock_get_tool, mock_get_llm, servicer):
        llm_stub = MagicMock()
        mock_get_llm.return_value = llm_stub
        tool_stub = MagicMock()
        mock_get_tool.return_value = tool_stub

        # First call: action, Second call: answer
        llm_stub.GenerateAnswer.side_effect = [
            MagicMock(completion='Thought: Search for products\nAction: product_search({"query": "widget"})'),
            MagicMock(completion="Thought: Found it\nAnswer: Widget is $50."),
        ]
        tool_stub.ExecuteTool.return_value = MagicMock(
            success=True, result='{"results": [{"product_name": "Widget", "price": 50}], "count": 1}'
        )

        result_text, tools_used, steps = servicer._run_agent_sub_loop(
            "product_specialist", "find widget",
            "No previous conversation.",
            "- product_search: Search\n- price_lookup: Prices",
            ["product_search", "price_lookup"],
            {},
        )
        assert "product_search" in tools_used
        assert "Widget is $50" in result_text

    @patch("agent_service.server.get_llm_stub")
    def test_max_iterations(self, mock_get_llm, servicer):
        stub = MagicMock()
        mock_get_llm.return_value = stub

        # Never gives an answer, always thinks
        stub.GenerateAnswer.return_value = MagicMock(
            completion="Thought: Still thinking\nAction: unknown_tool({})"
        )

        result_text, tools_used, steps = servicer._run_agent_sub_loop(
            "product_specialist", "complex query",
            "ctx", "- product_search: Search", ["product_search"], {},
        )
        # After max iterations (4), should return observations or "No findings."
        assert result_text is not None


class TestRunMultiAgentLoop:
    """Tests for _run_multi_agent_loop."""

    @patch("agent_service.server.get_llm_stub")
    @patch("agent_service.server.get_memory_stub")
    @patch("agent_service.server.get_rec_stub")
    def test_dispatches_specialists(self, mock_rec, mock_mem, mock_llm, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", False)
        monkeypatch.setattr("shared.config.Config.GUARDRAILS_ENABLED", False)
        monkeypatch.setattr("shared.config.Config.EVALUATION_STORAGE_ENABLED", False)
        monkeypatch.setattr("shared.config.Config.MULTI_AGENT_ENABLED", True)

        llm_stub = MagicMock()
        mock_llm.return_value = llm_stub
        mem_stub = MagicMock()
        mock_mem.return_value = mem_stub
        rec_stub = MagicMock()
        mock_rec.return_value = rec_stub
        rec_stub.GetFollowUpRecommendations.return_value = MagicMock(suggestions=["q1"])

        # Sub-loop answers immediately
        llm_stub.GenerateAnswer.return_value = MagicMock(
            completion="Thought: Done\nAnswer: Specialist findings here."
        )

        plan = {
            "needs_multi_agent": True,
            "specialist_agents": ["product_specialist", "comparison_specialist"],
            "plan_steps": [],
        }

        events = list(servicer._run_multi_agent_loop(
            "sess-1", "cust-1", "compare widgets", "comparison",
            "ctx", mem_stub, 1000000000.0, plan,
            "- product_search: Search", ["product_search"], {},
        ))

        event_types = [e.type for e in events]
        assert "agent_started" in event_types
        assert "agent_complete" in event_types
        assert "response_complete" in event_types

    @patch("agent_service.server.get_llm_stub")
    @patch("agent_service.server.get_memory_stub")
    @patch("agent_service.server.get_rec_stub")
    def test_events_emitted_per_agent(self, mock_rec, mock_mem, mock_llm, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", False)
        monkeypatch.setattr("shared.config.Config.GUARDRAILS_ENABLED", False)
        monkeypatch.setattr("shared.config.Config.EVALUATION_STORAGE_ENABLED", False)

        llm_stub = MagicMock()
        mock_llm.return_value = llm_stub
        mem_stub = MagicMock()
        mock_mem.return_value = mem_stub
        rec_stub = MagicMock()
        mock_rec.return_value = rec_stub
        rec_stub.GetFollowUpRecommendations.return_value = MagicMock(suggestions=[])

        llm_stub.GenerateAnswer.return_value = MagicMock(
            completion="Thought: Done\nAnswer: Result."
        )

        plan = {
            "needs_multi_agent": True,
            "specialist_agents": ["warranty_specialist"],
            "plan_steps": [],
        }

        events = list(servicer._run_multi_agent_loop(
            "s", "c", "warranty?", "warranty_question",
            "ctx", mem_stub, 1000000000.0, plan,
            "- warranty_check: Check", ["warranty_check"], {},
        ))

        # Should have agent_started and agent_complete for warranty_specialist
        started = [e for e in events if e.type == "agent_started"]
        completed = [e for e in events if e.type == "agent_complete"]
        assert len(started) == 1
        assert "warranty_specialist" in started[0].payload

    @patch("agent_service.server.get_llm_stub")
    @patch("agent_service.server.get_memory_stub")
    @patch("agent_service.server.get_rec_stub")
    def test_unknown_agents_fallback(self, mock_rec, mock_mem, mock_llm, servicer, monkeypatch):
        """Unknown agents in plan should be filtered, falling back to single ReACT."""
        monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", False)
        monkeypatch.setattr("shared.config.Config.GUARDRAILS_ENABLED", False)
        monkeypatch.setattr("shared.config.Config.EVALUATION_STORAGE_ENABLED", False)
        monkeypatch.setattr("shared.config.Config.REFLEXION_ENABLED", False)

        llm_stub = MagicMock()
        mock_llm.return_value = llm_stub
        mem_stub = MagicMock()
        mock_mem.return_value = mem_stub
        rec_stub = MagicMock()
        mock_rec.return_value = rec_stub
        rec_stub.GetFollowUpRecommendations.return_value = MagicMock(suggestions=[])

        llm_stub.GenerateAnswer.return_value = MagicMock(
            completion="Thought: Done\nAnswer: Fallback answer."
        )

        plan = {
            "needs_multi_agent": True,
            "specialist_agents": ["nonexistent_agent"],
            "plan_steps": [],
        }

        events = list(servicer._run_multi_agent_loop(
            "s", "c", "query", "intent",
            "ctx", mem_stub, 1000000000.0, plan,
            "- product_search: Search", ["product_search"], {},
        ))

        # Should fallback to ReACT loop (has agent_thinking events)
        event_types = [e.type for e in events]
        assert "agent_started" not in event_types  # No multi-agent dispatch

    @patch("agent_service.server.get_llm_stub")
    @patch("agent_service.server.get_memory_stub")
    @patch("agent_service.server.get_rec_stub")
    def test_synthesis(self, mock_rec, mock_mem, mock_llm, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", False)
        monkeypatch.setattr("shared.config.Config.GUARDRAILS_ENABLED", False)
        monkeypatch.setattr("shared.config.Config.EVALUATION_STORAGE_ENABLED", False)

        llm_stub = MagicMock()
        mock_llm.return_value = llm_stub
        mem_stub = MagicMock()
        mock_mem.return_value = mem_stub
        rec_stub = MagicMock()
        mock_rec.return_value = rec_stub
        rec_stub.GetFollowUpRecommendations.return_value = MagicMock(suggestions=[])

        # Sub-loop answer, synthesis call, frame call
        llm_stub.GenerateAnswer.side_effect = [
            MagicMock(completion="Thought: Done\nAnswer: Product info."),
            MagicMock(completion="Thought: Done\nAnswer: Comparison info."),
            MagicMock(completion="Combined synthesis of both specialists."),
            MagicMock(completion=json.dumps({
                "text": "Here's your comparison.", "confidence": 0.85, "sources": ["Widget A"],
            })),
        ]

        plan = {
            "needs_multi_agent": True,
            "specialist_agents": ["product_specialist", "comparison_specialist"],
            "plan_steps": [],
        }

        events = list(servicer._run_multi_agent_loop(
            "s", "c", "compare A and B", "comparison",
            "ctx", mem_stub, 1000000000.0, plan,
            "- product_search: S\n- product_compare: C",
            ["product_search", "product_compare"], {},
        ))

        # Should have token events from the synthesized response
        token_events = [e for e in events if e.type == "token"]
        assert len(token_events) > 0
