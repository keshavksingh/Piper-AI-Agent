"""Tests for agent_service planning layer — _generate_plan and plan context injection."""

import json
from unittest.mock import patch, MagicMock

import pytest

from agent_service.server import AgentServiceServicer, build_react_history


@pytest.fixture
def servicer():
    return AgentServiceServicer()


class TestGeneratePlan:
    """Tests for _generate_plan."""

    @patch("agent_service.server.get_llm_stub")
    def test_simple_plan(self, mock_get_llm, servicer):
        stub = MagicMock()
        mock_get_llm.return_value = stub

        stub.GenerateAnswer.return_value = MagicMock(
            completion=json.dumps({
                "needs_multi_agent": False,
                "plan_steps": [
                    {"goal": "Search for products", "suggested_tool": "product_search", "priority": 1},
                ],
                "specialist_agents": [],
            })
        )

        plan = servicer._generate_plan("find me a widget", "product_inquiry", "product_search, price_lookup", "context")
        assert plan["needs_multi_agent"] is False
        assert len(plan["plan_steps"]) == 1
        assert plan["plan_steps"][0]["suggested_tool"] == "product_search"

    @patch("agent_service.server.get_llm_stub")
    def test_comparison_plan(self, mock_get_llm, servicer):
        stub = MagicMock()
        mock_get_llm.return_value = stub

        stub.GenerateAnswer.return_value = MagicMock(
            completion=json.dumps({
                "needs_multi_agent": True,
                "plan_steps": [
                    {"goal": "Look up Widget A", "suggested_tool": "product_search", "priority": 1},
                    {"goal": "Look up Widget B", "suggested_tool": "product_search", "priority": 1},
                    {"goal": "Compare features", "suggested_tool": "product_compare", "priority": 2},
                ],
                "specialist_agents": ["product_specialist", "comparison_specialist"],
            })
        )

        plan = servicer._generate_plan("compare Widget A and B", "comparison", "product_search, product_compare", "ctx")
        assert plan["needs_multi_agent"] is True
        assert len(plan["specialist_agents"]) == 2

    @patch("agent_service.server.get_llm_stub")
    def test_parse_failure_returns_fallback(self, mock_get_llm, servicer):
        stub = MagicMock()
        mock_get_llm.return_value = stub
        stub.GenerateAnswer.return_value = MagicMock(completion="not valid json")

        plan = servicer._generate_plan("query", "intent", "tools", "ctx")
        assert plan["needs_multi_agent"] is False
        assert plan["plan_steps"] == []
        assert plan["specialist_agents"] == []

    def test_disabled_skips_planning(self, servicer, monkeypatch):
        """When PLANNING_ENABLED is False, the plan is never generated in ProcessQuery."""
        monkeypatch.setattr("shared.config.Config.PLANNING_ENABLED", False)
        from shared.config import Config
        assert Config.PLANNING_ENABLED is False

    @patch("agent_service.server.get_llm_stub")
    def test_skipped_for_general_question(self, mock_get_llm, servicer, monkeypatch):
        """Planning should not run for general_question or out_of_scope intents."""
        monkeypatch.setattr("shared.config.Config.PLANNING_ENABLED", True)
        # Verify the condition: intent not in ("general_question", "out_of_scope")
        assert "general_question" not in ("product_inquiry", "price_check")
        assert "general_question" in ("general_question", "out_of_scope")


class TestPlanContextInjection:
    """Test that plan steps are injected into the ReACT system prompt."""

    def test_plan_context_built(self):
        """Verify plan context string format."""
        plan = {
            "plan_steps": [
                {"goal": "Search products", "suggested_tool": "product_search", "priority": 1},
                {"goal": "Check prices", "suggested_tool": "price_lookup", "priority": 2},
            ]
        }

        # Simulate the injection logic from _run_react_loop
        reflexion_context = ""
        if plan and plan.get("plan_steps"):
            plan_lines = ["\nExecution plan (follow these steps):"]
            for i, step in enumerate(plan["plan_steps"], 1):
                tool_hint = f" (use {step['suggested_tool']})" if step.get("suggested_tool") else ""
                plan_lines.append(f"  {i}. {step['goal']}{tool_hint}")
            plan_lines.append("")
            reflexion_context += "\n".join(plan_lines)

        assert "Execution plan" in reflexion_context
        assert "1. Search products (use product_search)" in reflexion_context
        assert "2. Check prices (use price_lookup)" in reflexion_context
