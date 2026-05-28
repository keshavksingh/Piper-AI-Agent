"""Tests for agent_service ReACT parsing utilities."""

import json
import pytest

from agent_service.server import (
    parse_react_output,
    build_react_history,
    build_memory_context,
    _strip_llm_json,
)


class TestParseReactOutput:
    """Tests for parse_react_output."""

    def test_thought_and_action(self):
        text = (
            "Thought: I need to search for products under $100\n"
            'Action: product_search({"query": "products under 100"})'
        )
        thought, action, action_input, answer = parse_react_output(text)
        assert thought == "I need to search for products under $100"
        assert action == "product_search"
        assert "products under 100" in action_input
        assert answer is None

    def test_thought_and_answer(self):
        text = (
            "Thought: I have enough information to answer\n"
            "Answer: The UltraWidget Pro costs $299.99."
        )
        thought, action, action_input, answer = parse_react_output(text)
        assert thought == "I have enough information to answer"
        assert action is None
        assert answer == "The UltraWidget Pro costs $299.99."

    def test_nested_json_params(self):
        text = (
            "Thought: Need to compare\n"
            'Action: product_compare({"product_names": ["Widget A", "Widget B"]})'
        )
        thought, action, action_input, answer = parse_react_output(text)
        assert action == "product_compare"
        parsed = json.loads(action_input)
        assert "Widget A" in parsed["product_names"]

    def test_missing_thought(self):
        text = 'Action: product_search({"query": "test"})'
        thought, action, action_input, answer = parse_react_output(text)
        assert thought == ""
        assert action == "product_search"

    def test_missing_action_and_answer(self):
        text = "Thought: I'm not sure what to do next"
        thought, action, action_input, answer = parse_react_output(text)
        assert thought == "I'm not sure what to do next"
        assert action is None
        assert answer is None

    def test_multiline_thought(self):
        text = (
            "Thought: I need to consider multiple factors:\n"
            "1. Price range\n"
            "2. Warranty\n"
            "Answer: Based on your needs, I recommend the BasicWidget."
        )
        thought, action, action_input, answer = parse_react_output(text)
        assert "multiple factors" in thought
        assert answer is not None


class TestBuildReactHistory:
    """Tests for build_react_history."""

    def test_empty_steps(self):
        result = build_react_history([])
        assert result == "No previous reasoning steps."

    def test_with_steps(self):
        steps = [
            {
                "iteration": 1,
                "thought": "Search for products",
                "action": "product_search",
                "action_input": '{"query": "widgets"}',
                "observation": '{"results": []}',
            },
        ]
        result = build_react_history(steps)
        assert "Thought 1:" in result
        assert "Action 1:" in result
        assert "Observation 1:" in result

    def test_step_without_action(self):
        steps = [{"iteration": 1, "thought": "Just thinking"}]
        result = build_react_history(steps)
        assert "Thought 1:" in result
        assert "Action 1:" not in result

    def test_step_with_action_but_missing_action_input(self):
        """build_react_history should handle missing action_input gracefully."""
        steps = [
            {
                "iteration": 1,
                "thought": "Need to search",
                "action": "product_search",
                # No action_input key
                "observation": "some results",
            },
        ]
        result = build_react_history(steps)
        assert "Action 1: product_search({})" in result
        assert "Observation 1:" in result

    def test_step_with_action_but_missing_observation(self):
        """build_react_history should handle missing observation gracefully."""
        steps = [
            {
                "iteration": 1,
                "thought": "Need to search",
                "action": "product_search",
                "action_input": '{"query": "test"}',
                # No observation key
            },
        ]
        result = build_react_history(steps)
        assert "Observation 1: N/A" in result


class TestBuildMemoryContext:
    """Tests for build_memory_context."""

    def test_empty_turns(self):
        result = build_memory_context([])
        assert result == "No previous conversation."
        result = build_memory_context(None)
        assert result == "No previous conversation."

    def test_with_turns(self):
        class Turn:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        turns = [
            Turn("user", "Hello"),
            Turn("assistant", "Hi! How can I help?"),
        ]
        result = build_memory_context(turns)
        assert "User: Hello" in result
        assert "Assistant: Hi! How can I help?" in result


class TestStripLlmJson:
    """Tests for _strip_llm_json."""

    def test_strips_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = _strip_llm_json(text)
        assert result == '{"key": "value"}'

    def test_strips_bare_fence(self):
        text = '```\n{"key": "value"}\n```'
        result = _strip_llm_json(text)
        assert result == '{"key": "value"}'

    def test_no_fence_unchanged(self):
        text = '{"key": "value"}'
        result = _strip_llm_json(text)
        assert result == '{"key": "value"}'
