"""Tests for agent_service ReACT parsing utilities."""

import json
import pytest

from unittest.mock import patch, MagicMock

from agent_service.server import (
    parse_react_output,
    build_react_history,
    build_memory_context,
    _strip_llm_json,
    QUERY_REWRITE_PROMPT,
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

    class _MockToolCall:
        """Lightweight mock for memory_pb2.ToolCall."""
        def __init__(self, tool_name):
            self.tool_name = tool_name
            self.arguments = MagicMock(fields={})
            self.result = ""

    class Turn:
        def __init__(self, role, content, intent=None, tool_calls=None, created_at=None):
            self.role = role
            self.content = content
            self.intent = intent
            self.tool_calls = tool_calls or []
            self.created_at = created_at

    def test_empty_turns(self):
        result = build_memory_context([])
        assert result == "No previous conversation."
        result = build_memory_context(None)
        assert result == "No previous conversation."

    def test_with_turns(self):
        turns = [
            self.Turn("user", "Hello"),
            self.Turn("assistant", "Hi! How can I help?"),
        ]
        result = build_memory_context(turns)
        assert "User" in result
        assert "Hello" in result
        assert "Assistant" in result
        assert "Hi! How can I help?" in result

    def test_long_assistant_response_truncated_in_older_exchanges(self):
        """Older assistant turns in conversation flow should be truncated."""
        long_response = "A" * 500
        turns = [
            self.Turn("user", "Compare products"),
            self.Turn("assistant", long_response, intent="comparison"),
            self.Turn("user", "Help me decide"),
        ]
        result = build_memory_context(turns)
        # The older assistant turn should be truncated in the flow section
        assert "..." in result
        # Should not contain the full 500-char response in the flow
        # (it only appears in latest exchange if it's the most recent)
        flow_section = result.split("=== Active Context ===")[0]
        assert long_response not in flow_section

    def test_latest_exchange_preserved_in_full(self):
        """The latest exchange section should preserve the full assistant response."""
        long_response = "B" * 500
        turns = [
            self.Turn("user", "Compare products"),
            self.Turn("assistant", long_response, intent="comparison"),
        ]
        result = build_memory_context(turns)
        assert "Latest Exchange" in result
        assert long_response in result

    def test_active_context_includes_products(self):
        """Active context section should list products mentioned across all turns."""
        turns = [
            self.Turn("user", "Tell me about MegaBlender"),
            self.Turn("assistant", "The MegaBlender is great. Also check SuperVac.", intent="product_inquiry"),
        ]
        result = build_memory_context(turns)
        assert "Active Context" in result
        assert "MegaBlender" in result
        assert "SuperVac" in result

    def test_active_context_includes_intents(self):
        """Active context section should list intents used across all turns."""
        turns = [
            self.Turn("user", "Compare products"),
            self.Turn("assistant", "Here is the comparison.", intent="comparison"),
            self.Turn("user", "What about warranty?"),
            self.Turn("assistant", "Warranty info.", intent="warranty_question"),
        ]
        result = build_memory_context(turns)
        assert "Intents:" in result
        assert "comparison" in result
        assert "warranty_question" in result

    def test_active_context_includes_tools(self):
        """Active context section should list tools called across all turns."""
        turns = [
            self.Turn("user", "Search for products"),
            self.Turn("assistant", "Found results.",
                      intent="product_inquiry",
                      tool_calls=[self._MockToolCall("product_search"), self._MockToolCall("price_lookup")]),
        ]
        result = build_memory_context(turns)
        assert "Tools used:" in result
        assert "product_search" in result
        assert "price_lookup" in result

    def test_conversation_flow_numbered_exchanges(self):
        """Conversation flow should number user/assistant exchanges."""
        turns = [
            self.Turn("user", "Hello"),
            self.Turn("assistant", "Hi there!"),
            self.Turn("user", "Help me"),
            self.Turn("assistant", "Sure, what do you need?"),
        ]
        result = build_memory_context(turns)
        assert "Conversation Flow" in result
        assert "1. User:" in result
        assert "2. User:" in result

    def test_latest_exchange_shows_last_user_and_assistant(self):
        """Latest exchange section should contain the last user and assistant turn."""
        turns = [
            self.Turn("user", "First question"),
            self.Turn("assistant", "First answer"),
            self.Turn("user", "Tell me about UltraWasher 2503"),
            self.Turn("assistant", "The UltraWasher 2503 costs $349.", intent="product_inquiry"),
        ]
        result = build_memory_context(turns)
        latest_section = result.split("=== Latest Exchange ===")[1]
        assert "UltraWasher 2503" in latest_section
        assert "$349" in latest_section


class TestQueryRewritePrompt:
    """Tests for the query rewrite prompt template and _rewrite_query method."""

    def test_prompt_template_has_required_placeholders(self):
        """QUERY_REWRITE_PROMPT should have {history} and {query} placeholders."""
        assert "{history}" in QUERY_REWRITE_PROMPT
        assert "{query}" in QUERY_REWRITE_PROMPT

    @patch("agent_service.server.get_llm_stub")
    def test_rewrite_query_calls_llm(self, mock_llm):
        """_rewrite_query should call the LLM and return the rewritten query."""
        from agent_service.server import AgentServiceServicer

        llm_stub = MagicMock()
        mock_llm.return_value = llm_stub
        llm_stub.GenerateAnswer.return_value = MagicMock(
            completion="Compare UltraWasher 2503 with RoboCleaner 3120"
        )

        class Turn:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        servicer = AgentServiceServicer()
        history = [
            Turn("user", "Tell me about UltraWasher 2503"),
            Turn("assistant", "The UltraWasher 2503 costs $349."),
        ]
        result = servicer._rewrite_query("Compare this with RoboCleaner 3120", history)

        assert result == "Compare UltraWasher 2503 with RoboCleaner 3120"
        llm_stub.GenerateAnswer.assert_called_once()
        # Verify the prompt contains the history and query
        call_prompt = llm_stub.GenerateAnswer.call_args[0][0].prompt
        assert "UltraWasher 2503" in call_prompt
        assert "Compare this with RoboCleaner 3120" in call_prompt

    @patch("agent_service.server.get_llm_stub")
    def test_rewrite_query_empty_history_returns_original(self, mock_llm):
        """_rewrite_query with no history should return the original query unchanged."""
        from agent_service.server import AgentServiceServicer

        servicer = AgentServiceServicer()
        result = servicer._rewrite_query("Tell me about MegaBlender", [])

        assert result == "Tell me about MegaBlender"
        mock_llm.return_value.GenerateAnswer.assert_not_called()

    @patch("agent_service.server.get_llm_stub")
    def test_rewrite_query_sanity_check_rejects_too_long(self, mock_llm):
        """_rewrite_query should reject rewrites that are unreasonably long."""
        from agent_service.server import AgentServiceServicer

        llm_stub = MagicMock()
        mock_llm.return_value = llm_stub
        # LLM returns something way too long (e.g. a hallucinated paragraph)
        llm_stub.GenerateAnswer.return_value = MagicMock(
            completion="A" * 1000
        )

        class Turn:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        servicer = AgentServiceServicer()
        original = "Compare this"
        result = servicer._rewrite_query(original, [Turn("assistant", "Hi")])

        # Should fall back to original
        assert result == original

    @patch("agent_service.server.get_llm_stub")
    def test_rewrite_query_handles_empty_llm_response(self, mock_llm):
        """_rewrite_query should return original when LLM returns empty."""
        from agent_service.server import AgentServiceServicer

        llm_stub = MagicMock()
        mock_llm.return_value = llm_stub
        llm_stub.GenerateAnswer.return_value = MagicMock(completion="")

        class Turn:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        servicer = AgentServiceServicer()
        original = "What about the warranty?"
        result = servicer._rewrite_query(original, [Turn("assistant", "Info")])

        assert result == original


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
