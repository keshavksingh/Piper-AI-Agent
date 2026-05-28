"""Tests for llm_service — GenerateAnswer, ClassifyIntent, GenerateStructured."""

import json
from unittest.mock import patch, MagicMock

import grpc
import pytest

from llm_service.server import LLMServiceServicer, build_messages


@pytest.fixture
def servicer():
    return LLMServiceServicer()


@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.set_code = MagicMock()
    ctx.set_details = MagicMock()
    return ctx


class TestGenerateAnswer:
    """Tests for GenerateAnswer."""

    @patch("llm_service.server.client")
    def test_success(self, mock_client, servicer, mock_context):
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Hello! I can help with product questions.")]
        )

        request = MagicMock()
        request.prompt = "Hi there"
        request.memory = []
        request.system_prompt = ""
        request.temperature = 0.0
        request.max_tokens = 0

        response = servicer.GenerateAnswer(request, mock_context)
        assert response.completion == "Hello! I can help with product questions."

    @patch("llm_service.server.client")
    def test_custom_system_prompt(self, mock_client, servicer, mock_context):
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Custom response")]
        )

        request = MagicMock()
        request.prompt = "Test"
        request.memory = []
        request.system_prompt = "You are a custom assistant."
        request.temperature = 0.5
        request.max_tokens = 256

        servicer.GenerateAnswer(request, mock_context)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a custom assistant."
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 256

    @patch("llm_service.server.client")
    def test_with_memory(self, mock_client, servicer, mock_context):
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Based on our conversation...")]
        )

        request = MagicMock()
        request.prompt = "What else?"
        request.memory = ["What products do you have?", "We have widgets and gadgets."]
        request.system_prompt = ""
        request.temperature = 0.0
        request.max_tokens = 0

        servicer.GenerateAnswer(request, mock_context)

        call_kwargs = mock_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 3  # 2 memory turns + 1 prompt
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    @patch("llm_service.server.client")
    def test_error_handling(self, mock_client, servicer, mock_context):
        mock_client.messages.create.side_effect = Exception("API error")

        request = MagicMock()
        request.prompt = "Test"
        request.memory = []
        request.system_prompt = ""
        request.temperature = 0.0
        request.max_tokens = 0

        response = servicer.GenerateAnswer(request, mock_context)
        assert response.completion == ""
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)


class TestClassifyIntent:
    """Tests for ClassifyIntent."""

    @patch("llm_service.server.client")
    def test_valid_json(self, mock_client, servicer, mock_context):
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=json.dumps({
                "intent": "product_inquiry",
                "confidence": 0.92,
                "entities": ["widget"],
                "needs_clarification": False,
                "clarification_question": "",
            }))]
        )

        request = MagicMock()
        request.query = "Tell me about widgets"
        request.conversation_context = []

        response = servicer.ClassifyIntent(request, mock_context)
        assert response.intent == "product_inquiry"
        assert response.confidence == pytest.approx(0.92)
        assert "widget" in response.entities

    @patch("llm_service.server.client")
    def test_code_fence_stripping(self, mock_client, servicer, mock_context):
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='```json\n{"intent": "price_check", "confidence": 0.8, "entities": [], "needs_clarification": false, "clarification_question": ""}\n```')]
        )

        request = MagicMock()
        request.query = "How much?"
        request.conversation_context = []

        response = servicer.ClassifyIntent(request, mock_context)
        assert response.intent == "price_check"

    @patch("llm_service.server.client")
    def test_parse_failure_fallback(self, mock_client, servicer, mock_context):
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="I cannot classify this properly")]
        )

        request = MagicMock()
        request.query = "????"
        request.conversation_context = []

        response = servicer.ClassifyIntent(request, mock_context)
        assert response.intent == "general_question"
        assert response.confidence == 0.5

    @patch("llm_service.server.client")
    def test_api_error(self, mock_client, servicer, mock_context):
        mock_client.messages.create.side_effect = Exception("Network error")

        request = MagicMock()
        request.query = "test"
        request.conversation_context = []

        response = servicer.ClassifyIntent(request, mock_context)
        assert response.intent == "general_question"
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)


class TestGenerateStructured:
    """Tests for GenerateStructured."""

    @patch("llm_service.server.client")
    def test_valid_json_output(self, mock_client, servicer, mock_context):
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"plan_steps": [{"goal": "Search"}]}')]
        )

        request = MagicMock()
        request.prompt = "Generate a plan"
        request.system_prompt = ""
        request.temperature = 0.0
        request.max_tokens = 0

        response = servicer.GenerateStructured(request, mock_context)
        parsed = json.loads(response.json_output)
        assert "plan_steps" in parsed

    @patch("llm_service.server.client")
    def test_invalid_json_fallback(self, mock_client, servicer, mock_context):
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="This is not JSON at all")]
        )

        request = MagicMock()
        request.prompt = "Generate"
        request.system_prompt = ""
        request.temperature = 0.0
        request.max_tokens = 0

        response = servicer.GenerateStructured(request, mock_context)
        assert response.json_output == "{}"


class TestBuildMessages:
    """Tests for build_messages helper."""

    def test_no_memory(self):
        system, messages = build_messages("Hello", [])
        assert system is not None
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_with_memory(self):
        system, messages = build_messages(
            "Follow up",
            ["First question", "First answer"],
        )
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_odd_length_memory_merges_consecutive_roles(self):
        """Odd-length memory would produce two consecutive user messages without merging."""
        system, messages = build_messages(
            "Third question",
            ["First question", "First answer", "Second question"],
        )
        # Without merging: [user, assistant, user, user] — violates Claude API
        # With merging: [user, assistant, user] — second question + third question merged
        roles = [m["role"] for m in messages]
        # Verify no consecutive same roles
        for i in range(len(roles) - 1):
            assert roles[i] != roles[i + 1], f"Consecutive same role at index {i}: {roles}"
        assert messages[0]["role"] == "user"
        # The last message should contain both "Second question" and "Third question"
        assert "Second question" in messages[-1]["content"]
        assert "Third question" in messages[-1]["content"]
