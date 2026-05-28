"""Tests for agent_service reflection loop — evaluate, refine, loop control."""

import json
from unittest.mock import patch, MagicMock

import pytest

from agent_service.server import AgentServiceServicer


@pytest.fixture
def servicer():
    return AgentServiceServicer()


@pytest.fixture
def mock_llm_stub():
    stub = MagicMock()
    return stub


class TestRunReflectionLoop:
    """Tests for _run_reflection_loop orchestration."""

    @patch("agent_service.server.get_llm_stub")
    def test_high_score_stops_immediately(self, mock_get_llm, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLECTION_QUALITY_THRESHOLD", 0.75)

        stub = MagicMock()
        mock_get_llm.return_value = stub
        # Return high score evaluation
        stub.GenerateAnswer.return_value = MagicMock(
            completion=json.dumps({
                "completeness": 0.9, "accuracy": 0.9, "relevance": 0.9,
                "clarity": 0.9, "actionability": 0.9,
                "overall_score": 0.9, "issues": [], "suggestions": [],
                "needs_refinement": False,
            })
        )

        framed = {"text": "The widget costs $299.", "confidence": 0.8, "sources": []}
        result, events, original_score, last_eval = servicer._run_reflection_loop(
            "price?", framed, ["price_lookup"], [], "context"
        )
        assert original_score == 0.9
        # Should not have refining events (only evaluating + critique)
        event_types = [e.type for e in events]
        assert "reflection_refining" not in event_types

    @patch("agent_service.server.get_llm_stub")
    def test_low_score_triggers_refine(self, mock_get_llm, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLECTION_QUALITY_THRESHOLD", 0.75)
        monkeypatch.setattr("shared.config.Config.REFLECTION_MAX_ITERATIONS", 2)

        stub = MagicMock()
        mock_get_llm.return_value = stub

        # First call: evaluate (low score), Second call: refine, Third call: evaluate (high score)
        stub.GenerateAnswer.side_effect = [
            MagicMock(completion=json.dumps({
                "overall_score": 0.4, "issues": ["Incomplete info"],
                "suggestions": ["Add price details"], "needs_refinement": True,
                "completeness": 0.4, "accuracy": 0.5, "relevance": 0.6,
                "clarity": 0.5, "actionability": 0.3,
            })),
            MagicMock(completion=json.dumps({
                "text": "Improved response with more details.",
                "confidence": 0.85, "sources": ["UltraWidget"],
            })),
            MagicMock(completion=json.dumps({
                "overall_score": 0.85, "issues": [], "suggestions": [],
                "needs_refinement": False,
                "completeness": 0.9, "accuracy": 0.9, "relevance": 0.8,
                "clarity": 0.8, "actionability": 0.8,
            })),
        ]

        framed = {"text": "Widget info.", "confidence": 0.5, "sources": []}
        result, events, original_score, last_eval = servicer._run_reflection_loop(
            "query", framed, ["product_search"], [], "context"
        )
        assert original_score == 0.4
        assert result["text"] == "Improved response with more details."
        event_types = [e.type for e in events]
        assert "reflection_refining" in event_types

    @patch("agent_service.server.get_llm_stub")
    def test_max_iterations_reached(self, mock_get_llm, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLECTION_QUALITY_THRESHOLD", 0.99)
        monkeypatch.setattr("shared.config.Config.REFLECTION_MAX_ITERATIONS", 1)

        stub = MagicMock()
        mock_get_llm.return_value = stub

        # Low score but only 1 iteration allowed
        stub.GenerateAnswer.side_effect = [
            MagicMock(completion=json.dumps({
                "overall_score": 0.5, "issues": ["Meh"], "suggestions": ["Better"],
                "needs_refinement": True,
                "completeness": 0.5, "accuracy": 0.5, "relevance": 0.5,
                "clarity": 0.5, "actionability": 0.5,
            })),
            MagicMock(completion=json.dumps({
                "text": "Refined once.", "confidence": 0.7, "sources": [],
            })),
        ]

        framed = {"text": "Original.", "confidence": 0.5, "sources": []}
        result, events, _, _eval = servicer._run_reflection_loop(
            "query", framed, [], [], "ctx"
        )
        # After 1 iteration of refine, loop ends
        assert result["text"] == "Refined once."

    @patch("agent_service.server.get_llm_stub")
    def test_no_refinement_flag_stops(self, mock_get_llm, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLECTION_QUALITY_THRESHOLD", 0.9)

        stub = MagicMock()
        mock_get_llm.return_value = stub

        stub.GenerateAnswer.return_value = MagicMock(
            completion=json.dumps({
                "overall_score": 0.6, "issues": ["Minor"],
                "suggestions": [], "needs_refinement": False,
                "completeness": 0.6, "accuracy": 0.7, "relevance": 0.6,
                "clarity": 0.7, "actionability": 0.5,
            })
        )

        framed = {"text": "Okay.", "confidence": 0.6, "sources": []}
        result, events, _, _eval = servicer._run_reflection_loop(
            "q", framed, [], [], "ctx"
        )
        # Should stop without refining despite low score
        event_types = [e.type for e in events]
        assert "reflection_refining" not in event_types

    @patch("agent_service.server.get_llm_stub")
    def test_refine_failure_keeps_previous(self, mock_get_llm, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLECTION_QUALITY_THRESHOLD", 0.9)
        monkeypatch.setattr("shared.config.Config.REFLECTION_MAX_ITERATIONS", 2)

        stub = MagicMock()
        mock_get_llm.return_value = stub

        stub.GenerateAnswer.side_effect = [
            MagicMock(completion=json.dumps({
                "overall_score": 0.3, "issues": ["Bad"], "suggestions": ["Fix"],
                "needs_refinement": True,
                "completeness": 0.3, "accuracy": 0.3, "relevance": 0.3,
                "clarity": 0.3, "actionability": 0.3,
            })),
            # Refine returns invalid JSON
            MagicMock(completion="not valid json at all"),
        ]

        framed = {"text": "Keep this.", "confidence": 0.5, "sources": []}
        result, events, _, _eval = servicer._run_reflection_loop(
            "q", framed, [], [], "ctx"
        )
        # Should keep the original framed response
        assert result["text"] == "Keep this."

    @patch("agent_service.server.get_llm_stub")
    def test_events_emitted(self, mock_get_llm, servicer, monkeypatch):
        monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLECTION_QUALITY_THRESHOLD", 0.75)

        stub = MagicMock()
        mock_get_llm.return_value = stub
        stub.GenerateAnswer.return_value = MagicMock(
            completion=json.dumps({
                "overall_score": 0.8, "issues": [], "suggestions": [],
                "needs_refinement": False,
                "completeness": 0.8, "accuracy": 0.8, "relevance": 0.8,
                "clarity": 0.8, "actionability": 0.8,
            })
        )

        framed = {"text": "Good.", "confidence": 0.8, "sources": []}
        _, events, _, _eval = servicer._run_reflection_loop("q", framed, [], [], "ctx")
        assert len(events) >= 2  # evaluating + critique
        assert events[0].type == "reflection_evaluating"
        assert events[1].type == "reflection_critique"

    @patch("agent_service.server.get_llm_stub")
    def test_parse_failure_fallback_score_1(self, mock_get_llm, servicer, monkeypatch):
        """When evaluate returns unparseable JSON, assume score=1.0 (skip refinement)."""
        monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLECTION_QUALITY_THRESHOLD", 0.75)

        stub = MagicMock()
        mock_get_llm.return_value = stub
        stub.GenerateAnswer.return_value = MagicMock(completion="totally broken json")

        framed = {"text": "Original.", "confidence": 0.5, "sources": []}
        result, events, score, _eval = servicer._run_reflection_loop("q", framed, [], [], "ctx")
        # The _evaluate_response fallback returns score=1.0
        assert score == 1.0
        assert result["text"] == "Original."


    @patch("agent_service.server.get_llm_stub")
    def test_last_evaluation_returned(self, mock_get_llm, servicer, monkeypatch):
        """Verify _run_reflection_loop returns the last evaluation dict."""
        monkeypatch.setattr("shared.config.Config.REFLECTION_ENABLED", True)
        monkeypatch.setattr("shared.config.Config.REFLECTION_QUALITY_THRESHOLD", 0.75)

        stub = MagicMock()
        mock_get_llm.return_value = stub
        stub.GenerateAnswer.return_value = MagicMock(
            completion=json.dumps({
                "completeness": 0.9, "accuracy": 0.9, "relevance": 0.9,
                "clarity": 0.9, "actionability": 0.9,
                "overall_score": 0.9, "issues": [], "suggestions": [],
                "needs_refinement": False,
            })
        )

        framed = {"text": "Good answer.", "confidence": 0.8, "sources": []}
        _, _, _, last_eval = servicer._run_reflection_loop("q", framed, [], [], "ctx")
        assert last_eval["overall_score"] == 0.9
        assert last_eval["needs_refinement"] is False


class TestEvaluateResponse:
    """Tests for _evaluate_response directly."""

    @patch("agent_service.server.get_llm_stub")
    def test_valid_evaluation(self, mock_get_llm, servicer):
        stub = MagicMock()
        mock_get_llm.return_value = stub
        stub.GenerateAnswer.return_value = MagicMock(
            completion=json.dumps({
                "completeness": 0.8, "accuracy": 0.9, "relevance": 0.7,
                "clarity": 0.8, "actionability": 0.6,
                "overall_score": 0.76, "issues": ["Minor"], "suggestions": ["Improve"],
                "needs_refinement": True,
            })
        )

        result = servicer._evaluate_response("query", "response text", ["tool1"], 2, "context")
        assert result["overall_score"] == 0.76
        assert result["needs_refinement"] is True

    @patch("agent_service.server.get_llm_stub")
    def test_parse_failure_returns_high_score(self, mock_get_llm, servicer):
        stub = MagicMock()
        mock_get_llm.return_value = stub
        stub.GenerateAnswer.return_value = MagicMock(completion="not json")

        result = servicer._evaluate_response("q", "r", [], 0, "ctx")
        assert result["overall_score"] == 1.0
        assert result["needs_refinement"] is False


class TestRefineResponse:
    """Tests for _refine_response directly."""

    @patch("agent_service.server.get_llm_stub")
    def test_valid_refinement(self, mock_get_llm, servicer):
        stub = MagicMock()
        mock_get_llm.return_value = stub
        stub.GenerateAnswer.return_value = MagicMock(
            completion=json.dumps({
                "text": "Improved response.", "confidence": 0.85, "sources": ["Widget"],
            })
        )

        evaluation = {"overall_score": 0.4, "issues": ["Incomplete"], "suggestions": ["Add detail"]}
        result = servicer._refine_response("query", "original", ["tool"], evaluation, "observations")
        assert result["text"] == "Improved response."
        assert result["confidence"] == 0.85

    @patch("agent_service.server.get_llm_stub")
    def test_parse_failure_returns_none(self, mock_get_llm, servicer):
        stub = MagicMock()
        mock_get_llm.return_value = stub
        stub.GenerateAnswer.return_value = MagicMock(completion="invalid json response")

        result = servicer._refine_response("q", "r", [], {}, "obs")
        assert result is None
