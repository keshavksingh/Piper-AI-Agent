"""Tests for memory_service — sessions, conversation turns, episodic memory, audit."""

import json
import uuid
from unittest.mock import patch, MagicMock
from datetime import datetime

import grpc
import pytest

from memory_service.server import MemoryServiceServicer, _session_key, _turns_key


@pytest.fixture
def servicer():
    return MemoryServiceServicer()


@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.set_code = MagicMock()
    ctx.set_details = MagicMock()
    return ctx


class TestCreateSession:
    """Tests for CreateSession."""

    @patch("memory_service.server.get_ts_conn")
    @patch("memory_service.server.get_pg_conn")
    @patch("memory_service.server.redis_client")
    def test_creates_session(self, mock_redis, mock_pg, mock_ts, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        ts_conn = MagicMock()
        ts_cursor = MagicMock()
        ts_conn.cursor.return_value.__enter__ = MagicMock(return_value=ts_cursor)
        ts_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_ts.return_value = ts_conn

        request = MagicMock()
        request.customer_id = "cust-123"

        response = servicer.CreateSession(request, mock_context)

        assert response.customer_id == "cust-123"
        assert response.is_new is True
        assert len(response.session_id) == 36  # UUID format
        mock_redis.setex.assert_called_once()
        conn.commit.assert_called_once()

    @patch("memory_service.server.get_ts_conn")
    @patch("memory_service.server.get_pg_conn")
    @patch("memory_service.server.redis_client")
    def test_stores_in_redis(self, mock_redis, mock_pg, mock_ts, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        mock_ts.return_value = MagicMock()
        mock_ts.return_value.cursor.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_ts.return_value.cursor.return_value.__exit__ = MagicMock(return_value=False)

        request = MagicMock()
        request.customer_id = "cust-456"

        response = servicer.CreateSession(request, mock_context)
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0].startswith("session:")


class TestGetSession:
    """Tests for GetSession."""

    @patch("memory_service.server.redis_client")
    def test_cache_hit(self, mock_redis, servicer, mock_context):
        cached_data = json.dumps({
            "session_id": "sess-1",
            "customer_id": "cust-1",
            "created_at": "2024-01-01T00:00:00",
            "last_active_at": "2024-01-01T00:00:00",
        })
        mock_redis.get.return_value = cached_data

        request = MagicMock()
        request.session_id = "sess-1"

        response = servicer.GetSession(request, mock_context)
        assert response.session_id == "sess-1"
        assert response.customer_id == "cust-1"
        assert response.is_new is False

    @patch("memory_service.server.get_pg_conn")
    @patch("memory_service.server.redis_client")
    def test_cache_miss_pg_fallback(self, mock_redis, mock_pg, servicer, mock_context):
        mock_redis.get.return_value = None

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchone.return_value = {
            "id": "sess-2",
            "customer_id": "cust-2",
            "created_at": datetime(2024, 1, 1),
            "last_active_at": datetime(2024, 1, 1),
        }

        request = MagicMock()
        request.session_id = "sess-2"

        response = servicer.GetSession(request, mock_context)
        assert response.session_id == "sess-2"
        # Should re-cache in Redis
        mock_redis.setex.assert_called_once()

    @patch("memory_service.server.get_pg_conn")
    @patch("memory_service.server.redis_client")
    def test_not_found(self, mock_redis, mock_pg, servicer, mock_context):
        mock_redis.get.return_value = None

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        cursor.fetchone.return_value = None

        request = MagicMock()
        request.session_id = "nonexistent"

        servicer.GetSession(request, mock_context)
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)


class TestTouchSession:
    """Tests for TouchSession."""

    @patch("memory_service.server.get_pg_conn")
    @patch("memory_service.server.redis_client")
    def test_refreshes_ttl(self, mock_redis, mock_pg, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        mock_redis.get.return_value = json.dumps({
            "session_id": "sess-1",
            "customer_id": "cust-1",
            "created_at": "2024-01-01",
            "last_active_at": "2024-01-01",
        })

        request = MagicMock()
        request.session_id = "sess-1"

        response = servicer.TouchSession(request, mock_context)
        assert response.session_id == "sess-1"
        mock_redis.setex.assert_called_once()
        mock_redis.expire.assert_called_once()


class TestAddConversationTurn:
    """Tests for AddConversationTurn."""

    @patch("memory_service.server.get_pg_conn")
    @patch("memory_service.server.redis_client")
    def test_stores_in_pg_and_redis(self, mock_redis, mock_pg, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        request = MagicMock()
        request.session_id = "sess-1"
        request.role = "user"
        request.content = "Hello"
        request.intent = "general_question"
        request.confidence = 0.9
        request.tool_calls = ""

        response = servicer.AddConversationTurn(request, mock_context)
        assert len(response.turn_id) == 36
        conn.commit.assert_called_once()
        mock_redis.rpush.assert_called_once()
        mock_redis.expire.assert_called_once()


class TestGetConversationHistory:
    """Tests for GetConversationHistory."""

    @patch("memory_service.server.redis_client")
    def test_cache_hit(self, mock_redis, servicer, mock_context):
        turns = [
            json.dumps({"role": "user", "content": "Hi", "intent": "", "confidence": 0.0, "tool_calls": "", "created_at": "2024-01-01"}),
            json.dumps({"role": "assistant", "content": "Hello!", "intent": "", "confidence": 0.0, "tool_calls": "", "created_at": "2024-01-01"}),
        ]
        mock_redis.lrange.return_value = turns

        request = MagicMock()
        request.session_id = "sess-1"
        request.limit = 50

        response = servicer.GetConversationHistory(request, mock_context)
        assert len(response.turns) == 2
        assert response.turns[0].role == "user"

    @patch("memory_service.server.get_pg_conn")
    @patch("memory_service.server.redis_client")
    def test_cache_miss(self, mock_redis, mock_pg, servicer, mock_context):
        mock_redis.lrange.return_value = []

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchall.return_value = [
            {"role": "user", "content": "Test", "intent": None, "confidence": None, "tool_calls": None, "created_at": datetime(2024, 1, 1)},
        ]

        request = MagicMock()
        request.session_id = "sess-1"
        request.limit = 10

        response = servicer.GetConversationHistory(request, mock_context)
        assert len(response.turns) == 1

    @patch("memory_service.server.redis_client")
    def test_limit_applied(self, mock_redis, servicer, mock_context):
        turns = [json.dumps({"role": "user", "content": f"msg{i}", "intent": "", "confidence": 0.0, "tool_calls": "", "created_at": ""}) for i in range(10)]
        mock_redis.lrange.return_value = turns

        request = MagicMock()
        request.session_id = "sess-1"
        request.limit = 3

        response = servicer.GetConversationHistory(request, mock_context)
        assert len(response.turns) == 3


class TestStoreEpisodicMemory:
    """Tests for StoreEpisodicMemory."""

    @patch("memory_service.server.get_ts_conn")
    def test_stores_memory(self, mock_ts, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_ts.return_value = conn

        request = MagicMock()
        request.customer_id = "cust-1"
        request.session_id = "sess-1"
        request.event_type = "session_summary"
        request.summary = "User asked about widgets"
        request.key_topics = ["widgets"]
        request.resolution_status = "resolved"
        request.metadata = "{}"

        response = servicer.StoreEpisodicMemory(request, mock_context)
        assert len(response.memory_id) == 36
        conn.commit.assert_called_once()


class TestGetEpisodicMemories:
    """Tests for GetEpisodicMemories."""

    @patch("memory_service.server.get_ts_conn")
    def test_with_event_type_filter(self, mock_ts, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_ts.return_value = conn

        cursor.fetchall.return_value = [
            {
                "id": "mem-1", "event_type": "reflexion_insight",
                "summary": "Include prices", "key_topics": ["price"],
                "resolution_status": "resolved", "metadata": "{}",
                "created_at": datetime(2024, 1, 1),
            }
        ]

        request = MagicMock()
        request.customer_id = "cust-1"
        request.event_type = "reflexion_insight"
        request.limit = 5

        response = servicer.GetEpisodicMemories(request, mock_context)
        assert len(response.memories) == 1
        assert response.memories[0].event_type == "reflexion_insight"

    @patch("memory_service.server.get_ts_conn")
    def test_without_event_type_filter(self, mock_ts, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_ts.return_value = conn

        cursor.fetchall.return_value = []

        request = MagicMock()
        request.customer_id = "cust-1"
        request.event_type = ""
        request.limit = 5

        response = servicer.GetEpisodicMemories(request, mock_context)
        assert len(response.memories) == 0


class TestAudit:
    """Tests for AppendAuditEvent and GetSessionAuditTrail."""

    @patch("memory_service.server.get_ts_conn")
    def test_append_audit(self, mock_ts, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_ts.return_value = conn

        request = MagicMock()
        request.session_id = "sess-1"
        request.customer_id = "cust-1"
        request.event_type = "session_created"
        request.event_data = "{}"

        response = servicer.AppendAuditEvent(request, mock_context)
        assert len(response.event_id) == 36

    @patch("memory_service.server.get_ts_conn")
    def test_get_audit_trail(self, mock_ts, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_ts.return_value = conn

        cursor.fetchall.return_value = [
            {
                "id": "evt-1", "session_id": "sess-1", "customer_id": "cust-1",
                "event_type": "session_created", "event_data": "{}",
                "event_time": datetime(2024, 1, 1),
            }
        ]

        request = MagicMock()
        request.session_id = "sess-1"
        request.event_type = ""
        request.limit = 200

        response = servicer.GetSessionAuditTrail(request, mock_context)
        assert len(response.events) == 1

    @patch("memory_service.server.get_ts_conn")
    def test_audit_failure_swallowed(self, mock_ts, servicer):
        """_append_audit_internal should not raise even on DB error."""
        mock_ts.side_effect = Exception("DB down")

        # Should not raise
        event_id = servicer._append_audit_internal("sess-1", "cust-1", "test", "{}")
        assert len(event_id) == 36  # UUID still generated
