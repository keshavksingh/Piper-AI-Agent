"""Tests for gateway_server — HTTP endpoints and WebSocket protocol."""

import json
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

# Ensure the gateway can find its static/templates dirs by running from its directory
_gateway_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "gateway_server")
_orig_cwd = os.getcwd()
os.chdir(_gateway_dir)
from gateway_server.main import app, check_rate_limit, rate_limit_store
os.chdir(_orig_cwd)


@pytest.fixture(autouse=True)
def clear_rate_limits():
    """Clear rate limit store between tests."""
    rate_limit_store.clear()
    yield
    rate_limit_store.clear()


# ── HTTP Endpoint Tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "gateway"


@pytest.mark.asyncio
@patch("gateway_server.main.get_pg_conn")
async def test_login_success(mock_pg):
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_pg.return_value = conn

    from shared.auth import hash_password
    hashed = hash_password("correct_password")

    cursor.fetchone.return_value = {
        "id": "user-1",
        "email": "test@example.com",
        "password_hash": hashed,
        "display_name": "Test User",
        "is_active": True,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/login", json={
            "email": "test@example.com",
            "password": "correct_password",
        })

    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    assert data["user"]["email"] == "test@example.com"


@pytest.mark.asyncio
@patch("gateway_server.main.get_pg_conn")
async def test_login_invalid_password(mock_pg):
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_pg.return_value = conn

    from shared.auth import hash_password
    cursor.fetchone.return_value = {
        "id": "user-1",
        "email": "test@example.com",
        "password_hash": hash_password("real_password"),
        "display_name": "Test User",
        "is_active": True,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/login", json={
            "email": "test@example.com",
            "password": "wrong_password",
        })

    assert response.status_code == 401


@pytest.mark.asyncio
@patch("gateway_server.main.get_pg_conn")
async def test_login_user_not_found(mock_pg):
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_pg.return_value = conn
    cursor.fetchone.return_value = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/login", json={
            "email": "nobody@example.com",
            "password": "anything",
        })

    assert response.status_code == 401


@pytest.mark.asyncio
@patch("gateway_server.main.get_pg_conn")
async def test_login_account_disabled(mock_pg):
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_pg.return_value = conn

    from shared.auth import hash_password
    cursor.fetchone.return_value = {
        "id": "user-1",
        "email": "disabled@example.com",
        "password_hash": hash_password("password"),
        "display_name": "Disabled User",
        "is_active": False,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/login", json={
            "email": "disabled@example.com",
            "password": "password",
        })

    assert response.status_code == 401
    assert "disabled" in response.json()["error"].lower()


@pytest.mark.asyncio
async def test_login_missing_fields():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/login", json={"email": ""})

    assert response.status_code == 400


# ── WebSocket Tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("gateway_server.main.get_rec_stub")
@patch("gateway_server.main.get_memory_stub")
async def test_ws_session_start_success(mock_mem, mock_rec):
    from shared.auth import create_jwt

    token = create_jwt("user-1", "test@example.com")

    mem_stub = MagicMock()
    mock_mem.return_value = mem_stub
    mem_stub.CreateSession.return_value = MagicMock(session_id="sess-new")

    rec_stub = MagicMock()
    mock_rec.return_value = rec_stub
    rec_stub.GetStartRecommendations.return_value = MagicMock(suggestions=["Q1", "Q2"])

    from starlette.testclient import TestClient
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "session_start", "token": token}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "session_ready"
            assert "session_id" in resp


@pytest.mark.asyncio
async def test_ws_no_token():
    from starlette.testclient import TestClient
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "session_start", "token": ""}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "auth_error"
            assert resp["code"] == "AUTH_REQUIRED"


@pytest.mark.asyncio
async def test_ws_bad_token():
    from starlette.testclient import TestClient
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "session_start", "token": "invalid.jwt.token"}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "auth_error"
            assert resp["code"] == "AUTH_INVALID"


@pytest.mark.asyncio
async def test_ws_no_session():
    from starlette.testclient import TestClient
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "user_message", "text": "Hello"}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "error"
            assert resp["code"] == "NO_SESSION"


@pytest.mark.asyncio
async def test_ws_unknown_type():
    from starlette.testclient import TestClient
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "foobar"}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "error"
            assert resp["code"] == "UNKNOWN_TYPE"


@pytest.mark.asyncio
async def test_ws_invalid_json():
    from starlette.testclient import TestClient
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text("not valid json {{{")
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "error"
            assert resp["code"] == "INVALID_FORMAT"


@pytest.mark.asyncio
@patch("gateway_server.main.get_agent_stub")
@patch("gateway_server.main.get_rec_stub")
@patch("gateway_server.main.get_memory_stub")
async def test_ws_user_message_flow(mock_mem, mock_rec, mock_agent):
    from shared.auth import create_jwt

    token = create_jwt("user-1", "test@example.com")

    mem_stub = MagicMock()
    mock_mem.return_value = mem_stub
    mem_stub.CreateSession.return_value = MagicMock(session_id="sess-msg")

    rec_stub = MagicMock()
    mock_rec.return_value = rec_stub
    rec_stub.GetStartRecommendations.return_value = MagicMock(suggestions=[])

    agent_stub = MagicMock()
    mock_agent.return_value = agent_stub
    # Simulate agent returning token + response_complete events
    event1 = MagicMock()
    event1.type = "token"
    event1.payload = "Hello "
    event2 = MagicMock()
    event2.type = "response_complete"
    event2.payload = '{"response": {"text": "Hello"}, "recommendations": []}'
    agent_stub.ProcessQuery.return_value = [event1, event2]

    from starlette.testclient import TestClient
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as ws:
            # Start session first
            ws.send_text(json.dumps({"type": "session_start", "token": token}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "session_ready"
            # Consume recommendations message
            ws.receive_text()

            # Send user message
            ws.send_text(json.dumps({"type": "user_message", "text": "Hello"}))
            # Should receive the streamed events
            resp1 = json.loads(ws.receive_text())
            assert resp1["type"] == "token"
            resp2 = json.loads(ws.receive_text())
            assert resp2["type"] == "response_complete"


@pytest.mark.asyncio
@patch("gateway_server.main.get_rec_stub")
@patch("gateway_server.main.get_memory_stub")
async def test_ws_rate_limited(mock_mem, mock_rec):
    from shared.auth import create_jwt

    token = create_jwt("rate-user", "rate@example.com")

    mem_stub = MagicMock()
    mock_mem.return_value = mem_stub
    mem_stub.CreateSession.return_value = MagicMock(session_id="sess-rate")

    rec_stub = MagicMock()
    mock_rec.return_value = rec_stub
    rec_stub.GetStartRecommendations.return_value = MagicMock(suggestions=[])

    from starlette.testclient import TestClient
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as ws:
            # Start session
            ws.send_text(json.dumps({"type": "session_start", "token": token}))
            ws.receive_text()  # session_ready
            ws.receive_text()  # recommendations

            # Fill up rate limit
            import gateway_server.main as gw
            original_limit = gw.Config.RATE_LIMIT_PER_MINUTE
            gw.Config.RATE_LIMIT_PER_MINUTE = 1
            rate_limit_store.clear()
            check_rate_limit("rate-user")  # use up the one allowed request

            ws.send_text(json.dumps({"type": "user_message", "text": "test"}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "error"
            assert resp["code"] == "RATE_LIMITED"

            gw.Config.RATE_LIMIT_PER_MINUTE = original_limit


# ── Rate Limiting Tests ──────────────────────────────────────────────────────


class TestRateLimiting:
    def test_allows_under_limit(self, monkeypatch):
        monkeypatch.setattr("gateway_server.main.Config.RATE_LIMIT_PER_MINUTE", 5)
        for _ in range(5):
            assert check_rate_limit("cust-rate") is True

    def test_blocks_over_limit(self, monkeypatch):
        monkeypatch.setattr("gateway_server.main.Config.RATE_LIMIT_PER_MINUTE", 3)
        for _ in range(3):
            check_rate_limit("cust-limited")
        assert check_rate_limit("cust-limited") is False
