"""Gateway Service — FastAPI + WebSocket with JWT auth, structured JSON protocol, rate limiting."""

import asyncio
import json
import time
import sys
import threading
from collections import defaultdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import grpc
import psycopg2
import psycopg2.extras

sys.path.append("..")

import protos.agent_service_pb2 as agent_pb2
import protos.agent_service_pb2_grpc as agent_pb2_grpc
import protos.memory_service_pb2 as memory_pb2
import protos.memory_service_pb2_grpc as memory_pb2_grpc
import protos.recommendation_service_pb2 as rec_pb2
import protos.recommendation_service_pb2_grpc as rec_pb2_grpc

from shared.config import Config
from shared.logging_config import setup_logging
from shared.resilience import create_grpc_channel
from shared.auth import verify_password, create_jwt, verify_jwt

log = setup_logging("gateway_service")

app = FastAPI(title="Piper AI Agent Gateway")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Rate Limiting (in-memory, per customer) ───────────────────────

rate_limit_store: dict = defaultdict(list)


def check_rate_limit(customer_id: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.time()
    window = 60  # 1 minute window
    max_requests = Config.RATE_LIMIT_PER_MINUTE

    # Clean old entries and update in place
    timestamps = [ts for ts in rate_limit_store.get(customer_id, []) if now - ts < window]

    if len(timestamps) >= max_requests:
        rate_limit_store[customer_id] = timestamps
        return False

    # Allowed — add new entry
    timestamps.append(now)
    rate_limit_store[customer_id] = timestamps
    return True


# ── Database Connection ──────────────────────────────────────────

def get_pg_conn():
    return psycopg2.connect(Config.DATABASE_URL)


# ── gRPC Stubs (cached channels to prevent resource leaks) ────────

_channel_cache = {}
_channel_lock = threading.Lock()


def _get_cached_channel(addr):
    """Return a cached gRPC channel for the given address, creating one if needed."""
    if addr in _channel_cache:
        return _channel_cache[addr]
    with _channel_lock:
        if addr not in _channel_cache:
            _channel_cache[addr] = create_grpc_channel(addr)
        return _channel_cache[addr]


def get_agent_stub():
    channel = _get_cached_channel(Config.AGENT_SERVICE_ADDR)
    return agent_pb2_grpc.AgentServiceStub(channel)


def get_memory_stub():
    channel = _get_cached_channel(Config.MEMORY_SERVICE_ADDR)
    return memory_pb2_grpc.MemoryServiceStub(channel)


def get_rec_stub():
    channel = _get_cached_channel(Config.RECOMMENDATION_SERVICE_ADDR)
    return rec_pb2_grpc.RecommendationServiceStub(channel)


# ── Async gRPC Stream Helper ──────────────────────────────────────

_STREAM_SENTINEL = object()


async def async_grpc_stream(sync_iterator):
    """Wrap a blocking gRPC response iterator so each event yields to the async loop.

    Runs the synchronous gRPC iteration in a background thread and feeds events
    into an asyncio.Queue.  On consumer exit (normal completion, client disconnect,
    or any exception), the gRPC call is cancelled so the agent stops processing.
    """
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    cancelled = threading.Event()

    def _consume():
        try:
            for event in sync_iterator:
                if cancelled.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            if not cancelled.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _STREAM_SENTINEL)

    thread = threading.Thread(target=_consume, daemon=True)
    thread.start()

    try:
        while True:
            item = await queue.get()
            if item is _STREAM_SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        # Consumer exited (disconnect, error, or normal end) — stop the producer
        cancelled.set()
        if hasattr(sync_iterator, 'cancel'):
            sync_iterator.cancel()


# ── HTTP Endpoints ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy", "service": "gateway"})


@app.post("/api/login")
async def login(request: Request):
    """Authenticate user with email/password, return JWT token."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid request body"}, status_code=400)

    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    if not email or not password:
        return JSONResponse({"error": "Email and password are required"}, status_code=400)

    conn = get_pg_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT id, email, password_hash, display_name, is_active "
                "FROM users WHERE LOWER(email) = %s",
                (email,),
            )
            user = cur.fetchone()
    finally:
        conn.close()

    if not user:
        log.warning("login_failed", email=email, reason="user_not_found")
        return JSONResponse({"error": "Invalid email or password"}, status_code=401)

    if not user["is_active"]:
        log.warning("login_failed", email=email, reason="account_disabled")
        return JSONResponse({"error": "Account is disabled"}, status_code=401)

    if not verify_password(password, user["password_hash"]):
        log.warning("login_failed", email=email, reason="bad_password")
        return JSONResponse({"error": "Invalid email or password"}, status_code=401)

    # Update last_login_at (non-critical — don't fail login if this fails)
    try:
        conn = get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET last_login_at = NOW() WHERE id = %s", (str(user["id"]),))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        log.warning("login_update_last_login_failed", error=str(e), user_id=str(user["id"]))

    token = create_jwt(str(user["id"]), user["email"])

    log.info("login_success", email=email, user_id=str(user["id"]))

    return JSONResponse({
        "token": token,
        "user": {
            "id": str(user["id"]),
            "email": user["email"],
            "display_name": user["display_name"],
        },
    })


# ── WebSocket Endpoint ────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    session_id = None
    customer_id = None

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON message",
                    "code": "INVALID_FORMAT",
                }))
                continue

            msg_type = message.get("type", "")

            # ── Session Start ─────────────────────────────────────
            if msg_type == "session_start":
                token = message.get("token", "")

                # Verify JWT token
                if not token:
                    await websocket.send_text(json.dumps({
                        "type": "auth_error",
                        "message": "Authentication required. Please log in.",
                        "code": "AUTH_REQUIRED",
                    }))
                    continue

                try:
                    claims = verify_jwt(token)
                    customer_id = claims["user_id"]
                except Exception:
                    await websocket.send_text(json.dumps({
                        "type": "auth_error",
                        "message": "Invalid or expired token. Please log in again.",
                        "code": "AUTH_INVALID",
                    }))
                    continue

                try:
                    memory_stub = get_memory_stub()
                    session_resp = memory_stub.CreateSession(
                        memory_pb2.CreateSessionRequest(customer_id=customer_id)
                    )
                    session_id = session_resp.session_id

                    await websocket.send_text(json.dumps({
                        "type": "session_ready",
                        "session_id": session_id,
                    }))

                    # Send recommendations
                    try:
                        rec_stub = get_rec_stub()
                        rec_resp = rec_stub.GetStartRecommendations(
                            rec_pb2.StartRecommendationRequest(
                                customer_id=customer_id,
                                session_id=session_id,
                            )
                        )
                        await websocket.send_text(json.dumps({
                            "type": "recommendations",
                            "suggestions": list(rec_resp.suggestions),
                        }))
                    except Exception as e:
                        log.warning("recommendations_failed", error=str(e))

                except Exception as e:
                    log.error("session_create_failed", error=str(e))
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Failed to create session",
                        "code": "SESSION_ERROR",
                    }))

            # ── User Message ──────────────────────────────────────
            elif msg_type == "user_message":
                if not session_id:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Session not started. Send session_start first.",
                        "code": "NO_SESSION",
                    }))
                    continue

                # Rate limit check
                if not check_rate_limit(customer_id or "unknown"):
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Too many requests. Please wait.",
                        "code": "RATE_LIMITED",
                    }))
                    continue

                query = message.get("text", "")
                if not query.strip():
                    continue

                log.info("user_message", session_id=session_id, query=query[:100])

                try:
                    agent_stub = get_agent_stub()
                    agent_request = agent_pb2.AgentRequest(
                        session_id=session_id,
                        customer_id=customer_id or "",
                        query=query,
                    )

                    # Signal client that processing has begun
                    await websocket.send_text(json.dumps({
                        "type": "processing_started",
                        "payload": json.dumps({"query": query[:100]}),
                    }))

                    # Stream agent events to the client (async to avoid blocking the event loop)
                    async for event in async_grpc_stream(agent_stub.ProcessQuery(agent_request)):
                        await websocket.send_text(json.dumps({
                            "type": event.type,
                            "payload": event.payload,
                        }))

                except grpc.RpcError as e:
                    log.error("agent_call_failed", error=str(e))
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Agent service unavailable",
                        "code": "AGENT_ERROR",
                    }))

            # ── Clarification Response ────────────────────────────
            elif msg_type == "clarification_response":
                if not session_id:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Session not started. Send session_start first.",
                        "code": "NO_SESSION",
                    }))
                    continue

                selected = message.get("selected_option", "")
                freetext = message.get("freetext", "")

                try:
                    agent_stub = get_agent_stub()
                    clarification = agent_pb2.ClarificationResponse(
                        session_id=session_id,
                        selected_option=selected,
                        freetext=freetext,
                    )

                    # Signal client that processing has begun
                    await websocket.send_text(json.dumps({
                        "type": "processing_started",
                        "payload": json.dumps({"clarification": True}),
                    }))

                    async for event in async_grpc_stream(agent_stub.SubmitClarification(clarification)):
                        await websocket.send_text(json.dumps({
                            "type": event.type,
                            "payload": event.payload,
                        }))

                except grpc.RpcError as e:
                    log.error("clarification_failed", error=str(e))
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Failed to process clarification",
                        "code": "AGENT_ERROR",
                    }))

            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "code": "UNKNOWN_TYPE",
                }))

    except WebSocketDisconnect:
        log.info("websocket_disconnected", session_id=session_id)
