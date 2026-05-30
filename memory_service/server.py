"""Memory Service — Redis (short-term) + PostgreSQL (relational) + TimescaleDB (immutable episodic + audit)."""

import json
import uuid
import sys
import threading
import grpc
from concurrent import futures
from datetime import datetime

import redis
import psycopg2
import psycopg2.extras
import psycopg2.pool
from google.protobuf.struct_pb2 import Struct
from google.protobuf import json_format

sys.path.append("..")

import protos.memory_service_pb2 as pb2
import protos.memory_service_pb2_grpc as pb2_grpc
from shared.config import Config
from shared.logging_config import setup_logging

log = setup_logging("memory_service")

# ── Redis Connection ──────────────────────────────────────────────

_redis_client = None
_redis_lock = threading.Lock()


def _get_redis():
    global _redis_client
    if _redis_client is None:
        with _redis_lock:
            if _redis_client is None:
                _redis_client = redis.from_url(
                    Config.REDIS_URL, decode_responses=True,
                    socket_timeout=Config.REDIS_SOCKET_TIMEOUT,
                    socket_connect_timeout=Config.REDIS_CONNECT_TIMEOUT,
                )
    return _redis_client


# ── PostgreSQL Connection Pool (sessions, conversation turns) ─────

_pg_pool = None
_pg_lock = threading.Lock()


def _get_pool():
    global _pg_pool
    if _pg_pool is None:
        with _pg_lock:
            if _pg_pool is None:
                _pg_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=2, maxconn=10, dsn=Config.DATABASE_URL,
                    connect_timeout=Config.DB_CONNECT_TIMEOUT,
                    options=f"-c statement_timeout={Config.DB_STATEMENT_TIMEOUT_MS}",
                )
    return _pg_pool


def get_pg_conn():
    return _get_pool().getconn()


def put_pg_conn(conn):
    try:
        conn.rollback()
    except Exception:
        pass
    if _pg_pool is not None:
        _pg_pool.putconn(conn)


# ── TimescaleDB Connection Pool (episodic memories, audit trail) ──

_ts_pool = None
_ts_lock = threading.Lock()


def _create_ts_pool(retries=5, delay=2):
    for attempt in range(1, retries + 1):
        try:
            return psycopg2.pool.ThreadedConnectionPool(
                minconn=2, maxconn=10, dsn=Config.TIMESCALEDB_URL,
                connect_timeout=Config.DB_CONNECT_TIMEOUT,
                options=f"-c statement_timeout={Config.DB_STATEMENT_TIMEOUT_MS}",
            )
        except psycopg2.OperationalError:
            if attempt == retries:
                raise
            log.warning("timescaledb_connect_retry", attempt=attempt, max_retries=retries)
            import time; time.sleep(delay)


def _get_ts_pool():
    global _ts_pool
    if _ts_pool is None:
        with _ts_lock:
            if _ts_pool is None:
                _ts_pool = _create_ts_pool()
    return _ts_pool


def get_ts_conn():
    return _get_ts_pool().getconn()


def put_ts_conn(conn):
    try:
        conn.rollback()
    except Exception:
        pass
    if _ts_pool is not None:
        _ts_pool.putconn(conn)


# ── Helpers ───────────────────────────────────────────────────────

def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def _turns_key(session_id: str) -> str:
    return f"session:{session_id}:turns"


# ── gRPC Service Implementation ──────────────────────────────────

class MemoryServiceServicer(pb2_grpc.MemoryServiceServicer):

    # ══════════════════════════════════════════════════════════════
    # Session Management (PostgreSQL + Redis cache)
    # ══════════════════════════════════════════════════════════════

    def CreateSession(self, request, context):
        customer_id = request.customer_id
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Store in PostgreSQL
        conn = get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sessions (id, customer_id, created_at, last_active_at) "
                    "VALUES (%s, %s, %s, %s)",
                    (session_id, customer_id, now, now),
                )
            conn.commit()
        finally:
            put_pg_conn(conn)

        # Cache in Redis (non-critical — log and continue on failure)
        try:
            session_data = json.dumps({
                "session_id": session_id,
                "customer_id": customer_id,
                "created_at": now,
                "last_active_at": now,
            })
            _get_redis().setex(
                _session_key(session_id),
                Config.SESSION_TTL_SECONDS,
                session_data,
            )
        except Exception as e:
            log.warning("redis_cache_session_failed", error=str(e), session_id=session_id)

        log.info("session_created", session_id=session_id, customer_id=customer_id)

        # Append audit event (fire-and-forget)
        self._append_audit_internal(session_id, customer_id, "session_created", {})

        return pb2.SessionResponse(
            session_id=session_id,
            customer_id=customer_id,
            created_at=now,
            last_active_at=now,
            is_new=True,
        )

    def GetSession(self, request, context):
        session_id = request.session_id

        # Try Redis first (non-critical — fall through to PG on any failure)
        try:
            cached = _get_redis().get(_session_key(session_id))
            if cached:
                data = json.loads(cached)
                try:
                    _get_redis().expire(_session_key(session_id), Config.SESSION_TTL_SECONDS)
                except Exception:
                    pass  # TTL refresh is non-critical
                return pb2.SessionResponse(
                    session_id=data["session_id"],
                    customer_id=data["customer_id"],
                    created_at=data["created_at"],
                    last_active_at=data["last_active_at"],
                    is_new=False,
                )
        except Exception as e:
            log.warning("redis_get_session_failed", error=str(e), session_id=session_id)

        # Fallback to PostgreSQL
        conn = get_pg_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    "SELECT id, customer_id, created_at, last_active_at "
                    "FROM sessions WHERE id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
        finally:
            put_pg_conn(conn)

        if not row:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Session {session_id} not found")
            return pb2.SessionResponse()

        # Re-cache in Redis (non-critical)
        try:
            session_data = json.dumps({
                "session_id": str(row["id"]),
                "customer_id": row["customer_id"],
                "created_at": row["created_at"].isoformat(),
                "last_active_at": row["last_active_at"].isoformat(),
            })
            _get_redis().setex(
                _session_key(session_id),
                Config.SESSION_TTL_SECONDS,
                session_data,
            )
        except Exception as e:
            log.warning("redis_recache_session_failed", error=str(e), session_id=session_id)

        return pb2.SessionResponse(
            session_id=str(row["id"]),
            customer_id=row["customer_id"],
            created_at=row["created_at"].isoformat(),
            last_active_at=row["last_active_at"].isoformat(),
            is_new=False,
        )

    def TouchSession(self, request, context):
        session_id = request.session_id
        now = datetime.utcnow().isoformat()

        # Update PostgreSQL
        conn = get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET last_active_at = %s WHERE id = %s",
                    (now, session_id),
                )
            conn.commit()
        finally:
            put_pg_conn(conn)

        # Refresh Redis TTL and update last_active_at (non-critical)
        try:
            cached = _get_redis().get(_session_key(session_id))
            if cached:
                data = json.loads(cached)
                data["last_active_at"] = now
                _get_redis().setex(
                    _session_key(session_id),
                    Config.SESSION_TTL_SECONDS,
                    json.dumps(data),
                )

            _get_redis().expire(_turns_key(session_id), Config.SESSION_TTL_SECONDS)
        except Exception as e:
            log.warning("redis_touch_session_failed", error=str(e), session_id=session_id)

        return pb2.SessionResponse(session_id=session_id, last_active_at=now)

    # ══════════════════════════════════════════════════════════════
    # Conversation Turns (PostgreSQL + Redis cache)
    # ══════════════════════════════════════════════════════════════

    def AddConversationTurn(self, request, context):
        session_id = request.session_id
        turn_id = str(uuid.uuid4())

        # Serialize repeated ToolCall messages to JSON for DB and Redis
        tool_calls_list = []
        for tc in request.tool_calls:
            tool_calls_list.append({
                "tool": tc.tool_name,
                "arguments": json_format.MessageToDict(tc.arguments) if tc.arguments.fields else {},
                "result": tc.result,
            })
        tool_calls_json = json.dumps(tool_calls_list) if tool_calls_list else "[]"

        turn_data = {
            "turn_id": turn_id,
            "role": request.role,
            "content": request.content,
            "intent": request.intent,
            "confidence": request.confidence,
            "tool_calls": tool_calls_json,
            "created_at": datetime.utcnow().isoformat(),
        }

        # Store in PostgreSQL
        conn = get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversation_turns "
                    "(id, session_id, role, content, intent, confidence, tool_calls) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        turn_id,
                        session_id,
                        request.role,
                        request.content,
                        request.intent or None,
                        request.confidence or None,
                        tool_calls_json,
                    ),
                )
            conn.commit()
        finally:
            put_pg_conn(conn)

        # Append to Redis list (non-critical)
        try:
            _get_redis().rpush(_turns_key(session_id), json.dumps(turn_data))
            _get_redis().expire(_turns_key(session_id), Config.SESSION_TTL_SECONDS)
        except Exception as e:
            log.warning("redis_append_turn_failed", error=str(e), session_id=session_id)

        log.info("turn_added", session_id=session_id, turn_id=turn_id, role=request.role)

        return pb2.AddTurnResponse(turn_id=turn_id)

    def GetConversationHistory(self, request, context):
        session_id = request.session_id
        limit = request.limit if request.limit > 0 else 50

        # Try Redis first (non-critical — fall through to PG on any failure)
        try:
            cached_turns = _get_redis().lrange(_turns_key(session_id), 0, -1)
            if cached_turns:
                turns = [json.loads(t) for t in cached_turns]
                if limit > 0:
                    turns = turns[-limit:]
                return pb2.GetHistoryResponse(
                    turns=[
                        pb2.ConversationTurn(
                            role=t["role"],
                            content=t["content"],
                            intent=t.get("intent", ""),
                            confidence=t.get("confidence", 0.0),
                            tool_calls=self._parse_tool_calls_from_json(t.get("tool_calls", "[]")),
                            created_at=t.get("created_at", ""),
                        )
                        for t in turns
                    ]
                )
        except Exception as e:
            log.warning("redis_get_history_failed", error=str(e), session_id=session_id)

        # Fallback to PostgreSQL
        # Use subquery to get the NEWEST N turns, then re-sort ASC for chronological order.
        conn = get_pg_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    "SELECT role, content, intent, confidence, tool_calls, created_at FROM ("
                    "  SELECT role, content, intent, confidence, tool_calls, created_at "
                    "  FROM conversation_turns WHERE session_id = %s "
                    "  ORDER BY created_at DESC LIMIT %s"
                    ") sub ORDER BY created_at ASC",
                    (session_id, limit),
                )
                rows = cur.fetchall()
        finally:
            put_pg_conn(conn)

        return pb2.GetHistoryResponse(
            turns=[
                pb2.ConversationTurn(
                    role=row["role"],
                    content=row["content"],
                    intent=row["intent"] or "",
                    confidence=row["confidence"] or 0.0,
                    tool_calls=self._parse_tool_calls_from_db(row["tool_calls"]),
                    created_at=row["created_at"].isoformat(),
                )
                for row in rows
            ]
        )

    # ── Tool-call serialization helpers ────────────────────────────

    def _parse_tool_calls_from_json(self, raw):
        """Parse a JSON string (from Redis) into repeated ToolCall messages."""
        try:
            items = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except (json.JSONDecodeError, TypeError):
            return []
        return self._items_to_tool_calls(items)

    def _parse_tool_calls_from_db(self, raw):
        """Parse a DB column (str or list) into repeated ToolCall messages."""
        if isinstance(raw, str):
            try:
                items = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return []
        elif isinstance(raw, list):
            items = raw
        else:
            return []
        return self._items_to_tool_calls(items)

    @staticmethod
    def _items_to_tool_calls(items):
        """Convert a list of dicts to ToolCall protobuf messages."""
        if not isinstance(items, list):
            return []
        result = []
        for item in items:
            if not isinstance(item, dict):
                continue
            tc = pb2.ToolCall(
                tool_name=item.get("tool", item.get("tool_name", "")),
                result=item.get("result", ""),
            )
            args = item.get("arguments", item.get("args", {}))
            if isinstance(args, dict) and args:
                json_format.ParseDict(args, tc.arguments)
            result.append(tc)
        return result

    # ══════════════════════════════════════════════════════════════
    # Episodic Memory (TimescaleDB — immutable append-only)
    # ══════════════════════════════════════════════════════════════

    def StoreEpisodicMemory(self, request, context):
        memory_id = str(uuid.uuid4())

        # Convert Struct metadata to JSON string for DB storage
        if request.metadata.fields:
            metadata_json = json.dumps(json_format.MessageToDict(request.metadata))
        else:
            metadata_json = "{}"

        conn = get_ts_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO episodic_memories "
                    "(id, customer_id, session_id, event_type, summary, key_topics, "
                    " resolution_status, metadata) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        memory_id,
                        request.customer_id,
                        request.session_id or None,
                        request.event_type or "session_summary",
                        request.summary,
                        list(request.key_topics),
                        request.resolution_status or "resolved",
                        metadata_json,
                    ),
                )
            conn.commit()
        finally:
            put_ts_conn(conn)

        log.info(
            "episodic_memory_stored",
            memory_id=memory_id,
            customer_id=request.customer_id,
            event_type=request.event_type or "session_summary",
        )

        return pb2.StoreEpisodicResponse(memory_id=memory_id)

    def GetEpisodicMemories(self, request, context):
        limit = request.limit or 5

        conn = get_ts_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                if request.event_type:
                    cur.execute(
                        "SELECT id, event_type, summary, key_topics, resolution_status, "
                        "       metadata, created_at "
                        "FROM episodic_memories "
                        "WHERE customer_id = %s AND event_type = %s "
                        "ORDER BY created_at DESC LIMIT %s",
                        (request.customer_id, request.event_type, limit),
                    )
                else:
                    cur.execute(
                        "SELECT id, event_type, summary, key_topics, resolution_status, "
                        "       metadata, created_at "
                        "FROM episodic_memories WHERE customer_id = %s "
                        "ORDER BY created_at DESC LIMIT %s",
                        (request.customer_id, limit),
                    )
                rows = cur.fetchall()
        finally:
            put_ts_conn(conn)

        memories = []
        for row in rows:
            mem = pb2.EpisodicMemory(
                memory_id=str(row["id"]),
                event_type=row["event_type"],
                summary=row["summary"],
                key_topics=row["key_topics"] or [],
                resolution_status=row["resolution_status"],
                created_at=row["created_at"].isoformat(),
            )
            # Parse metadata from DB into Struct
            meta_raw = row["metadata"]
            if isinstance(meta_raw, dict):
                meta_dict = meta_raw
            elif isinstance(meta_raw, str) and meta_raw:
                try:
                    meta_dict = json.loads(meta_raw)
                except (json.JSONDecodeError, TypeError):
                    meta_dict = {}
            else:
                meta_dict = {}
            if meta_dict:
                json_format.ParseDict(meta_dict, mem.metadata)
            memories.append(mem)

        return pb2.GetEpisodicResponse(memories=memories)

    # ══════════════════════════════════════════════════════════════
    # Session Audit Trail (TimescaleDB — immutable append-only)
    # ══════════════════════════════════════════════════════════════

    def AppendAuditEvent(self, request, context):
        # Convert Struct event_data to dict for internal helper
        if request.event_data.fields:
            event_data = json_format.MessageToDict(request.event_data)
        else:
            event_data = {}
        event_id = self._append_audit_internal(
            request.session_id,
            request.customer_id,
            request.event_type,
            event_data,
        )
        return pb2.AuditEventResponse(event_id=event_id)

    def GetSessionAuditTrail(self, request, context):
        session_id = request.session_id
        limit = request.limit or 200

        conn = get_ts_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                if request.event_type:
                    cur.execute(
                        "SELECT id, session_id, customer_id, event_type, event_data, event_time "
                        "FROM session_audit_trail "
                        "WHERE session_id = %s AND event_type = %s "
                        "ORDER BY event_time ASC LIMIT %s",
                        (session_id, request.event_type, limit),
                    )
                else:
                    cur.execute(
                        "SELECT id, session_id, customer_id, event_type, event_data, event_time "
                        "FROM session_audit_trail "
                        "WHERE session_id = %s "
                        "ORDER BY event_time ASC LIMIT %s",
                        (session_id, limit),
                    )
                rows = cur.fetchall()
        finally:
            put_ts_conn(conn)

        events = []
        for row in rows:
            evt = pb2.AuditEvent(
                event_id=str(row["id"]),
                session_id=str(row["session_id"]),
                customer_id=row["customer_id"],
                event_type=row["event_type"],
                event_time=row["event_time"].isoformat(),
            )
            # Parse event_data from DB into Struct
            ed_raw = row["event_data"]
            if isinstance(ed_raw, dict):
                ed_dict = ed_raw
            elif isinstance(ed_raw, str) and ed_raw:
                try:
                    ed_dict = json.loads(ed_raw)
                except (json.JSONDecodeError, TypeError):
                    ed_dict = {}
            else:
                ed_dict = {}
            if ed_dict:
                json_format.ParseDict(ed_dict, evt.event_data)
            events.append(evt)

        return pb2.GetAuditTrailResponse(events=events)

    # ── Internal helper for audit writes ──────────────────────────

    def _append_audit_internal(self, session_id, customer_id, event_type, event_data):
        """Append an immutable event to the TimescaleDB audit trail.
        Returns the event_id. Non-critical — failures are logged but not propagated."""
        event_id = str(uuid.uuid4())
        try:
            # Normalise event_data to a JSON string for the DB
            if isinstance(event_data, str):
                data_json = event_data if event_data else "{}"
            elif isinstance(event_data, dict):
                data_json = json.dumps(event_data)
            else:
                data_json = "{}"

            conn = get_ts_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO session_audit_trail "
                        "(id, session_id, customer_id, event_type, event_data) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (event_id, session_id, customer_id, event_type, data_json),
                    )
                conn.commit()
            finally:
                put_ts_conn(conn)

            log.info("audit_event_appended", event_id=event_id, event_type=event_type, session_id=session_id)
        except Exception as e:
            log.warning("audit_append_failed", error=str(e), event_type=event_type)

        return event_id


# ── Server Startup ────────────────────────────────────────────────

def serve():
    with open(Config.TLS_SERVER_CERT, "rb") as f:
        server_cert = f.read()
    with open(Config.TLS_SERVER_KEY, "rb") as f:
        server_key = f.read()
    with open(Config.TLS_CA_CERT, "rb") as f:
        ca_cert = f.read()
    credentials = grpc.ssl_server_credentials(
        [(server_key, server_cert)],
        root_certificates=ca_cert,
        require_client_auth=False,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MemoryServiceServicer_to_server(MemoryServiceServicer(), server)
    server.add_secure_port("[::]:50055", credentials)
    server.start()
    log.info("server_started", port=50055, tls=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
