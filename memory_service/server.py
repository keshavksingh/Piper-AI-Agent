"""Memory Service — Redis (short-term) + PostgreSQL (relational) + TimescaleDB (immutable episodic + audit)."""

import json
import uuid
import sys
import grpc
from concurrent import futures
from datetime import datetime

import redis
import psycopg2
import psycopg2.extras
import psycopg2.pool

sys.path.append("..")

import protos.memory_service_pb2 as pb2
import protos.memory_service_pb2_grpc as pb2_grpc
from shared.config import Config
from shared.logging_config import setup_logging

log = setup_logging("memory_service")

# ── Redis Connection ──────────────────────────────────────────────

redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)


# ── PostgreSQL Connection Pool (sessions, conversation turns) ─────

_pg_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=2, maxconn=10, dsn=Config.DATABASE_URL
)


def get_pg_conn():
    return _pg_pool.getconn()


def put_pg_conn(conn):
    try:
        conn.rollback()
    except Exception:
        pass
    _pg_pool.putconn(conn)


# ── TimescaleDB Connection Pool (episodic memories, audit trail) ──

def _create_ts_pool(retries=5, delay=2):
    for attempt in range(1, retries + 1):
        try:
            return psycopg2.pool.ThreadedConnectionPool(
                minconn=2, maxconn=10, dsn=Config.TIMESCALEDB_URL
            )
        except psycopg2.OperationalError:
            if attempt == retries:
                raise
            log.warning("timescaledb_connect_retry", attempt=attempt, max_retries=retries)
            import time; time.sleep(delay)

_ts_pool = _create_ts_pool()


def get_ts_conn():
    return _ts_pool.getconn()


def put_ts_conn(conn):
    try:
        conn.rollback()
    except Exception:
        pass
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
            redis_client.setex(
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
            cached = redis_client.get(_session_key(session_id))
            if cached:
                data = json.loads(cached)
                try:
                    redis_client.expire(_session_key(session_id), Config.SESSION_TTL_SECONDS)
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
            redis_client.setex(
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
            cached = redis_client.get(_session_key(session_id))
            if cached:
                data = json.loads(cached)
                data["last_active_at"] = now
                redis_client.setex(
                    _session_key(session_id),
                    Config.SESSION_TTL_SECONDS,
                    json.dumps(data),
                )

            redis_client.expire(_turns_key(session_id), Config.SESSION_TTL_SECONDS)
        except Exception as e:
            log.warning("redis_touch_session_failed", error=str(e), session_id=session_id)

        return pb2.SessionResponse(session_id=session_id, last_active_at=now)

    # ══════════════════════════════════════════════════════════════
    # Conversation Turns (PostgreSQL + Redis cache)
    # ══════════════════════════════════════════════════════════════

    def AddConversationTurn(self, request, context):
        session_id = request.session_id
        turn_id = str(uuid.uuid4())

        turn_data = {
            "turn_id": turn_id,
            "role": request.role,
            "content": request.content,
            "intent": request.intent,
            "confidence": request.confidence,
            "tool_calls": request.tool_calls,
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
                        request.tool_calls or "[]",
                    ),
                )
            conn.commit()
        finally:
            put_pg_conn(conn)

        # Append to Redis list (non-critical)
        try:
            redis_client.rpush(_turns_key(session_id), json.dumps(turn_data))
            redis_client.expire(_turns_key(session_id), Config.SESSION_TTL_SECONDS)
        except Exception as e:
            log.warning("redis_append_turn_failed", error=str(e), session_id=session_id)

        log.info("turn_added", session_id=session_id, turn_id=turn_id, role=request.role)

        return pb2.AddTurnResponse(turn_id=turn_id)

    def GetConversationHistory(self, request, context):
        session_id = request.session_id
        limit = request.limit if request.limit > 0 else 50

        # Try Redis first (non-critical — fall through to PG on any failure)
        try:
            cached_turns = redis_client.lrange(_turns_key(session_id), 0, -1)
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
                            tool_calls=t.get("tool_calls", ""),
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
                    tool_calls=row["tool_calls"] if isinstance(row["tool_calls"], str) else (json.dumps(row["tool_calls"]) if row["tool_calls"] else ""),
                    created_at=row["created_at"].isoformat(),
                )
                for row in rows
            ]
        )

    # ══════════════════════════════════════════════════════════════
    # Episodic Memory (TimescaleDB — immutable append-only)
    # ══════════════════════════════════════════════════════════════

    def StoreEpisodicMemory(self, request, context):
        memory_id = str(uuid.uuid4())

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
                        request.metadata or "{}",
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

        return pb2.GetEpisodicResponse(
            memories=[
                pb2.EpisodicMemory(
                    memory_id=str(row["id"]),
                    event_type=row["event_type"],
                    summary=row["summary"],
                    key_topics=row["key_topics"] or [],
                    resolution_status=row["resolution_status"],
                    metadata=json.dumps(row["metadata"]) if isinstance(row["metadata"], dict) else (row["metadata"] or "{}"),
                    created_at=row["created_at"].isoformat(),
                )
                for row in rows
            ]
        )

    # ══════════════════════════════════════════════════════════════
    # Session Audit Trail (TimescaleDB — immutable append-only)
    # ══════════════════════════════════════════════════════════════

    def AppendAuditEvent(self, request, context):
        event_id = self._append_audit_internal(
            request.session_id,
            request.customer_id,
            request.event_type,
            request.event_data,
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

        return pb2.GetAuditTrailResponse(
            events=[
                pb2.AuditEvent(
                    event_id=str(row["id"]),
                    session_id=str(row["session_id"]),
                    customer_id=row["customer_id"],
                    event_type=row["event_type"],
                    event_data=json.dumps(row["event_data"]) if isinstance(row["event_data"], dict) else (row["event_data"] or "{}"),
                    event_time=row["event_time"].isoformat(),
                )
                for row in rows
            ]
        )

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
