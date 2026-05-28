"""Tool Service — Tool registry, validation, and execution engine."""

import json
import time
import sys
import uuid
import grpc
from concurrent import futures

import psycopg2
import psycopg2.extras

sys.path.append("..")

import protos.tool_service_pb2 as pb2
import protos.tool_service_pb2_grpc as pb2_grpc
import protos.knowledge_service_pb2 as knowledge_pb2
import protos.knowledge_service_pb2_grpc as knowledge_pb2_grpc

from shared.config import Config
from shared.logging_config import setup_logging
from shared.resilience import create_grpc_channel, grpc_retry

log = setup_logging("tool_service")


# ── Database Connection ───────────────────────────────────────────

def get_pg_conn():
    return psycopg2.connect(Config.DATABASE_URL)


# ── Knowledge Service Stub ────────────────────────────────────────

def get_knowledge_stub():
    channel = create_grpc_channel(Config.KNOWLEDGE_SERVICE_ADDR)
    return knowledge_pb2_grpc.KnowledgeServiceStub(channel)


# ── Tool Implementations ─────────────────────────────────────────

def tool_product_search(params: dict) -> dict:
    """Search products by semantic similarity via Knowledge Service."""
    query = params.get("query", "")
    top_k = params.get("top_k", 5)

    stub = get_knowledge_stub()
    response = stub.RetrieveRelevantDocs(
        knowledge_pb2.KnowledgeRequest(query=query, top_k=top_k)
    )

    products = []
    for p in response.products:
        products.append({
            "product_name": p.product_name,
            "description": p.description,
            "price": p.price,
            "warranty_months": p.warranty_months,
            "similarity_score": round(p.similarity_score, 3),
        })

    # Fallback to text documents if structured products aren't available
    if not products and response.documents:
        return {"results": list(response.documents), "count": len(response.documents)}

    return {"results": products, "count": len(products)}


def tool_price_lookup(params: dict) -> dict:
    """Look up prices by product name or price range."""
    product_name = params.get("product_name", "")
    min_price = params.get("min_price")
    max_price = params.get("max_price")

    conn = get_pg_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if product_name:
                cur.execute(
                    "SELECT product_name, price, warranty_months "
                    "FROM products WHERE LOWER(product_name) LIKE LOWER(%s) "
                    "ORDER BY price ASC LIMIT 10",
                    (f"%{product_name}%",),
                )
            elif min_price is not None or max_price is not None:
                conditions = []
                values = []
                if min_price is not None:
                    conditions.append("price >= %s")
                    values.append(min_price)
                if max_price is not None:
                    conditions.append("price <= %s")
                    values.append(max_price)
                where = " AND ".join(conditions)
                cur.execute(
                    f"SELECT product_name, price, warranty_months "
                    f"FROM products WHERE {where} ORDER BY price ASC LIMIT 10",
                    tuple(values),
                )
            else:
                return {"error": "Provide product_name or price range (min_price/max_price)"}

            rows = cur.fetchall()
    finally:
        conn.close()

    results = [
        {
            "product_name": row["product_name"],
            "price": float(row["price"]),
            "warranty_months": row["warranty_months"],
        }
        for row in rows
    ]
    return {"results": results, "count": len(results)}


def tool_warranty_check(params: dict) -> dict:
    """Check warranty info for a product."""
    product_name = params.get("product_name", "")

    conn = get_pg_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT product_name, warranty_months, manufacturing_date, price "
                "FROM products WHERE LOWER(product_name) LIKE LOWER(%s) LIMIT 5",
                (f"%{product_name}%",),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    results = [
        {
            "product_name": row["product_name"],
            "warranty_months": row["warranty_months"],
            "manufacturing_date": row["manufacturing_date"].isoformat() if row["manufacturing_date"] else None,
            "price": float(row["price"]),
        }
        for row in rows
    ]
    return {"results": results, "count": len(results)}


def tool_product_compare(params: dict) -> dict:
    """Compare products side by side."""
    product_names = params.get("product_names", [])
    if not product_names:
        return {"error": "Provide a list of product_names to compare"}

    conn = get_pg_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Use ILIKE for flexible matching
            conditions = " OR ".join(["LOWER(product_name) LIKE LOWER(%s)"] * len(product_names))
            values = [f"%{name}%" for name in product_names]
            cur.execute(
                f"SELECT product_name, description, price, warranty_months, manufacturing_date "
                f"FROM products WHERE {conditions} LIMIT 10",
                tuple(values),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    results = [
        {
            "product_name": row["product_name"],
            "description": row["description"],
            "price": float(row["price"]),
            "warranty_months": row["warranty_months"],
            "manufacturing_date": row["manufacturing_date"].isoformat() if row["manufacturing_date"] else None,
        }
        for row in rows
    ]
    return {"comparison": results, "count": len(results)}


# Tool dispatch table
TOOL_HANDLERS = {
    "product_search": tool_product_search,
    "price_lookup": tool_price_lookup,
    "warranty_check": tool_warranty_check,
    "product_compare": tool_product_compare,
}


# ── gRPC Service Implementation ──────────────────────────────────

class ToolServiceServicer(pb2_grpc.ToolServiceServicer):

    def ListTools(self, request, context):
        """Return all active tool definitions from the database."""
        conn = get_pg_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    "SELECT name, description, parameter_schema "
                    "FROM tool_definitions WHERE is_active = TRUE"
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        tools = [
            pb2.ToolDefinition(
                name=row["name"],
                description=row["description"],
                parameter_schema=json.dumps(row["parameter_schema"]) if isinstance(row["parameter_schema"], dict) else row["parameter_schema"],
            )
            for row in rows
        ]

        return pb2.ListToolsResponse(tools=tools)

    def ExecuteTool(self, request, context):
        """Execute a tool by name with given parameters."""
        tool_name = request.tool_name
        session_id = request.session_id

        start_time = time.time()

        log.info("tool_execute_start", tool=tool_name, session_id=session_id)

        # Parse parameters
        try:
            params = json.loads(request.parameters) if request.parameters else {}
        except json.JSONDecodeError:
            return pb2.ExecuteToolResponse(
                success=False,
                error=f"Invalid JSON parameters: {request.parameters}",
            )

        # Find handler
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return pb2.ExecuteToolResponse(
                success=False,
                error=f"Unknown tool: {tool_name}",
            )

        # Execute
        try:
            result = handler(params)
            execution_time = int((time.time() - start_time) * 1000)

            log.info(
                "tool_execute_complete",
                tool=tool_name,
                execution_time_ms=execution_time,
            )

            # Log execution
            self._log_execution(session_id, tool_name, params, result, execution_time, "success")

            return pb2.ExecuteToolResponse(
                success=True,
                result=json.dumps(result),
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            log.error("tool_execute_error", tool=tool_name, error=str(e))
            self._log_execution(session_id, tool_name, params, {"error": str(e)}, execution_time, "error")

            return pb2.ExecuteToolResponse(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )

    def _log_execution(self, session_id, tool_name, input_params, output_result, execution_time_ms, status):
        """Log tool execution to the database."""
        conn = None
        try:
            conn = get_pg_conn()
            with conn.cursor() as cur:
                # Get tool_id
                cur.execute("SELECT id FROM tool_definitions WHERE name = %s", (tool_name,))
                row = cur.fetchone()
                tool_id = str(row[0]) if row else None

                cur.execute(
                    "INSERT INTO tool_execution_logs "
                    "(id, session_id, tool_id, input_params, output_result, execution_time_ms, status) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        str(uuid.uuid4()),
                        session_id if session_id else None,
                        tool_id,
                        json.dumps(input_params),
                        json.dumps(output_result),
                        execution_time_ms,
                        status,
                    ),
                )
            conn.commit()
        except Exception as e:
            log.warning("log_execution_failed", error=str(e))
        finally:
            if conn:
                conn.close()


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
    pb2_grpc.add_ToolServiceServicer_to_server(ToolServiceServicer(), server)
    server.add_secure_port("[::]:50056", credentials)
    server.start()
    log.info("server_started", port=50056, tls=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
