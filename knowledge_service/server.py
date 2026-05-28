"""Knowledge Service — pgvector-based semantic search (replaces FAISS)."""

import os
import sys
import json
import grpc
import numpy as np
from concurrent import futures
from dotenv import load_dotenv
import voyageai

import psycopg2
import psycopg2.extras

sys.path.append("..")
load_dotenv()

import protos.knowledge_service_pb2 as pb2
import protos.knowledge_service_pb2_grpc as pb2_grpc

from shared.config import Config
from shared.logging_config import setup_logging

log = setup_logging("knowledge_service")

# ── Voyage AI Client (Anthropic-recommended embedding provider) ──

vo_client = voyageai.Client(api_key=Config.VOYAGE_API_KEY)


# ── Database Connection ───────────────────────────────────────────

def get_pg_conn():
    return psycopg2.connect(Config.DATABASE_URL)


# ── Embedding ─────────────────────────────────────────────────────

def embed_query(text: str) -> list:
    """Generate embedding for a query string using Voyage AI."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty query")
    result = vo_client.embed([text], model=Config.EMBEDDING_MODEL)
    if not result.embeddings or not result.embeddings[0]:
        raise ValueError(f"Voyage AI returned empty embeddings for query: {text[:100]}")
    return result.embeddings[0]


# ── gRPC Service ──────────────────────────────────────────────────

class KnowledgeServiceServicer(pb2_grpc.KnowledgeServiceServicer):

    def RetrieveRelevantDocs(self, request, context):
        query = request.query
        top_k = request.top_k if request.top_k > 0 else 5

        log.info("retrieve_docs", query=query, top_k=top_k)

        try:
            # Generate query embedding
            query_embedding = embed_query(query)
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

            # Search pgvector
            conn = get_pg_conn()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT
                            p.id,
                            p.product_name,
                            p.description,
                            p.price,
                            p.warranty_months,
                            p.manufacturing_date,
                            1 - (pe.embedding <=> %s::vector) AS similarity
                        FROM product_embeddings pe
                        JOIN products p ON pe.product_id = p.id
                        ORDER BY pe.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (embedding_str, embedding_str, min(top_k, 100)),
                    )
                    rows = cur.fetchall()
            finally:
                conn.close()

            # Build response
            documents = []
            products = []
            for row in rows:
                # Formatted text (backward compatible)
                doc_text = (
                    f"Product Name: {row['product_name']}\n"
                    f"Description: {row['description']}\n"
                    f"Price: {row['price']}\n"
                    f"Warranty (months): {row['warranty_months']}"
                )
                documents.append(doc_text)

                # Structured product
                products.append(pb2.ProductDocument(
                    product_id=str(row["id"]),
                    product_name=row["product_name"] or "",
                    description=row["description"] or "",
                    price=float(row["price"]) if row["price"] is not None else 0.0,
                    warranty_months=row["warranty_months"] or 0,
                    manufacturing_date=row["manufacturing_date"].isoformat() if row["manufacturing_date"] else "",
                    similarity_score=float(row["similarity"]) if row["similarity"] is not None else 0.0,
                ))

            log.info("docs_retrieved", count=len(documents))

            return pb2.KnowledgeResponse(
                documents=documents,
                products=products,
            )

        except Exception as e:
            log.error("retrieve_failed", error=str(e))
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.KnowledgeResponse()


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
    pb2_grpc.add_KnowledgeServiceServicer_to_server(KnowledgeServiceServicer(), server)
    server.add_secure_port("[::]:50052", credentials)
    server.start()
    log.info("server_started", port=50052, tls=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
