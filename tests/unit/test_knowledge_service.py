"""Tests for knowledge_service — embedding, pgvector search."""

import json
from unittest.mock import patch, MagicMock
from datetime import date

import grpc
import pytest

from knowledge_service.server import KnowledgeServiceServicer, embed_query


@pytest.fixture
def servicer():
    return KnowledgeServiceServicer()


@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.set_code = MagicMock()
    ctx.set_details = MagicMock()
    return ctx


class TestRetrieveRelevantDocs:
    """Tests for RetrieveRelevantDocs."""

    @patch("knowledge_service.server.get_pg_conn")
    @patch("knowledge_service.server.embed_query")
    def test_products_returned(self, mock_embed, mock_pg, servicer, mock_context):
        mock_embed.return_value = [0.1] * 1024

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchall.return_value = [
            {
                "id": "p1",
                "product_name": "UltraWidget Pro",
                "description": "A premium widget",
                "price": 299.99,
                "warranty_months": 24,
                "manufacturing_date": date(2024, 1, 15),
                "similarity": 0.92,
            }
        ]

        request = MagicMock()
        request.query = "premium widget"
        request.top_k = 5

        response = servicer.RetrieveRelevantDocs(request, mock_context)
        assert len(response.products) == 1
        assert response.products[0].product_name == "UltraWidget Pro"
        assert response.products[0].similarity_score == pytest.approx(0.92)
        assert len(response.documents) == 1
        assert "UltraWidget Pro" in response.documents[0]

    @patch("knowledge_service.server.get_pg_conn")
    @patch("knowledge_service.server.embed_query")
    def test_text_formatted(self, mock_embed, mock_pg, servicer, mock_context):
        mock_embed.return_value = [0.5] * 1024

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchall.return_value = [
            {
                "id": "p2",
                "product_name": "BasicWidget",
                "description": "Entry-level widget",
                "price": 49.99,
                "warranty_months": 12,
                "manufacturing_date": date(2024, 3, 1),
                "similarity": 0.85,
            }
        ]

        request = MagicMock()
        request.query = "cheap widget"
        request.top_k = 0  # Should default to 5

        response = servicer.RetrieveRelevantDocs(request, mock_context)
        doc = response.documents[0]
        assert "Product Name: BasicWidget" in doc
        assert "Price: 49.99" in doc
        assert "Warranty (months): 12" in doc

    @patch("knowledge_service.server.get_pg_conn")
    @patch("knowledge_service.server.embed_query")
    def test_default_top_k(self, mock_embed, mock_pg, servicer, mock_context):
        mock_embed.return_value = [0.1] * 1024

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        cursor.fetchall.return_value = []

        request = MagicMock()
        request.query = "test"
        request.top_k = 0

        servicer.RetrieveRelevantDocs(request, mock_context)
        # Verify the SQL was called with top_k=5 (default)
        execute_args = cursor.execute.call_args[0]
        assert 5 in execute_args[1]

    @patch("knowledge_service.server.get_pg_conn")
    @patch("knowledge_service.server.embed_query")
    def test_null_fields_handled(self, mock_embed, mock_pg, servicer, mock_context):
        mock_embed.return_value = [0.1] * 1024

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchall.return_value = [
            {
                "id": "p3",
                "product_name": "Mystery Widget",
                "description": None,
                "price": 0.0,
                "warranty_months": None,
                "manufacturing_date": None,
                "similarity": None,
            }
        ]

        request = MagicMock()
        request.query = "mystery"
        request.top_k = 1

        response = servicer.RetrieveRelevantDocs(request, mock_context)
        assert response.products[0].description == ""
        assert response.products[0].warranty_months == 0
        assert response.products[0].manufacturing_date == ""
        assert response.products[0].similarity_score == 0.0


class TestEmbedQuery:
    """Tests for embed_query."""

    @patch("knowledge_service.server.vo_client")
    def test_voyage_ai_called(self, mock_vo):
        mock_vo.embed.return_value = MagicMock(embeddings=[[0.1, 0.2, 0.3]])
        result = embed_query("test query")
        assert result == [0.1, 0.2, 0.3]
        mock_vo.embed.assert_called_once()


class TestErrorHandling:
    """Tests for error handling in RetrieveRelevantDocs."""

    @patch("knowledge_service.server.embed_query")
    def test_embedding_failure(self, mock_embed, servicer, mock_context):
        mock_embed.side_effect = Exception("Voyage API down")

        request = MagicMock()
        request.query = "test"
        request.top_k = 5

        response = servicer.RetrieveRelevantDocs(request, mock_context)
        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
        assert len(response.documents) == 0
