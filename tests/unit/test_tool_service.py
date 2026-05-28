"""Tests for tool_service — ListTools, ExecuteTool, handlers, logging."""

import json
import time
from unittest.mock import patch, MagicMock

import grpc
import pytest

from tool_service.server import (
    ToolServiceServicer,
    tool_product_search,
    tool_price_lookup,
    tool_warranty_check,
    tool_product_compare,
    TOOL_HANDLERS,
)


@pytest.fixture
def servicer():
    return ToolServiceServicer()


@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.set_code = MagicMock()
    ctx.set_details = MagicMock()
    return ctx


class TestListTools:
    """Tests for ListTools."""

    @patch("tool_service.server.get_pg_conn")
    def test_active_tools_from_db(self, mock_pg, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchall.return_value = [
            {"name": "product_search", "description": "Search products", "parameter_schema": '{"type": "object"}'},
            {"name": "price_lookup", "description": "Look up prices", "parameter_schema": {"type": "object"}},
        ]

        request = MagicMock()
        response = servicer.ListTools(request, mock_context)
        assert len(response.tools) == 2
        assert response.tools[0].name == "product_search"


class TestExecuteTool:
    """Tests for ExecuteTool."""

    @patch("tool_service.server.get_pg_conn")
    @patch("tool_service.server.get_knowledge_stub")
    def test_product_search_handler(self, mock_knowledge, mock_pg, servicer, mock_context):
        knowledge_stub = MagicMock()
        mock_knowledge.return_value = knowledge_stub
        knowledge_stub.RetrieveRelevantDocs.return_value = MagicMock(
            products=[
                MagicMock(product_name="Widget", description="A widget", price=50.0, warranty_months=12, similarity_score=0.9),
            ],
            documents=[],
        )

        # Mock PG for _log_execution
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        cursor.fetchone.return_value = ("tool-id-1",)

        request = MagicMock()
        request.tool_name = "product_search"
        request.session_id = "sess-1"
        request.parameters = json.dumps({"query": "widget", "top_k": 3})

        response = servicer.ExecuteTool(request, mock_context)
        assert response.success is True
        result = json.loads(response.result)
        assert result["count"] == 1

    @patch("tool_service.server.get_pg_conn")
    def test_price_lookup_handler(self, mock_pg, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchall.return_value = [
            {"product_name": "Widget", "price": 49.99, "warranty_months": 12},
        ]
        cursor.fetchone.return_value = ("tool-id-2",)

        request = MagicMock()
        request.tool_name = "price_lookup"
        request.session_id = "sess-1"
        request.parameters = json.dumps({"product_name": "Widget"})

        response = servicer.ExecuteTool(request, mock_context)
        assert response.success is True
        result = json.loads(response.result)
        assert result["results"][0]["price"] == 49.99

    @patch("tool_service.server.get_pg_conn")
    def test_warranty_check_handler(self, mock_pg, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        from datetime import date
        cursor.fetchall.return_value = [
            {"product_name": "Widget", "warranty_months": 24, "manufacturing_date": date(2024, 1, 1), "price": 299.99},
        ]
        cursor.fetchone.return_value = ("tool-id-3",)

        request = MagicMock()
        request.tool_name = "warranty_check"
        request.session_id = "sess-1"
        request.parameters = json.dumps({"product_name": "Widget"})

        response = servicer.ExecuteTool(request, mock_context)
        assert response.success is True
        result = json.loads(response.result)
        assert result["results"][0]["warranty_months"] == 24

    @patch("tool_service.server.get_pg_conn")
    def test_product_compare_handler(self, mock_pg, servicer, mock_context):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        from datetime import date
        cursor.fetchall.return_value = [
            {"product_name": "Widget A", "description": "Desc A", "price": 50.0, "warranty_months": 12, "manufacturing_date": date(2024, 1, 1)},
            {"product_name": "Widget B", "description": "Desc B", "price": 100.0, "warranty_months": 24, "manufacturing_date": date(2024, 6, 1)},
        ]
        cursor.fetchone.return_value = ("tool-id-4",)

        request = MagicMock()
        request.tool_name = "product_compare"
        request.session_id = "sess-1"
        request.parameters = json.dumps({"product_names": ["Widget A", "Widget B"]})

        response = servicer.ExecuteTool(request, mock_context)
        assert response.success is True
        result = json.loads(response.result)
        assert result["count"] == 2

    def test_unknown_tool(self, servicer, mock_context):
        request = MagicMock()
        request.tool_name = "nonexistent_tool"
        request.session_id = "sess-1"
        request.parameters = "{}"

        response = servicer.ExecuteTool(request, mock_context)
        assert response.success is False
        assert "Unknown tool" in response.error

    def test_invalid_json_params(self, servicer, mock_context):
        request = MagicMock()
        request.tool_name = "product_search"
        request.session_id = "sess-1"
        request.parameters = "not json {"

        response = servicer.ExecuteTool(request, mock_context)
        assert response.success is False
        assert "Invalid JSON" in response.error

    @patch("tool_service.server.get_pg_conn")
    @patch("tool_service.server.get_knowledge_stub")
    def test_exception_in_handler(self, mock_knowledge, mock_pg, servicer, mock_context):
        mock_knowledge.side_effect = Exception("Connection refused")

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        cursor.fetchone.return_value = None

        request = MagicMock()
        request.tool_name = "product_search"
        request.session_id = "sess-1"
        request.parameters = json.dumps({"query": "test"})

        response = servicer.ExecuteTool(request, mock_context)
        assert response.success is False
        assert "Connection refused" in response.error

    @patch("tool_service.server.get_pg_conn")
    @patch("tool_service.server.get_knowledge_stub")
    def test_execution_logging(self, mock_knowledge, mock_pg, servicer, mock_context):
        knowledge_stub = MagicMock()
        mock_knowledge.return_value = knowledge_stub
        knowledge_stub.RetrieveRelevantDocs.return_value = MagicMock(
            products=[], documents=["Some doc"],
        )

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        cursor.fetchone.return_value = ("tool-id-1",)

        request = MagicMock()
        request.tool_name = "product_search"
        request.session_id = "sess-1"
        request.parameters = json.dumps({"query": "test"})

        servicer.ExecuteTool(request, mock_context)
        # _log_execution should insert into tool_execution_logs
        assert cursor.execute.call_count >= 2  # SELECT tool_id + INSERT log


class TestPriceLookupRangeQuery:
    """Tests for price_lookup with min/max price ranges."""

    @patch("tool_service.server.get_pg_conn")
    def test_price_range_filter(self, mock_pg):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchall.return_value = [
            {"product_name": "Widget", "price": 75.0, "warranty_months": 12},
        ]

        result = tool_price_lookup({"min_price": 50, "max_price": 100})
        assert result["count"] == 1
        assert result["results"][0]["price"] == 75.0

    @patch("tool_service.server.get_pg_conn")
    def test_no_params_returns_error(self, mock_pg):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        result = tool_price_lookup({})
        assert "error" in result

    @patch("tool_service.server.get_pg_conn")
    def test_min_price_only(self, mock_pg):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        cursor.fetchall.return_value = []

        result = tool_price_lookup({"min_price": 500})
        assert result["count"] == 0


class TestProductCompareEdgeCases:
    """Tests for product_compare edge cases."""

    def test_empty_product_names(self):
        result = tool_product_compare({"product_names": []})
        assert "error" in result

    def test_missing_product_names(self):
        result = tool_product_compare({})
        assert "error" in result


class TestProductSearchFallback:
    """Test product_search fallback to text documents."""

    @patch("tool_service.server.get_knowledge_stub")
    def test_fallback_to_documents(self, mock_knowledge):
        stub = MagicMock()
        mock_knowledge.return_value = stub
        stub.RetrieveRelevantDocs.return_value = MagicMock(
            products=[],
            documents=["Product Name: Widget\nDescription: Great\nPrice: 50\nWarranty: 12"],
        )

        result = tool_product_search({"query": "widget", "top_k": 5})
        assert result["count"] == 1
        assert "Widget" in result["results"][0]
