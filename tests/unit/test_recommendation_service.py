"""Tests for recommendation_service — context-aware recommendations with cold start."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest

from recommendation_service.server import (
    RecommendationServiceServicer,
    _get_product_catalog_summary,
    _build_session_context,
    _build_customer_profile,
    _gap_analysis,
    _fill_templates,
    _select_template_set,
    _gap_fallbacks,
    _get_cross_user_popular_queries,
    _catalog_cache,
    _catalog_lock,
    CATALOG_TTL,
    FALLBACK_BRANDS,
    TEMPLATES,
    ALL_INTENTS,
    INTENT_PRIORITY,
)


@pytest.fixture
def servicer():
    return RecommendationServiceServicer()


@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.set_code = MagicMock()
    ctx.set_details = MagicMock()
    return ctx


@pytest.fixture(autouse=True)
def reset_catalog_cache():
    """Reset catalog cache before each test."""
    with _catalog_lock:
        _catalog_cache["data"] = None
        _catalog_cache["timestamp"] = 0
    yield
    with _catalog_lock:
        _catalog_cache["data"] = None
        _catalog_cache["timestamp"] = 0


def _make_catalog():
    """Helper: build a realistic catalog summary for tests."""
    return {
        "brands": {
            "UltraWasher": {"count": 5, "price_min": 121.0, "price_max": 333.0, "warranties": [6, 12, 24, 36]},
            "RoboCleaner": {"count": 7, "price_min": 129.0, "price_max": 499.0, "warranties": [6, 12, 18, 24]},
            "PowerDrill": {"count": 10, "price_min": 54.0, "price_max": 486.0, "warranties": [6, 12, 24, 36]},
            "MegaBlender": {"count": 8, "price_min": 61.0, "price_max": 336.0, "warranties": [6, 12, 18, 24, 36]},
            "EcoKettle": {"count": 5, "price_min": 87.0, "price_max": 448.0, "warranties": [6, 12]},
        },
        "all_product_names": [
            "UltraWasher 8262", "UltraWasher 5155", "UltraWasher 5944",
            "RoboCleaner 3120", "RoboCleaner 8285", "RoboCleaner 4653",
            "PowerDrill 5641", "PowerDrill 8255", "PowerDrill 9154",
            "MegaBlender 5588", "MegaBlender 8913", "MegaBlender 9904",
            "EcoKettle 1042", "EcoKettle 3468", "EcoKettle 8031",
        ],
        "price_min": 54.0,
        "price_max": 499.0,
    }


def _make_session_context(**overrides):
    """Helper: build a session context dict for tests."""
    ctx = {
        "intents_used": set(),
        "products_mentioned": set(),
        "brands_mentioned": set(),
        "tools_used": set(),
        "last_intent": None,
        "last_user_query": None,
        "last_assistant_response": None,
    }
    ctx.update(overrides)
    return ctx


def _make_customer_profile(**overrides):
    """Helper: build a customer profile dict for tests."""
    profile = {
        "brands_explored": set(),
        "intents_history": set(),
        "topics_explored": [],
        "tools_used_historically": set(),
        "has_history": False,
    }
    profile.update(overrides)
    return profile


# ══════════════════════════════════════════════════════════════════
# Cold Start Tiers
# ══════════════════════════════════════════════════════════════════

class TestColdStartTier1:
    """Tier 1 — Returning user with episodic memories."""

    @patch("recommendation_service.server._get_cross_user_popular_queries")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_returning_user_personalized(self, mock_catalog, mock_profile, mock_popular, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_profile.return_value = _make_customer_profile(
            has_history=True,
            topics_explored=["UltraWasher 8262", "price comparison"],
            brands_explored={"UltraWasher"},
            intents_history={"product_inquiry"},
        )
        mock_popular.return_value = []

        request = MagicMock()
        request.customer_id = "cust-1"
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) >= 3
        # Should mention last topic
        assert any("UltraWasher 8262" in s for s in suggestions)
        # Should suggest an unexplored brand
        explored = {"UltraWasher"}
        assert any(
            brand in s
            for s in suggestions
            for brand in ["RoboCleaner", "PowerDrill", "MegaBlender", "EcoKettle"]
        )

    @patch("recommendation_service.server._get_cross_user_popular_queries")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_returning_user_missing_intent_suggestion(self, mock_catalog, mock_profile, mock_popular, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_profile.return_value = _make_customer_profile(
            has_history=True,
            topics_explored=["PowerDrill"],
            brands_explored={"PowerDrill"},
            intents_history={"product_inquiry", "price_check"},
        )
        mock_popular.return_value = []

        request = MagicMock()
        request.customer_id = "cust-2"
        request.session_id = "sess-2"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # comparison and warranty_question are missing — should suggest one
        assert any(
            "compare" in s.lower() or "warranty" in s.lower()
            for s in suggestions
        )


class TestColdStartTier2:
    """Tier 2 — New user, system has cross-user data."""

    @patch("recommendation_service.server._get_cross_user_popular_queries")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_cross_user_popular(self, mock_catalog, mock_profile, mock_popular, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_profile.return_value = _make_customer_profile(has_history=False)
        mock_popular.return_value = [
            "Show me PowerDrill products",
            "What's the cheapest RoboCleaner?",
            "Compare UltraWasher with MegaBlender",
        ]

        request = MagicMock()
        request.customer_id = "new-cust"
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) >= 3
        assert any("PowerDrill" in s for s in suggestions)


class TestColdStartTier3:
    """Tier 3 — Empty system, catalog-aware defaults."""

    @patch("recommendation_service.server._get_cross_user_popular_queries")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_catalog_aware_defaults(self, mock_catalog, mock_profile, mock_popular, servicer, mock_context):
        catalog = _make_catalog()
        mock_catalog.return_value = catalog
        mock_profile.return_value = _make_customer_profile(has_history=False)
        mock_popular.return_value = []

        request = MagicMock()
        request.customer_id = ""
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) >= 3
        assert len(suggestions) <= 5
        # Should mention real brand names from catalog
        all_brands = set(catalog["brands"].keys())
        assert any(
            brand in s for s in suggestions for brand in all_brands
        )
        # Should NOT contain generic "What products do you have?"
        assert not any(s == "What products do you have?" for s in suggestions)


class TestColdStartFallbackCascade:
    """Tier cascading: if tier 1 fails, fall to tier 2/3."""

    @patch("recommendation_service.server._get_cross_user_popular_queries")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_tier1_failure_cascades(self, mock_catalog, mock_profile, mock_popular, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_profile.side_effect = Exception("Memory service unavailable")
        mock_popular.return_value = ["Compare PowerDrill with RoboCleaner"]

        request = MagicMock()
        request.customer_id = "cust-1"
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # Should still return suggestions (from tier 2 or 3)
        assert len(suggestions) >= 3

    @patch("recommendation_service.server._get_cross_user_popular_queries")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_all_tiers_fail_still_returns(self, mock_catalog, mock_profile, mock_popular, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_profile.side_effect = Exception("Memory down")
        mock_popular.side_effect = Exception("DB down")

        request = MagicMock()
        request.customer_id = "cust-1"
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # Tier 3 catalog-aware defaults should still work
        assert len(suggestions) >= 3

    @patch("recommendation_service.server._get_cross_user_popular_queries")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_max_5_suggestions(self, mock_catalog, mock_profile, mock_popular, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_profile.return_value = _make_customer_profile(
            has_history=True,
            topics_explored=["UltraWasher", "PowerDrill"],
            brands_explored={"UltraWasher"},
        )
        mock_popular.return_value = [
            f"Query {i}" for i in range(10)
        ]

        request = MagicMock()
        request.customer_id = "cust-1"
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        assert len(list(response.suggestions)) <= 5


# ══════════════════════════════════════════════════════════════════
# Follow-Up Quality
# ══════════════════════════════════════════════════════════════════

class TestFollowUpQuality:
    """Follow-up suggestions are product-focused, not meta-questions."""

    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_product_focused_not_meta(self, mock_catalog, mock_session, mock_profile, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"product_inquiry"},
            products_mentioned={"UltraWasher 8262"},
            brands_mentioned={"UltraWasher"},
            last_intent="product_inquiry",
            last_user_query="Tell me about UltraWasher 8262",
        )
        mock_profile.return_value = _make_customer_profile()

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "Tell me about UltraWasher 8262"
        request.last_response = "UltraWasher 8262 is a high-performance washer..."
        request.intent = "product_inquiry"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) == 3
        # No meta-questions
        meta_keywords = ["conversation history", "delete", "export", "chat log", "previous questions"]
        for s in suggestions:
            for meta in meta_keywords:
                assert meta not in s.lower(), f"Meta-question detected: {s}"

    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_no_echo_of_last_query(self, mock_catalog, mock_session, mock_profile, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"product_inquiry"},
            products_mentioned={"PowerDrill 5641"},
            brands_mentioned={"PowerDrill"},
            last_intent="product_inquiry",
            last_user_query="How much does PowerDrill 5641 cost?",
        )
        mock_profile.return_value = _make_customer_profile()

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "How much does PowerDrill 5641 cost?"
        request.last_response = "PowerDrill 5641 costs $380.88"
        request.intent = "price_check"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # Should not echo the exact last query
        assert "How much does PowerDrill 5641 cost?" not in suggestions

    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_deduplication(self, mock_catalog, mock_session, mock_profile, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"product_inquiry"},
            products_mentioned={"MegaBlender 5588"},
            brands_mentioned={"MegaBlender"},
            last_intent="product_inquiry",
        )
        mock_profile.return_value = _make_customer_profile()

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "Tell me about MegaBlender"
        request.last_response = "MegaBlender info..."
        request.intent = "product_inquiry"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # No duplicates
        assert len(suggestions) == len(set(s.lower() for s in suggestions))

    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_exactly_3_suggestions(self, mock_catalog, mock_session, mock_profile, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"price_check"},
            products_mentioned=set(),
            brands_mentioned=set(),
            last_intent="price_check",
        )
        mock_profile.return_value = _make_customer_profile()

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "What's the cheapest product?"
        request.last_response = "The cheapest is PowerDrill 7464 at $53.81"
        request.intent = "price_check"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        assert len(list(response.suggestions)) == 3


# ══════════════════════════════════════════════════════════════════
# Gap Analysis
# ══════════════════════════════════════════════════════════════════

class TestGapAnalysis:
    """Gap analysis correctly identifies missing intents, brands, tools."""

    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_missing_intents_priority_order(self, mock_catalog):
        mock_catalog.return_value = _make_catalog()
        session_ctx = _make_session_context(intents_used={"product_inquiry"})
        profile = _make_customer_profile()

        gaps = _gap_analysis(session_ctx, profile)

        # comparison is highest priority, should be first
        assert gaps["missing_intents"][0] == "comparison"
        assert "product_inquiry" not in gaps["missing_intents"]

    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_brand_exclusion(self, mock_catalog):
        mock_catalog.return_value = _make_catalog()
        session_ctx = _make_session_context(brands_mentioned={"PowerDrill", "UltraWasher"})
        profile = _make_customer_profile(brands_explored={"MegaBlender"})

        gaps = _gap_analysis(session_ctx, profile)

        assert "PowerDrill" not in gaps["unexplored_brands"]
        assert "UltraWasher" not in gaps["unexplored_brands"]
        assert "MegaBlender" not in gaps["unexplored_brands"]
        # RoboCleaner and EcoKettle should be there
        assert "RoboCleaner" in gaps["unexplored_brands"]
        assert "EcoKettle" in gaps["unexplored_brands"]

    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_unexplored_brands_sorted_by_count(self, mock_catalog):
        mock_catalog.return_value = _make_catalog()
        session_ctx = _make_session_context()
        profile = _make_customer_profile()

        gaps = _gap_analysis(session_ctx, profile)

        # PowerDrill has 10 products, should be first
        assert gaps["unexplored_brands"][0] == "PowerDrill"

    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_tool_gaps(self, mock_catalog):
        mock_catalog.return_value = _make_catalog()
        session_ctx = _make_session_context(tools_used={"product_search"})
        profile = _make_customer_profile()

        gaps = _gap_analysis(session_ctx, profile)

        assert "product_search" not in gaps["missing_tools"]
        assert "price_lookup" in gaps["missing_tools"]

    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_suggested_price_range(self, mock_catalog):
        mock_catalog.return_value = _make_catalog()
        session_ctx = _make_session_context()
        profile = _make_customer_profile()

        gaps = _gap_analysis(session_ctx, profile)

        # Midpoint of 54..499
        assert gaps["suggested_price_range"] == int((54.0 + 499.0) / 2)


# ══════════════════════════════════════════════════════════════════
# Template Filling
# ══════════════════════════════════════════════════════════════════

class TestTemplateFilling:
    """Templates are filled with real product data."""

    def test_product_name_from_context(self):
        catalog = _make_catalog()
        context = _make_session_context(
            products_mentioned={"UltraWasher 8262"},
            brands_mentioned={"UltraWasher"},
        )
        gaps = {"unexplored_brands": ["RoboCleaner"], "suggested_price_range": 200}

        templates = [
            "What's the warranty on {product_name}?",
            "Compare {product_name} with {new_brand} products",
        ]
        filled = _fill_templates(templates, context, catalog, gaps)

        assert "UltraWasher 8262" in filled[0]
        assert "RoboCleaner" in filled[1]

    def test_brand_extraction(self):
        catalog = _make_catalog()
        context = _make_session_context(
            brands_mentioned={"PowerDrill"},
        )
        gaps = {"unexplored_brands": ["EcoKettle"], "suggested_price_range": 250}

        templates = ["Compare {brand} with {new_brand}"]
        filled = _fill_templates(templates, context, catalog, gaps)

        assert "PowerDrill" in filled[0]
        assert "EcoKettle" in filled[0]

    def test_new_brand_from_gaps(self):
        catalog = _make_catalog()
        context = _make_session_context()
        gaps = {"unexplored_brands": ["MegaBlender", "RoboCleaner"], "suggested_price_range": 200}

        templates = ["Show me {new_brand} products"]
        filled = _fill_templates(templates, context, catalog, gaps)

        assert "MegaBlender" in filled[0]

    def test_price_threshold_filled(self):
        catalog = _make_catalog()
        context = _make_session_context()
        gaps = {"unexplored_brands": ["PowerDrill"], "suggested_price_range": 276}

        templates = ["Show me products under ${price_threshold}"]
        filled = _fill_templates(templates, context, catalog, gaps)

        assert "$276" in filled[0]


class TestTemplateSelection:
    """Template set selection based on intent and context."""

    def test_same_product_when_products_mentioned(self):
        context = _make_session_context(products_mentioned={"UltraWasher 8262"})
        gaps = {"unexplored_brands": ["RoboCleaner"]}

        key = _select_template_set("product_inquiry", context, gaps)
        assert key == ("product_inquiry", "same_product")

    def test_new_brand_when_no_products(self):
        context = _make_session_context()
        gaps = {"unexplored_brands": ["RoboCleaner"]}

        key = _select_template_set("product_inquiry", context, gaps)
        assert key == ("product_inquiry", "new_brand")

    def test_unknown_intent_falls_to_general(self):
        context = _make_session_context()
        gaps = {"unexplored_brands": ["PowerDrill"]}

        key = _select_template_set("unknown_intent_xyz", context, gaps)
        assert key[0] == "general_question"


# ══════════════════════════════════════════════════════════════════
# Catalog Cache
# ══════════════════════════════════════════════════════════════════

class TestCatalogCache:
    """Catalog cache with TTL and DB failure fallback."""

    @patch("recommendation_service.server.get_pg_conn")
    def test_ttl_cache_reuse(self, mock_pg):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        cursor.fetchall.return_value = [
            {"product_name": "TestBrand 1000", "price": 100.0, "warranty_months": 12},
        ]

        # First call — hits DB
        result1 = _get_product_catalog_summary()
        assert "TestBrand" in result1["brands"]
        assert mock_pg.call_count == 1

        # Second call — should use cache
        result2 = _get_product_catalog_summary()
        assert result2 is result1
        assert mock_pg.call_count == 1  # No additional DB call

    @patch("recommendation_service.server.get_pg_conn")
    def test_db_failure_returns_fallback(self, mock_pg):
        mock_pg.side_effect = Exception("DB connection failed")

        result = _get_product_catalog_summary()

        # Should return fallback brands
        assert "UltraWasher" in result["brands"]
        assert "PowerDrill" in result["brands"]
        assert len(result["brands"]) == len(FALLBACK_BRANDS)


# ══════════════════════════════════════════════════════════════════
# Session Context
# ══════════════════════════════════════════════════════════════════

class TestSessionContext:
    """Session context builder extracts intents, products, brands, tools."""

    @patch("recommendation_service.server.get_pg_conn")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_extracts_intents_and_products(self, mock_catalog, mock_pg):
        mock_catalog.return_value = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        cursor.fetchall.return_value = [
            {"role": "user", "content": "Tell me about UltraWasher 8262", "intent": "product_inquiry", "tool_calls": None},
            {"role": "assistant", "content": "UltraWasher 8262 is a great washer...", "intent": None, "tool_calls": [{"tool": "product_search"}]},
        ]

        ctx = _build_session_context("sess-1")

        assert "product_inquiry" in ctx["intents_used"]
        assert "UltraWasher 8262" in ctx["products_mentioned"]
        assert "UltraWasher" in ctx["brands_mentioned"]
        assert "product_search" in ctx["tools_used"]
        assert ctx["last_user_query"] == "Tell me about UltraWasher 8262"

    def test_empty_session_id(self):
        ctx = _build_session_context("")
        assert ctx["intents_used"] == set()
        assert ctx["products_mentioned"] == set()


# ══════════════════════════════════════════════════════════════════
# Customer Profile
# ══════════════════════════════════════════════════════════════════

class TestCustomerProfile:
    """Customer profile builder extracts brands, topics, intents from episodic memories."""

    @patch("recommendation_service.server._get_product_catalog_summary")
    @patch("recommendation_service.server.get_memory_stub")
    def test_profile_from_memories(self, mock_mem, mock_catalog):
        mock_catalog.return_value = _make_catalog()
        mem_stub = MagicMock()
        mock_mem.return_value = mem_stub

        memory1 = MagicMock()
        memory1.key_topics = ["UltraWasher 8262", "price comparison"]
        memory1.metadata = json.dumps({"intents": ["product_inquiry", "price_check"], "tools_used": ["product_search"]})

        memory2 = MagicMock()
        memory2.key_topics = ["PowerDrill 5641"]
        memory2.metadata = "{}"

        mem_stub.GetEpisodicMemories.return_value = MagicMock(memories=[memory1, memory2])

        profile = _build_customer_profile("cust-1")

        assert profile["has_history"] is True
        assert "UltraWasher" in profile["brands_explored"]
        assert "PowerDrill" in profile["brands_explored"]
        assert "product_inquiry" in profile["intents_history"]
        assert "product_search" in profile["tools_used_historically"]
        assert "UltraWasher 8262" in profile["topics_explored"]

    @patch("recommendation_service.server.get_memory_stub")
    def test_no_memories_returns_empty(self, mock_mem):
        mem_stub = MagicMock()
        mock_mem.return_value = mem_stub
        mem_stub.GetEpisodicMemories.return_value = MagicMock(memories=[])

        profile = _build_customer_profile("cust-new")

        assert profile["has_history"] is False
        assert len(profile["brands_explored"]) == 0

    @patch("recommendation_service.server.get_memory_stub")
    def test_memory_service_failure(self, mock_mem):
        mock_mem.side_effect = Exception("Memory service down")

        profile = _build_customer_profile("cust-1")

        assert profile["has_history"] is False

    def test_empty_customer_id(self):
        profile = _build_customer_profile("")
        assert profile["has_history"] is False


# ══════════════════════════════════════════════════════════════════
# Graceful Degradation
# ══════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    """Every external call failure still returns suggestions."""

    @patch("recommendation_service.server.get_pg_conn")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_all_services_down_followup(self, mock_catalog, mock_session, mock_profile, mock_pg, servicer, mock_context):
        # Catalog still works (fallback brands), but everything else fails
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context()
        mock_profile.return_value = _make_customer_profile()
        mock_pg.side_effect = Exception("DB down")

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "something"
        request.last_response = "response"
        request.intent = "product_inquiry"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) == 3

    @patch("recommendation_service.server._get_cross_user_popular_queries")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_catalog_fallback_still_works(self, mock_catalog, mock_profile, mock_popular, servicer, mock_context):
        # Even with fallback catalog, should produce suggestions
        mock_catalog.return_value = {
            "brands": dict(FALLBACK_BRANDS),
            "all_product_names": [f"{b} 1000" for b in FALLBACK_BRANDS],
            "price_min": 50.0,
            "price_max": 500.0,
        }
        mock_profile.return_value = _make_customer_profile()
        mock_popular.return_value = []

        request = MagicMock()
        request.customer_id = ""
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) >= 3


# ══════════════════════════════════════════════════════════════════
# Gap Progression
# ══════════════════════════════════════════════════════════════════

class TestGapProgression:
    """After using some intents, remaining intents appear as suggestions."""

    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_warranty_and_comparison_suggested_after_inquiry_and_price(
        self, mock_catalog, mock_session, mock_profile, servicer, mock_context
    ):
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"product_inquiry", "price_check"},
            brands_mentioned={"UltraWasher"},
            products_mentioned={"UltraWasher 8262"},
            last_intent="price_check",
            last_user_query="How much is UltraWasher 8262?",
        )
        mock_profile.return_value = _make_customer_profile()

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "How much is UltraWasher 8262?"
        request.last_response = "UltraWasher 8262 costs $121.24"
        request.intent = "price_check"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # warranty or comparison should appear
        all_text = " ".join(s.lower() for s in suggestions)
        assert "warranty" in all_text or "compare" in all_text


# ══════════════════════════════════════════════════════════════════
# Cross-User Popular Queries
# ══════════════════════════════════════════════════════════════════

class TestCrossUserPopularQueries:
    """Cross-user popular query aggregation."""

    @patch("recommendation_service.server.get_pg_conn")
    def test_returns_popular_queries(self, mock_pg):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        cursor.fetchall.return_value = [
            {"content": "Show me PowerDrill products", "intent": "product_inquiry", "freq": 15},
            {"content": "What's the cheapest item?", "intent": "price_check", "freq": 10},
        ]

        result = _get_cross_user_popular_queries(limit=5)

        assert len(result) == 2
        assert "Show me PowerDrill products" in result

    @patch("recommendation_service.server.get_pg_conn")
    def test_db_failure_returns_empty(self, mock_pg):
        mock_pg.side_effect = Exception("DB down")

        result = _get_cross_user_popular_queries()

        assert result == []


# ══════════════════════════════════════════════════════════════════
# Edge Cases
# ══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases: empty catalog, single brand, self-comparison, deterministic ordering."""

    def test_gap_fallbacks_empty_catalog_no_crash(self):
        """_gap_fallbacks must not crash when catalog has no brands."""
        gaps = {"unexplored_brands": [], "missing_intents": ["comparison", "price_check"], "suggested_price_range": 200}
        empty_catalog = {"brands": {}, "all_product_names": [], "price_min": 50.0, "price_max": 500.0}

        # Should not raise IndexError
        result = _gap_fallbacks(gaps, empty_catalog)
        assert len(result) >= 2

    def test_gap_fallbacks_product_inquiry_handled(self):
        """_gap_fallbacks handles product_inquiry intent (not just comparison/warranty/price)."""
        catalog = _make_catalog()
        gaps = {
            "unexplored_brands": ["RoboCleaner"],
            "missing_intents": ["product_inquiry"],
            "suggested_price_range": 200,
        }

        result = _gap_fallbacks(gaps, catalog)
        assert any("RoboCleaner" in s for s in result)

    def test_fill_templates_no_self_comparison(self):
        """When only 1 brand exists, templates with both {brand} and {new_brand} are skipped."""
        catalog = {
            "brands": {"OnlyBrand": {"count": 1, "price_min": 100, "price_max": 200, "warranties": [12]}},
            "all_product_names": ["OnlyBrand 1000"],
            "price_min": 100.0,
            "price_max": 200.0,
        }
        context = _make_session_context()
        gaps = {"unexplored_brands": [], "suggested_price_range": 150}

        templates = ["Compare {brand} with {new_brand}"]
        filled = _fill_templates(templates, context, catalog, gaps)

        # With only 1 brand, the template using both {brand} and {new_brand}
        # is skipped entirely to avoid "Compare X with X".
        assert len(filled) == 0

    def test_fill_templates_single_brand_keeps_non_comparison_templates(self):
        """Single-brand catalog: templates using only {brand} or only {new_brand} are kept."""
        catalog = {
            "brands": {"OnlyBrand": {"count": 1, "price_min": 100, "price_max": 200, "warranties": [12]}},
            "all_product_names": ["OnlyBrand 1000"],
            "price_min": 100.0,
            "price_max": 200.0,
        }
        context = _make_session_context()
        gaps = {"unexplored_brands": [], "suggested_price_range": 150}

        templates = [
            "Show me {brand} products",
            "Compare {brand} with {new_brand}",
            "What does {new_brand} offer?",
        ]
        filled = _fill_templates(templates, context, catalog, gaps)

        # The comparison template is skipped; the other two are kept
        assert len(filled) == 2
        assert "Show me OnlyBrand products" in filled
        assert "What does OnlyBrand offer?" in filled

    def test_fill_templates_two_brands_no_self_comparison(self):
        """With 2 brands, template filling must not produce self-comparison."""
        catalog = {
            "brands": {
                "BrandA": {"count": 2, "price_min": 100, "price_max": 200, "warranties": [12]},
                "BrandB": {"count": 3, "price_min": 50, "price_max": 300, "warranties": [6, 24]},
            },
            "all_product_names": ["BrandA 1000", "BrandA 1001", "BrandB 2000"],
            "price_min": 50.0,
            "price_max": 300.0,
        }
        # User mentioned BrandA, so brand=BrandA. new_brand should be BrandB.
        context = _make_session_context(brands_mentioned={"BrandA"})
        gaps = {"unexplored_brands": ["BrandB"], "suggested_price_range": 175}

        templates = ["Compare {brand} with {new_brand}"]
        filled = _fill_templates(templates, context, catalog, gaps)

        assert "BrandA" in filled[0]
        assert "BrandB" in filled[0]
        assert filled[0] != "Compare BrandA with BrandA"

    def test_fill_templates_deterministic_ordering(self):
        """Multiple calls with same input produce same output."""
        catalog = _make_catalog()
        context = _make_session_context(
            products_mentioned={"PowerDrill 5641", "UltraWasher 8262"},
            brands_mentioned={"PowerDrill", "UltraWasher"},
        )
        gaps = {"unexplored_brands": ["EcoKettle", "MegaBlender"], "suggested_price_range": 200}

        templates = ["Tell me about {product_name}", "Show me {new_brand}"]

        results = set()
        for _ in range(10):
            filled = _fill_templates(templates, context, catalog, gaps)
            results.add(tuple(filled))

        # All 10 calls should produce identical output
        assert len(results) == 1

    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_followup_empty_catalog_no_crash(self, mock_catalog, mock_session, mock_profile, servicer, mock_context):
        """GetFollowUpRecommendations must not crash with empty product catalog."""
        mock_catalog.return_value = {
            "brands": {},
            "all_product_names": [],
            "price_min": 50.0,
            "price_max": 500.0,
        }
        mock_session.return_value = _make_session_context()
        mock_profile.return_value = _make_customer_profile()

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "Tell me about products"
        request.last_response = "We have many products"
        request.intent = "product_inquiry"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # Should return 3 suggestions without crashing
        assert len(suggestions) == 3

    @patch("recommendation_service.server._get_cross_user_popular_queries")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_start_empty_catalog_no_crash(self, mock_catalog, mock_profile, mock_popular, servicer, mock_context):
        """GetStartRecommendations must not crash with empty product catalog."""
        mock_catalog.return_value = {
            "brands": {},
            "all_product_names": [],
            "price_min": 50.0,
            "price_max": 500.0,
        }
        mock_profile.return_value = _make_customer_profile()
        mock_popular.return_value = []

        request = MagicMock()
        request.customer_id = ""
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) >= 2  # At least price + warranty defaults

    @patch("recommendation_service.server._get_cross_user_popular_queries")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_tier1_product_inquiry_as_only_missing_intent(self, mock_catalog, mock_profile, mock_popular, servicer, mock_context):
        """Tier 1: when product_inquiry is the only missing intent, it should be suggested."""
        mock_catalog.return_value = _make_catalog()
        mock_profile.return_value = _make_customer_profile(
            has_history=True,
            topics_explored=["UltraWasher 8262"],
            brands_explored={"UltraWasher"},
            intents_history={"comparison", "warranty_question", "price_check"},
        )
        mock_popular.return_value = []

        request = MagicMock()
        request.customer_id = "cust-1"
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # Should have a product inquiry suggestion (e.g., "Tell me about X products")
        assert any("tell me about" in s.lower() or "products" in s.lower() for s in suggestions)

    @patch("recommendation_service.server._get_cross_user_popular_queries")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_tier1_warranty_no_brands_grammar(self, mock_catalog, mock_profile, mock_popular, servicer, mock_context):
        """Tier 1: warranty suggestion with empty brands_list uses correct grammar."""
        mock_catalog.return_value = {
            "brands": {},
            "all_product_names": [],
            "price_min": 50.0,
            "price_max": 500.0,
        }
        mock_profile.return_value = _make_customer_profile(
            has_history=True,
            topics_explored=["some topic"],
            brands_explored=set(),
            intents_history={"product_inquiry", "price_check", "comparison"},
        )
        mock_popular.return_value = []

        request = MagicMock()
        request.customer_id = "cust-1"
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # Must not contain "What warranty does products offer?" (bad grammar)
        for s in suggestions:
            assert "does products" not in s.lower(), f"Bad grammar in suggestion: {s}"
