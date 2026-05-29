"""Tests for recommendation_service — context-aware, memory-aware recommendations."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest

from recommendation_service.server import (
    RecommendationServiceServicer,
    _get_product_catalog_summary,
    _build_session_context,
    _build_customer_profile,
    _extract_current_focus,
    _get_cross_user_popular_products,
    _get_cooccurring_products,
    _get_premium_showcase_products,
    _find_price_alternative,
    _price_alternative_from_catalog,
    _build_focus_anchored_suggestions,
    _catalog_aware_generics,
    _catalog_cache,
    _catalog_lock,
    CATALOG_TTL,
    FALLBACK_BRANDS,
    INTENT_STRATEGY,
    DEFAULT_STRATEGY,
    ALL_INTENTS,
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
        "products_by_brand": {
            "UltraWasher": [
                {"name": "UltraWasher 8262", "price": 333.0, "warranty": 36},
                {"name": "UltraWasher 5155", "price": 121.0, "warranty": 12},
                {"name": "UltraWasher 5944", "price": 250.0, "warranty": 24},
            ],
            "RoboCleaner": [
                {"name": "RoboCleaner 3120", "price": 499.0, "warranty": 24},
                {"name": "RoboCleaner 8285", "price": 350.0, "warranty": 18},
                {"name": "RoboCleaner 4653", "price": 129.0, "warranty": 6},
            ],
            "PowerDrill": [
                {"name": "PowerDrill 5641", "price": 486.0, "warranty": 36},
                {"name": "PowerDrill 8255", "price": 200.0, "warranty": 12},
                {"name": "PowerDrill 9154", "price": 54.0, "warranty": 6},
            ],
            "MegaBlender": [
                {"name": "MegaBlender 5588", "price": 336.0, "warranty": 36},
                {"name": "MegaBlender 8913", "price": 150.0, "warranty": 12},
                {"name": "MegaBlender 9904", "price": 61.0, "warranty": 6},
            ],
            "EcoKettle": [
                {"name": "EcoKettle 1042", "price": 448.0, "warranty": 12},
                {"name": "EcoKettle 3468", "price": 200.0, "warranty": 6},
                {"name": "EcoKettle 8031", "price": 87.0, "warranty": 6},
            ],
        },
        "price_min": 54.0,
        "price_max": 499.0,
    }


def _make_catalog_with_products():
    """Helper: catalog with products_by_brand for premium showcase tests."""
    return _make_catalog()


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
        "current_product": None,
        "current_brand": None,
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
# Extract Current Focus
# ══════════════════════════════════════════════════════════════════

class TestExtractCurrentFocus:
    """_extract_current_focus derives product/brand from last exchange."""

    def test_product_from_response(self):
        catalog = _make_catalog()
        focus = _extract_current_focus(
            "Tell me about cleaners",
            "The RoboCleaner 3120 is our top-of-line cleaner at $499.",
            catalog,
        )
        assert focus["current_product"] == "RoboCleaner 3120"
        assert focus["current_brand"] == "RoboCleaner"

    def test_product_from_query_fallback(self):
        catalog = _make_catalog()
        focus = _extract_current_focus(
            "What about RoboCleaner 3120?",
            "Sure, let me look that up for you.",
            catalog,
        )
        assert focus["current_product"] == "RoboCleaner 3120"
        assert focus["current_brand"] == "RoboCleaner"

    def test_brand_only_fallback(self):
        """When text mentions brand but no full product name, brand-only fallback triggers."""
        # Use a catalog where product names are NOT substrings found in the text
        catalog = {
            "brands": {
                "Acme": {"count": 1, "price_min": 100, "price_max": 200, "warranties": [12]},
                "Zenith": {"count": 1, "price_min": 150, "price_max": 300, "warranties": [24]},
            },
            "all_product_names": ["Acme ProMax 7000", "Zenith Ultra 9000"],
            "products_by_brand": {},
            "price_min": 100.0,
            "price_max": 300.0,
        }
        focus = _extract_current_focus(
            "What does Zenith make?",
            "Zenith is a well-known brand with several options.",
            catalog,
        )
        # "Zenith Ultra 9000" is NOT in either text, but "Zenith" brand IS
        assert focus["current_product"] is None
        assert focus["current_brand"] == "Zenith"

    def test_brand_only_when_no_product_substring(self):
        """Brand name matches but no full product name appears."""
        catalog = {
            "brands": {"Acme": {"count": 1, "price_min": 100, "price_max": 200, "warranties": [12]}},
            "all_product_names": ["Acme Deluxe 3000"],
            "products_by_brand": {},
            "price_min": 100.0,
            "price_max": 200.0,
        }
        focus = _extract_current_focus(
            "Tell me about Acme",
            "Acme makes great products.",
            catalog,
        )
        # "Acme Deluxe 3000" won't be found in text, but "Acme" brand will
        assert focus["current_product"] is None
        assert focus["current_brand"] == "Acme"

    def test_no_match_returns_none(self):
        catalog = _make_catalog()
        focus = _extract_current_focus(
            "What's the weather today?",
            "I can only help with product questions.",
            catalog,
        )
        assert focus["current_product"] is None
        assert focus["current_brand"] is None

    def test_earliest_position_wins(self):
        catalog = _make_catalog()
        focus = _extract_current_focus(
            "Compare products",
            "PowerDrill 5641 costs more than EcoKettle 1042",
            catalog,
        )
        assert focus["current_product"] == "PowerDrill 5641"

    def test_deterministic_across_calls(self):
        catalog = _make_catalog()
        results = set()
        for _ in range(10):
            focus = _extract_current_focus(
                "What about these?",
                "The RoboCleaner 3120 and PowerDrill 5641 are popular.",
                catalog,
            )
            results.add(focus["current_product"])
        assert len(results) == 1  # Always the same product

    def test_empty_catalog(self):
        catalog = {"brands": {}, "all_product_names": [], "products_by_brand": {}, "price_min": 50.0, "price_max": 500.0}
        focus = _extract_current_focus("Tell me about something", "Here's info.", catalog)
        assert focus["current_product"] is None
        assert focus["current_brand"] is None

    def test_none_inputs(self):
        catalog = _make_catalog()
        focus = _extract_current_focus(None, None, catalog)
        assert focus["current_product"] is None
        assert focus["current_brand"] is None

    def test_longer_match_preferred_at_same_position(self):
        """When two products start at the same position, longer name wins."""
        catalog = {
            "brands": {
                "PowerDrill": {"count": 2, "price_min": 100, "price_max": 300, "warranties": [12, 24]},
            },
            "all_product_names": ["PowerDrill 5", "PowerDrill 5641"],
            "products_by_brand": {
                "PowerDrill": [
                    {"name": "PowerDrill 5", "price": 100, "warranty": 12},
                    {"name": "PowerDrill 5641", "price": 300, "warranty": 24},
                ],
            },
            "price_min": 100.0,
            "price_max": 300.0,
        }
        focus = _extract_current_focus(
            "Tell me about it",
            "The PowerDrill 5641 is a great choice.",
            catalog,
        )
        # "PowerDrill 5" and "PowerDrill 5641" both match at same position
        # but "PowerDrill 5641" is longer and more specific
        assert focus["current_product"] == "PowerDrill 5641"

    def test_whitespace_only_product_name_in_catalog(self):
        """Whitespace-only product names in catalog don't crash extraction."""
        catalog = {
            "brands": {"Acme": {"count": 1, "price_min": 100, "price_max": 200, "warranties": [12]}},
            "all_product_names": ["  ", "Acme ProMax 7000"],
            "products_by_brand": {},
            "price_min": 100.0,
            "price_max": 200.0,
        }
        # Should not raise; whitespace-only names should be harmless
        focus = _extract_current_focus(
            "Tell me about Acme ProMax 7000",
            "Here's the info.",
            catalog,
        )
        assert focus["current_product"] == "Acme ProMax 7000"


# ══════════════════════════════════════════════════════════════════
# Cold Start Tier 1C — Returning User Override
# ══════════════════════════════════════════════════════════════════

class TestColdStartTier1C:
    """Tier 1C — Returning user gets 1 personal + platform suggestions."""

    @patch("recommendation_service.server._get_premium_showcase_products")
    @patch("recommendation_service.server._get_cross_user_popular_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_returning_user_blend(self, mock_catalog, mock_profile, mock_popular, mock_premium, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_profile.return_value = _make_customer_profile(
            has_history=True,
            topics_explored=["UltraWasher 8262", "price comparison"],
            brands_explored={"UltraWasher"},
            intents_history={"product_inquiry"},
        )
        mock_popular.return_value = [
            {"product": "RoboCleaner 3120", "brand": "RoboCleaner", "customer_count": 5},
            {"product": "PowerDrill 5641", "brand": "PowerDrill", "customer_count": 3},
        ]
        mock_premium.return_value = []

        request = MagicMock()
        request.customer_id = "cust-1"
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) >= 3
        # First suggestion should be personal (continue where left off)
        assert "UltraWasher 8262" in suggestions[0]
        # Should also have platform suggestions
        assert any("RoboCleaner" in s or "PowerDrill" in s for s in suggestions[1:])

    @patch("recommendation_service.server._get_premium_showcase_products")
    @patch("recommendation_service.server._get_cross_user_popular_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_returning_user_no_popular_blends_with_showcase(
        self, mock_catalog, mock_profile, mock_popular, mock_premium, servicer, mock_context
    ):
        mock_catalog.return_value = _make_catalog()
        mock_profile.return_value = _make_customer_profile(
            has_history=True,
            topics_explored=["PowerDrill 5641"],
            brands_explored={"PowerDrill"},
        )
        mock_popular.return_value = []
        mock_premium.return_value = [
            {"product": "RoboCleaner 3120", "brand": "RoboCleaner", "price": 499.0},
            {"product": "PowerDrill 5641", "brand": "PowerDrill", "price": 486.0},
        ]

        request = MagicMock()
        request.customer_id = "cust-2"
        request.session_id = "sess-2"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) >= 3
        # Personal + showcase blend
        assert "PowerDrill 5641" in suggestions[0]
        assert any("premium" in s.lower() or "Check out" in s for s in suggestions)

    @patch("recommendation_service.server._get_premium_showcase_products")
    @patch("recommendation_service.server._get_cross_user_popular_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_returning_user_dedup_against_history(
        self, mock_catalog, mock_profile, mock_popular, mock_premium, servicer, mock_context
    ):
        """Skip products already explored in personal history."""
        mock_catalog.return_value = _make_catalog()
        mock_profile.return_value = _make_customer_profile(
            has_history=True,
            topics_explored=["RoboCleaner 3120"],
            brands_explored={"RoboCleaner"},
        )
        # Popular product is the same as history — should still work without dups
        mock_popular.return_value = [
            {"product": "PowerDrill 5641", "brand": "PowerDrill", "customer_count": 5},
        ]
        mock_premium.return_value = []

        request = MagicMock()
        request.customer_id = "cust-3"
        request.session_id = "sess-3"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) >= 3
        # No duplicate suggestions
        assert len(suggestions) == len(set(s.lower() for s in suggestions))


# ══════════════════════════════════════════════════════════════════
# Cold Start Tier 1A — Cross-User Popular Products
# ══════════════════════════════════════════════════════════════════

class TestColdStartTier1A:
    """Tier 1A — Cross-user popular products by entity aggregation."""

    @patch("recommendation_service.server.get_pg_conn")
    def test_popular_products_entity_aggregation(self, mock_pg):
        """Count by product entity, not raw query text."""
        catalog = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        # Two different queries mentioning the same product from different customers
        cursor.fetchall.return_value = [
            {"content": "Tell me about RoboCleaner 3120", "customer_id": "cust-1"},
            {"content": "How much is RoboCleaner 3120?", "customer_id": "cust-2"},
            {"content": "Show me PowerDrill 5641", "customer_id": "cust-3"},
        ]

        result = _get_cross_user_popular_products(catalog, limit=3)

        assert len(result) >= 1
        # RoboCleaner 3120 has 2 unique customers, should be first
        assert result[0]["product"] == "RoboCleaner 3120"
        assert result[0]["customer_count"] == 2

    @patch("recommendation_service.server.get_pg_conn")
    def test_popular_products_cross_user_dedup(self, mock_pg):
        """Same user asking twice counts as 1."""
        catalog = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchall.return_value = [
            {"content": "Tell me about RoboCleaner 3120", "customer_id": "cust-1"},
            {"content": "RoboCleaner 3120 warranty?", "customer_id": "cust-1"},
            {"content": "PowerDrill 5641 price", "customer_id": "cust-2"},
            {"content": "PowerDrill 5641 warranty", "customer_id": "cust-3"},
        ]

        result = _get_cross_user_popular_products(catalog, limit=3)

        # PowerDrill 5641 has 2 unique customers, RoboCleaner 3120 has 1
        assert result[0]["product"] == "PowerDrill 5641"
        assert result[0]["customer_count"] == 2
        assert result[1]["product"] == "RoboCleaner 3120"
        assert result[1]["customer_count"] == 1

    @patch("recommendation_service.server._get_premium_showcase_products")
    @patch("recommendation_service.server._get_cross_user_popular_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_natural_suggestion_format(self, mock_catalog, mock_profile, mock_popular, mock_premium, servicer, mock_context):
        """Suggestions include brand context."""
        mock_catalog.return_value = _make_catalog()
        mock_profile.return_value = _make_customer_profile(has_history=False)
        mock_popular.return_value = [
            {"product": "RoboCleaner 3120", "brand": "RoboCleaner", "customer_count": 5},
        ]
        mock_premium.return_value = []

        request = MagicMock()
        request.customer_id = ""
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # Should have readable suggestion with brand context
        popular_suggestions = [s for s in suggestions if "most popular" in s.lower()]
        assert len(popular_suggestions) >= 1
        assert "RoboCleaner" in popular_suggestions[0]

    @patch("recommendation_service.server._get_premium_showcase_products")
    @patch("recommendation_service.server._get_cross_user_popular_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_fallback_to_1B_when_no_popular(self, mock_catalog, mock_profile, mock_popular, mock_premium, servicer, mock_context):
        """Empty popular products → falls to Tier 1B."""
        mock_catalog.return_value = _make_catalog()
        mock_profile.return_value = _make_customer_profile(has_history=False)
        mock_popular.return_value = []
        mock_premium.return_value = [
            {"product": "RoboCleaner 3120", "brand": "RoboCleaner", "price": 499.0},
        ]

        request = MagicMock()
        request.customer_id = ""
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) >= 3
        assert any("premium" in s.lower() or "Check out" in s for s in suggestions)


# ══════════════════════════════════════════════════════════════════
# Cold Start Tier 1B — Premium Showcase
# ══════════════════════════════════════════════════════════════════

class TestPremiumShowcase:
    """Premium showcase — most expensive per brand."""

    @patch("recommendation_service.server.get_pg_conn")
    def test_top_priced_per_brand(self, mock_pg):
        catalog = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchall.return_value = [
            {"product_name": "RoboCleaner 3120", "price": 499.0, "brand": "RoboCleaner"},
            {"product_name": "PowerDrill 5641", "price": 486.0, "brand": "PowerDrill"},
            {"product_name": "EcoKettle 1042", "price": 448.0, "brand": "EcoKettle"},
        ]

        result = _get_premium_showcase_products(catalog, limit=3)

        assert len(result) == 3
        # Should be sorted by price descending
        assert result[0]["price"] >= result[1]["price"]
        assert result[1]["price"] >= result[2]["price"]

    @patch("recommendation_service.server.get_pg_conn")
    def test_distinct_brands_enforced(self, mock_pg):
        catalog = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchall.return_value = [
            {"product_name": "RoboCleaner 3120", "price": 499.0, "brand": "RoboCleaner"},
            {"product_name": "RoboCleaner 8285", "price": 350.0, "brand": "RoboCleaner"},
            {"product_name": "PowerDrill 5641", "price": 486.0, "brand": "PowerDrill"},
            {"product_name": "EcoKettle 1042", "price": 448.0, "brand": "EcoKettle"},
        ]

        result = _get_premium_showcase_products(catalog, limit=3)

        brands = [r["brand"] for r in result]
        assert len(brands) == len(set(brands)), "Duplicate brands found"

    @patch("recommendation_service.server.get_pg_conn")
    def test_db_failure_fallback_to_catalog_summary(self, mock_pg):
        catalog = _make_catalog()
        mock_pg.side_effect = Exception("DB down")

        result = _get_premium_showcase_products(catalog, limit=3)

        assert len(result) == 3
        # Should use catalog's products_by_brand
        brands = [r["brand"] for r in result]
        assert len(brands) == len(set(brands))
        # All brands should be from catalog
        for r in result:
            assert r["brand"] in catalog["brands"]


# ══════════════════════════════════════════════════════════════════
# Cold Start Fallback Cascade
# ══════════════════════════════════════════════════════════════════

class TestColdStartFallbackCascade:
    """Tier cascading: 1C → 1A → 1B → generic."""

    @patch("recommendation_service.server._get_premium_showcase_products")
    @patch("recommendation_service.server._get_cross_user_popular_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_tier1c_failure_cascades(self, mock_catalog, mock_profile, mock_popular, mock_premium, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_profile.side_effect = Exception("Memory service unavailable")
        mock_popular.return_value = [
            {"product": "PowerDrill 5641", "brand": "PowerDrill", "customer_count": 5},
        ]
        mock_premium.return_value = []

        request = MagicMock()
        request.customer_id = "cust-1"
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) >= 3

    @patch("recommendation_service.server._get_premium_showcase_products")
    @patch("recommendation_service.server._get_cross_user_popular_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_all_tiers_fail_still_returns(self, mock_catalog, mock_profile, mock_popular, mock_premium, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_profile.side_effect = Exception("Memory down")
        mock_popular.side_effect = Exception("DB down")
        mock_premium.side_effect = Exception("DB down")

        request = MagicMock()
        request.customer_id = "cust-1"
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # Generic catalog-aware defaults should still work
        assert len(suggestions) >= 3

    @patch("recommendation_service.server._get_premium_showcase_products")
    @patch("recommendation_service.server._get_cross_user_popular_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_max_5_suggestions(self, mock_catalog, mock_profile, mock_popular, mock_premium, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_profile.return_value = _make_customer_profile(
            has_history=True,
            topics_explored=["UltraWasher", "PowerDrill"],
            brands_explored={"UltraWasher"},
        )
        mock_popular.return_value = [
            {"product": f"RoboCleaner {3120 + i}", "brand": "RoboCleaner", "customer_count": 10 - i}
            for i in range(5)
        ]
        mock_premium.return_value = [
            {"product": "EcoKettle 1042", "brand": "EcoKettle", "price": 448.0},
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
    """Follow-up suggestions are product-focused, anchored to current focus."""

    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_product_inquiry_strategy(self, mock_catalog, mock_session, mock_profile, mock_alt, mock_cooccur, servicer, mock_context):
        """After product inquiry, suggest warranty and price of same product."""
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"product_inquiry"},
            products_mentioned={"RoboCleaner 3120"},
            brands_mentioned={"RoboCleaner"},
            last_intent="product_inquiry",
            last_user_query="Tell me about RoboCleaner 3120",
            current_product="RoboCleaner 3120",
            current_brand="RoboCleaner",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = ["PowerDrill 5641"]
        mock_alt.return_value = None

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "Tell me about RoboCleaner 3120"
        request.last_response = "The RoboCleaner 3120 is a premium robotic vacuum."
        request.intent = "product_inquiry"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) == 3
        # Should mention RoboCleaner 3120 in warranty/price suggestions
        assert any("RoboCleaner 3120" in s and "warranty" in s.lower() for s in suggestions)
        assert any("RoboCleaner 3120" in s and "cost" in s.lower() for s in suggestions)

    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_price_check_strategy(self, mock_catalog, mock_session, mock_profile, mock_alt, mock_cooccur, servicer, mock_context):
        """After price check, suggest warranty and comparison."""
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"price_check"},
            products_mentioned={"PowerDrill 5641"},
            brands_mentioned={"PowerDrill"},
            last_intent="price_check",
            last_user_query="How much does PowerDrill 5641 cost?",
            current_product="PowerDrill 5641",
            current_brand="PowerDrill",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = "RoboCleaner 3120"

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "How much does PowerDrill 5641 cost?"
        request.last_response = "PowerDrill 5641 costs $486.00"
        request.intent = "price_check"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) == 3
        assert any("PowerDrill 5641" in s and "warranty" in s.lower() for s in suggestions)

    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_warranty_question_strategy(self, mock_catalog, mock_session, mock_profile, mock_alt, mock_cooccur, servicer, mock_context):
        """After warranty question, suggest price and cross-brand comparison."""
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"warranty_question"},
            products_mentioned={"EcoKettle 1042"},
            brands_mentioned={"EcoKettle"},
            last_intent="warranty_question",
            last_user_query="What's the warranty on EcoKettle 1042?",
            current_product="EcoKettle 1042",
            current_brand="EcoKettle",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = ["MegaBlender 5588"]
        mock_alt.return_value = None

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "What's the warranty on EcoKettle 1042?"
        request.last_response = "EcoKettle 1042 comes with a 12-month warranty."
        request.intent = "warranty_question"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) == 3
        assert any("EcoKettle 1042" in s and "cost" in s.lower() for s in suggestions)

    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_comparison_strategy(self, mock_catalog, mock_session, mock_profile, mock_alt, mock_cooccur, servicer, mock_context):
        """After comparison, suggest features and price."""
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"comparison"},
            products_mentioned={"MegaBlender 5588", "PowerDrill 5641"},
            brands_mentioned={"MegaBlender", "PowerDrill"},
            last_intent="comparison",
            last_user_query="Compare MegaBlender 5588 with PowerDrill 5641",
            current_product="MegaBlender 5588",
            current_brand="MegaBlender",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = "EcoKettle 1042"

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "Compare MegaBlender 5588 with PowerDrill 5641"
        request.last_response = "MegaBlender 5588 is $336 while PowerDrill 5641 is $486."
        request.intent = "comparison"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) == 3
        assert any("MegaBlender 5588" in s for s in suggestions)

    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_follow_up_uses_previous_intent(self, mock_catalog, mock_session, mock_profile, mock_alt, mock_cooccur, servicer, mock_context):
        """follow_up intent resolves to previous intent's strategy."""
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"product_inquiry", "follow_up"},
            products_mentioned={"UltraWasher 8262"},
            brands_mentioned={"UltraWasher"},
            last_intent="product_inquiry",
            last_user_query="Tell me more",
            current_product="UltraWasher 8262",
            current_brand="UltraWasher",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = "RoboCleaner 3120"

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "Tell me more"
        request.last_response = "The UltraWasher 8262 also features..."
        request.intent = "follow_up"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) == 3
        # Should use product_inquiry strategy since last_intent is product_inquiry
        assert any("UltraWasher 8262" in s for s in suggestions)
        # Verify it used product_inquiry strategy (has warranty/cost mentions)
        # and not just DEFAULT_STRATEGY
        combined = " ".join(suggestions).lower()
        assert "warranty" in combined or "cost" in combined or "price" in combined, (
            f"Expected product_inquiry strategy keywords but got: {suggestions}"
        )

    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_cross_user_slot3(self, mock_catalog, mock_session, mock_profile, mock_alt, mock_cooccur, servicer, mock_context):
        """Slot 3 uses co-occurrence data."""
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"product_inquiry"},
            products_mentioned={"RoboCleaner 3120"},
            brands_mentioned={"RoboCleaner"},
            last_intent="product_inquiry",
            current_product="RoboCleaner 3120",
            current_brand="RoboCleaner",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = ["PowerDrill 5641"]
        mock_alt.return_value = None

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "Tell me about RoboCleaner 3120"
        request.last_response = "The RoboCleaner 3120 is great."
        request.intent = "product_inquiry"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # Slot 3 should reference co-occurring product
        assert any("PowerDrill 5641" in s for s in suggestions)

    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_slot3_fallback_price_alternative(self, mock_catalog, mock_session, mock_profile, mock_alt, mock_cooccur, servicer, mock_context):
        """No co-occurrence → slot 3 uses price alternative."""
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"product_inquiry"},
            products_mentioned={"PowerDrill 5641"},
            brands_mentioned={"PowerDrill"},
            last_intent="product_inquiry",
            current_product="PowerDrill 5641",
            current_brand="PowerDrill",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = "RoboCleaner 3120"

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "Tell me about PowerDrill 5641"
        request.last_response = "PowerDrill 5641 is a heavy-duty drill."
        request.intent = "product_inquiry"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # Slot 3 should be a comparison with price alternative
        assert any("RoboCleaner 3120" in s for s in suggestions)

    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_no_echo_of_last_query(self, mock_catalog, mock_session, mock_profile, mock_alt, mock_cooccur, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"product_inquiry"},
            products_mentioned={"PowerDrill 5641"},
            brands_mentioned={"PowerDrill"},
            last_intent="product_inquiry",
            last_user_query="How much does PowerDrill 5641 cost?",
            current_product="PowerDrill 5641",
            current_brand="PowerDrill",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = "RoboCleaner 3120"

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "How much does PowerDrill 5641 cost?"
        request.last_response = "PowerDrill 5641 costs $380.88"
        request.intent = "price_check"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert "How much does PowerDrill 5641 cost?" not in suggestions

    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_deduplication(self, mock_catalog, mock_session, mock_profile, mock_alt, mock_cooccur, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"product_inquiry"},
            products_mentioned={"MegaBlender 5588"},
            brands_mentioned={"MegaBlender"},
            last_intent="product_inquiry",
            current_product="MegaBlender 5588",
            current_brand="MegaBlender",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = "PowerDrill 5641"

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

    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_exactly_3_suggestions(self, mock_catalog, mock_session, mock_profile, mock_alt, mock_cooccur, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"price_check"},
            products_mentioned=set(),
            brands_mentioned=set(),
            last_intent="price_check",
            current_product=None,
            current_brand=None,
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = None

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "What's the cheapest product?"
        request.last_response = "The cheapest is PowerDrill 9154 at $54.00"
        request.intent = "price_check"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        assert len(list(response.suggestions)) == 3


# ══════════════════════════════════════════════════════════════════
# Intent Strategy
# ══════════════════════════════════════════════════════════════════

class TestIntentStrategy:
    """Verify strategy dict coverage and fillability."""

    def test_all_core_intents_have_strategies(self):
        """Every core intent has a strategy."""
        core_intents = {"product_inquiry", "price_check", "warranty_question", "comparison", "session_query"}
        for intent in core_intents:
            assert intent in INTENT_STRATEGY, f"Missing strategy for {intent}"

    def test_each_strategy_has_2_templates(self):
        for intent, templates in INTENT_STRATEGY.items():
            assert len(templates) == 2, f"{intent} should have 2 templates, has {len(templates)}"

    def test_templates_contain_product_placeholder(self):
        """At least one template per strategy references {current_product}."""
        for intent, templates in INTENT_STRATEGY.items():
            if intent == "session_query":
                continue  # session_query has special templates
            assert any("{current_product}" in t for t in templates), (
                f"{intent} has no template with {{current_product}}"
            )

    def test_default_strategy_exists(self):
        assert len(DEFAULT_STRATEGY) == 2
        assert all("{current_product}" in t for t in DEFAULT_STRATEGY)

    def test_follow_up_not_in_strategy(self):
        """follow_up resolves to previous intent — no direct strategy."""
        assert "follow_up" not in INTENT_STRATEGY


# ══════════════════════════════════════════════════════════════════
# Cross-User Popular Products
# ══════════════════════════════════════════════════════════════════

class TestCrossUserPopularProducts:
    """Cross-user popular product entity aggregation."""

    @patch("recommendation_service.server.get_pg_conn")
    def test_returns_ranked_products(self, mock_pg):
        catalog = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchall.return_value = [
            {"content": "Tell me about RoboCleaner 3120", "customer_id": "cust-1"},
            {"content": "RoboCleaner 3120 price", "customer_id": "cust-2"},
            {"content": "PowerDrill 5641 info", "customer_id": "cust-3"},
        ]

        result = _get_cross_user_popular_products(catalog, limit=3)

        assert len(result) == 2
        assert result[0]["product"] == "RoboCleaner 3120"
        assert result[0]["brand"] == "RoboCleaner"

    @patch("recommendation_service.server.get_pg_conn")
    def test_empty_db_returns_empty(self, mock_pg):
        catalog = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        cursor.fetchall.return_value = []

        result = _get_cross_user_popular_products(catalog, limit=3)

        assert result == []

    @patch("recommendation_service.server.get_pg_conn")
    def test_db_failure_returns_empty(self, mock_pg):
        catalog = _make_catalog()
        mock_pg.side_effect = Exception("DB down")

        result = _get_cross_user_popular_products(catalog, limit=3)

        assert result == []

    def test_empty_catalog_returns_empty(self):
        catalog = {"brands": {}, "all_product_names": [], "products_by_brand": {}, "price_min": 50.0, "price_max": 500.0}
        result = _get_cross_user_popular_products(catalog, limit=3)
        assert result == []


# ══════════════════════════════════════════════════════════════════
# Co-occurring Products
# ══════════════════════════════════════════════════════════════════

class TestCooccurringProducts:
    """Co-occurring product discovery across sessions."""

    @patch("recommendation_service.server.get_pg_conn")
    def test_finds_cooccurring(self, mock_pg):
        catalog = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        # Step 1: sessions with current product
        # Step 2: content from those sessions
        cursor.fetchall.side_effect = [
            [{"session_id": "sess-1"}, {"session_id": "sess-2"}],
            [
                {"content": "Tell me about RoboCleaner 3120"},
                {"content": "Also show PowerDrill 5641"},
                {"content": "What about EcoKettle 1042?"},
                {"content": "PowerDrill 5641 price"},
            ],
        ]

        result = _get_cooccurring_products("RoboCleaner 3120", catalog, limit=3)

        assert "PowerDrill 5641" in result
        assert "EcoKettle 1042" in result
        assert "RoboCleaner 3120" not in result

    @patch("recommendation_service.server.get_pg_conn")
    def test_excludes_current_product(self, mock_pg):
        catalog = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchall.side_effect = [
            [{"session_id": "sess-1"}],
            [{"content": "RoboCleaner 3120 is great. Also RoboCleaner 3120 again."}],
        ]

        result = _get_cooccurring_products("RoboCleaner 3120", catalog, limit=3)

        assert "RoboCleaner 3120" not in result

    @patch("recommendation_service.server.get_pg_conn")
    def test_db_failure_returns_empty(self, mock_pg):
        catalog = _make_catalog()
        mock_pg.side_effect = Exception("DB down")

        result = _get_cooccurring_products("RoboCleaner 3120", catalog, limit=3)

        assert result == []

    def test_none_product_returns_empty(self):
        catalog = _make_catalog()
        result = _get_cooccurring_products(None, catalog, limit=3)
        assert result == []


# ══════════════════════════════════════════════════════════════════
# Price Alternative
# ══════════════════════════════════════════════════════════════════

class TestPriceAlternative:
    """_find_price_alternative and _price_alternative_from_catalog."""

    @patch("recommendation_service.server.get_pg_conn")
    def test_finds_similar_price_different_brand(self, mock_pg):
        catalog = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        # First query: get current product price
        # Second query: find alternative
        cursor.fetchone.side_effect = [
            {"price": 486.0},
            {"product_name": "RoboCleaner 3120", "price": 499.0},
        ]

        result = _find_price_alternative("PowerDrill 5641", "PowerDrill", catalog)

        assert result == "RoboCleaner 3120"

    @patch("recommendation_service.server.get_pg_conn")
    def test_no_db_alternative_falls_to_catalog(self, mock_pg):
        catalog = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        # Price found, but no alternative in range
        cursor.fetchone.side_effect = [
            {"price": 486.0},
            None,
        ]

        result = _find_price_alternative("PowerDrill 5641", "PowerDrill", catalog)

        # Catalog fallback: first product from a different brand
        assert result is not None
        assert not result.startswith("PowerDrill")

    @patch("recommendation_service.server.get_pg_conn")
    def test_none_brand_still_works(self, mock_pg):
        """current_brand=None should not produce bad SQL."""
        catalog = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn

        cursor.fetchone.side_effect = [
            {"price": 200.0},
            {"product_name": "UltraWasher 8262", "price": 210.0},
        ]

        result = _find_price_alternative("SomeProd 123", None, catalog)

        assert result == "UltraWasher 8262"
        # Verify the SQL used != instead of LIKE
        second_call_sql = cursor.execute.call_args_list[1][0][0]
        assert "!=" in second_call_sql
        assert "LIKE" not in second_call_sql

    def test_none_product_returns_none(self):
        catalog = _make_catalog()
        result = _find_price_alternative(None, None, catalog)
        assert result is None

    @patch("recommendation_service.server.get_pg_conn")
    def test_db_failure_falls_to_catalog(self, mock_pg):
        catalog = _make_catalog()
        mock_pg.side_effect = Exception("DB down")

        result = _find_price_alternative("PowerDrill 5641", "PowerDrill", catalog)

        # Should return catalog fallback (first non-PowerDrill product)
        assert result is not None
        assert not result.startswith("PowerDrill")

    def test_catalog_fallback_single_brand_returns_none(self):
        """If catalog has only one brand, no alternative exists."""
        catalog = {
            "brands": {"OnlyBrand": {"count": 1}},
            "all_product_names": ["OnlyBrand 1000"],
            "products_by_brand": {},
            "price_min": 100.0,
            "price_max": 200.0,
        }
        result = _price_alternative_from_catalog("OnlyBrand 1000", "OnlyBrand", catalog)
        assert result is None


# ══════════════════════════════════════════════════════════════════
# Catalog-Aware Generics
# ══════════════════════════════════════════════════════════════════

class TestCatalogAwareGenerics:
    """_catalog_aware_generics padding and dedup behavior."""

    def test_returns_at_least_3_with_brands(self):
        catalog = _make_catalog()
        result = _catalog_aware_generics(catalog)
        assert len(result) >= 3

    def test_returns_at_least_3_without_brands(self):
        catalog = {"brands": {}, "all_product_names": [], "products_by_brand": {}, "price_min": 50.0, "price_max": 500.0}
        result = _catalog_aware_generics(catalog)
        assert len(result) >= 3

    def test_exclude_filters_suggestions(self):
        catalog = _make_catalog()
        brand_list = list(catalog["brands"].keys())
        first_suggestion = f"Tell me about {brand_list[0]} products"
        result = _catalog_aware_generics(catalog, exclude={first_suggestion})
        assert first_suggestion not in result

    def test_no_duplicates(self):
        catalog = _make_catalog()
        result = _catalog_aware_generics(catalog)
        assert len(result) == len(set(r.lower() for r in result))

    def test_exclude_case_insensitive(self):
        catalog = _make_catalog()
        brand_list = list(catalog["brands"].keys())
        first_suggestion = f"Tell me about {brand_list[0]} products"
        result = _catalog_aware_generics(catalog, exclude={first_suggestion.upper()})
        assert first_suggestion not in result


# ══════════════════════════════════════════════════════════════════
# Build Focus-Anchored Suggestions
# ══════════════════════════════════════════════════════════════════

class TestBuildFocusAnchoredSuggestions:
    """Focus-anchored suggestion builder."""

    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._get_cooccurring_products")
    def test_anchored_to_product(self, mock_cooccur, mock_alt):
        """Suggestions are anchored to the current product."""
        catalog = _make_catalog()
        focus = {"current_product": "RoboCleaner 3120", "current_brand": "RoboCleaner"}
        session_ctx = _make_session_context(products_mentioned={"RoboCleaner 3120"})
        profile = _make_customer_profile()
        mock_cooccur.return_value = ["PowerDrill 5641"]
        mock_alt.return_value = None

        suggestions = _build_focus_anchored_suggestions(
            focus, "product_inquiry", session_ctx, catalog, profile,
        )

        assert len(suggestions) >= 2
        assert all("RoboCleaner 3120" in s or "PowerDrill" in s for s in suggestions)

    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._get_cooccurring_products")
    def test_follow_up_resolves_to_last_intent(self, mock_cooccur, mock_alt):
        """follow_up intent uses the previous intent's strategy."""
        catalog = _make_catalog()
        focus = {"current_product": "EcoKettle 1042", "current_brand": "EcoKettle"}
        session_ctx = _make_session_context(
            last_intent="warranty_question",
            products_mentioned={"EcoKettle 1042"},
        )
        profile = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = "MegaBlender 5588"

        suggestions = _build_focus_anchored_suggestions(
            focus, "follow_up", session_ctx, catalog, profile,
        )

        # Should use warranty_question strategy: price + compare warranty
        assert any("EcoKettle 1042" in s and "cost" in s.lower() for s in suggestions)

    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._get_cooccurring_products")
    def test_slot3_cooccurrence(self, mock_cooccur, mock_alt):
        """Slot 3 uses co-occurrence when available."""
        catalog = _make_catalog()
        focus = {"current_product": "RoboCleaner 3120", "current_brand": "RoboCleaner"}
        session_ctx = _make_session_context(products_mentioned={"RoboCleaner 3120"})
        profile = _make_customer_profile()
        mock_cooccur.return_value = ["PowerDrill 5641"]
        mock_alt.return_value = None

        suggestions = _build_focus_anchored_suggestions(
            focus, "product_inquiry", session_ctx, catalog, profile,
        )

        assert any("PowerDrill 5641" in s for s in suggestions)

    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._get_cooccurring_products")
    def test_slot3_fallback_to_price_alt(self, mock_cooccur, mock_alt):
        """No co-occurrence → slot 3 uses price alternative."""
        catalog = _make_catalog()
        focus = {"current_product": "PowerDrill 5641", "current_brand": "PowerDrill"}
        session_ctx = _make_session_context(products_mentioned={"PowerDrill 5641"})
        profile = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = "RoboCleaner 3120"

        suggestions = _build_focus_anchored_suggestions(
            focus, "product_inquiry", session_ctx, catalog, profile,
        )

        assert any("RoboCleaner 3120" in s for s in suggestions)

    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._get_cooccurring_products")
    def test_follow_up_chain_uses_default(self, mock_cooccur, mock_alt):
        """follow_up with last_intent also follow_up uses DEFAULT_STRATEGY."""
        catalog = _make_catalog()
        focus = {"current_product": "RoboCleaner 3120", "current_brand": "RoboCleaner"}
        session_ctx = _make_session_context(
            last_intent="follow_up",
            products_mentioned={"RoboCleaner 3120"},
        )
        profile = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = "PowerDrill 5641"

        suggestions = _build_focus_anchored_suggestions(
            focus, "follow_up", session_ctx, catalog, profile,
        )

        # Should use DEFAULT_STRATEGY since follow_up can't resolve to non-follow_up
        assert len(suggestions) >= 2
        assert any("RoboCleaner 3120" in s for s in suggestions)

    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._get_cooccurring_products")
    def test_episodic_dedup(self, mock_cooccur, mock_alt):
        """Co-occurring products already in episodic memory are skipped."""
        catalog = _make_catalog()
        focus = {"current_product": "RoboCleaner 3120", "current_brand": "RoboCleaner"}
        session_ctx = _make_session_context(products_mentioned={"RoboCleaner 3120"})
        profile = _make_customer_profile(
            topics_explored=["PowerDrill 5641"],
            has_history=True,
        )
        mock_cooccur.return_value = ["PowerDrill 5641", "EcoKettle 1042"]
        mock_alt.return_value = None

        suggestions = _build_focus_anchored_suggestions(
            focus, "product_inquiry", session_ctx, catalog, profile,
        )

        # PowerDrill 5641 is in episodic memory — should be skipped
        slot3_candidates = [s for s in suggestions if "also explored" in s.lower()]
        if slot3_candidates:
            assert "EcoKettle 1042" in slot3_candidates[0]
            assert "PowerDrill 5641" not in slot3_candidates[0]


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

    @patch("recommendation_service.server.get_pg_conn")
    def test_catalog_includes_products_by_brand(self, mock_pg):
        """Catalog summary includes products_by_brand dict."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        cursor.fetchall.return_value = [
            {"product_name": "TestBrand 1000", "price": 100.0, "warranty_months": 12},
            {"product_name": "TestBrand 2000", "price": 200.0, "warranty_months": 24},
        ]

        result = _get_product_catalog_summary()

        assert "products_by_brand" in result
        assert "TestBrand" in result["products_by_brand"]
        assert len(result["products_by_brand"]["TestBrand"]) == 2

    @patch("recommendation_service.server.get_pg_conn")
    def test_fallback_includes_products_by_brand(self, mock_pg):
        """Fallback catalog also includes products_by_brand."""
        mock_pg.side_effect = Exception("DB down")

        result = _get_product_catalog_summary()

        assert "products_by_brand" in result
        assert len(result["products_by_brand"]) == len(FALLBACK_BRANDS)


# ══════════════════════════════════════════════════════════════════
# Session Context
# ══════════════════════════════════════════════════════════════════

class TestSessionContext:
    """Session context builder extracts intents, products, brands, tools, current focus."""

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

    @patch("recommendation_service.server.get_pg_conn")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_extracts_current_focus(self, mock_catalog, mock_pg):
        mock_catalog.return_value = _make_catalog()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pg.return_value = conn
        cursor.fetchall.return_value = [
            {"role": "user", "content": "Tell me about RoboCleaner 3120", "intent": "product_inquiry", "tool_calls": None},
            {"role": "assistant", "content": "The RoboCleaner 3120 costs $499.", "intent": None, "tool_calls": None},
        ]

        ctx = _build_session_context("sess-1")

        assert ctx["current_product"] == "RoboCleaner 3120"
        assert ctx["current_brand"] == "RoboCleaner"

    def test_empty_session_id(self):
        ctx = _build_session_context("")
        assert ctx["intents_used"] == set()
        assert ctx["products_mentioned"] == set()
        assert ctx["current_product"] is None
        assert ctx["current_brand"] is None


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

    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server.get_pg_conn")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_all_services_down_followup(self, mock_catalog, mock_session, mock_profile, mock_pg, mock_cooccur, mock_alt, servicer, mock_context):
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context()
        mock_profile.return_value = _make_customer_profile()
        mock_pg.side_effect = Exception("DB down")
        mock_cooccur.return_value = []
        mock_alt.return_value = None

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "something"
        request.last_response = "response"
        request.intent = "product_inquiry"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) == 3

    @patch("recommendation_service.server._get_premium_showcase_products")
    @patch("recommendation_service.server._get_cross_user_popular_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_catalog_fallback_still_works(self, mock_catalog, mock_profile, mock_popular, mock_premium, servicer, mock_context):
        mock_catalog.return_value = {
            "brands": dict(FALLBACK_BRANDS),
            "all_product_names": [f"{b} 1000" for b in FALLBACK_BRANDS],
            "products_by_brand": {b: [{"name": f"{b} 1000", "price": info["price_max"], "warranty": 12}] for b, info in FALLBACK_BRANDS.items()},
            "price_min": 50.0,
            "price_max": 500.0,
        }
        mock_profile.return_value = _make_customer_profile()
        mock_popular.return_value = []
        mock_premium.return_value = []

        request = MagicMock()
        request.customer_id = ""
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) >= 3


# ══════════════════════════════════════════════════════════════════
# Gap Progression → Intent Strategy Progression
# ══════════════════════════════════════════════════════════════════

class TestGapProgression:
    """After using some intents, follow-up suggestions adapt based on intent strategy."""

    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_warranty_and_comparison_suggested_after_inquiry_and_price(
        self, mock_catalog, mock_session, mock_profile, mock_alt, mock_cooccur, servicer, mock_context
    ):
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"product_inquiry", "price_check"},
            brands_mentioned={"UltraWasher"},
            products_mentioned={"UltraWasher 8262"},
            last_intent="price_check",
            last_user_query="How much is UltraWasher 8262?",
            current_product="UltraWasher 8262",
            current_brand="UltraWasher",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = "RoboCleaner 3120"

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "How much is UltraWasher 8262?"
        request.last_response = "UltraWasher 8262 costs $333.00"
        request.intent = "price_check"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # price_check strategy: warranty on current + compare alternatives
        all_text = " ".join(s.lower() for s in suggestions)
        assert "warranty" in all_text or "compare" in all_text


# ══════════════════════════════════════════════════════════════════
# Edge Cases
# ══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases: empty catalog, single brand, unknown intent, deterministic ordering."""

    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_followup_empty_catalog_no_crash(self, mock_catalog, mock_session, mock_profile, mock_cooccur, mock_alt, servicer, mock_context):
        """GetFollowUpRecommendations must not crash with empty product catalog."""
        mock_catalog.return_value = {
            "brands": {},
            "all_product_names": [],
            "products_by_brand": {},
            "price_min": 50.0,
            "price_max": 500.0,
        }
        mock_session.return_value = _make_session_context()
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = None

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "Tell me about products"
        request.last_response = "We have many products"
        request.intent = "product_inquiry"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) == 3

    @patch("recommendation_service.server._get_premium_showcase_products")
    @patch("recommendation_service.server._get_cross_user_popular_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_start_empty_catalog_no_crash(self, mock_catalog, mock_profile, mock_popular, mock_premium, servicer, mock_context):
        """GetStartRecommendations must not crash with empty product catalog."""
        mock_catalog.return_value = {
            "brands": {},
            "all_product_names": [],
            "products_by_brand": {},
            "price_min": 50.0,
            "price_max": 500.0,
        }
        mock_profile.return_value = _make_customer_profile()
        mock_popular.return_value = []
        mock_premium.return_value = []

        request = MagicMock()
        request.customer_id = ""
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) >= 2  # At least price + warranty defaults

    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_unknown_intent_uses_default_strategy(self, mock_catalog, mock_session, mock_profile, mock_cooccur, mock_alt, servicer, mock_context):
        """Unknown intent falls through to DEFAULT_STRATEGY."""
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            products_mentioned={"RoboCleaner 3120"},
            brands_mentioned={"RoboCleaner"},
            current_product="RoboCleaner 3120",
            current_brand="RoboCleaner",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = "PowerDrill 5641"

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "Some unknown query type"
        request.last_response = "The RoboCleaner 3120 is interesting."
        request.intent = "totally_unknown_intent"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) == 3
        # Should still reference current product
        assert any("RoboCleaner 3120" in s for s in suggestions)

    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_no_product_no_brand_graceful(self, mock_catalog, mock_session, mock_profile, mock_cooccur, mock_alt, servicer, mock_context):
        """No product or brand match → still returns 3 suggestions via generics."""
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context()
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = None

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "What's the weather?"
        request.last_response = "I can only help with products."
        request.intent = "general_question"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) == 3

    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_single_brand_catalog(self, mock_catalog, mock_session, mock_profile, mock_cooccur, mock_alt, servicer, mock_context):
        """Single-brand catalog produces valid suggestions."""
        single_brand_catalog = {
            "brands": {"OnlyBrand": {"count": 2, "price_min": 100, "price_max": 200, "warranties": [12]}},
            "all_product_names": ["OnlyBrand 1000", "OnlyBrand 2000"],
            "products_by_brand": {"OnlyBrand": [
                {"name": "OnlyBrand 1000", "price": 100, "warranty": 12},
                {"name": "OnlyBrand 2000", "price": 200, "warranty": 12},
            ]},
            "price_min": 100.0,
            "price_max": 200.0,
        }
        mock_catalog.return_value = single_brand_catalog
        mock_session.return_value = _make_session_context(
            products_mentioned={"OnlyBrand 1000"},
            brands_mentioned={"OnlyBrand"},
            current_product="OnlyBrand 1000",
            current_brand="OnlyBrand",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = []
        mock_alt.return_value = None

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "Tell me about OnlyBrand 1000"
        request.last_response = "OnlyBrand 1000 costs $100."
        request.intent = "product_inquiry"
        request.customer_id = ""

        response = servicer.GetFollowUpRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert len(suggestions) == 3
        # No duplicate suggestions
        assert len(suggestions) == len(set(s.lower() for s in suggestions))

    @patch("recommendation_service.server._find_price_alternative")
    @patch("recommendation_service.server._get_cooccurring_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._build_session_context")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_deterministic_ordering(self, mock_catalog, mock_session, mock_profile, mock_cooccur, mock_alt, servicer, mock_context):
        """Multiple calls with same input produce same output."""
        mock_catalog.return_value = _make_catalog()
        mock_session.return_value = _make_session_context(
            intents_used={"product_inquiry"},
            products_mentioned={"PowerDrill 5641", "UltraWasher 8262"},
            brands_mentioned={"PowerDrill", "UltraWasher"},
            last_intent="product_inquiry",
            current_product="PowerDrill 5641",
            current_brand="PowerDrill",
        )
        mock_profile.return_value = _make_customer_profile()
        mock_cooccur.return_value = ["RoboCleaner 3120"]
        mock_alt.return_value = None

        request = MagicMock()
        request.session_id = "sess-1"
        request.last_query = "Tell me about PowerDrill 5641"
        request.last_response = "PowerDrill 5641 is a drill."
        request.intent = "product_inquiry"
        request.customer_id = ""

        results = set()
        for _ in range(10):
            response = servicer.GetFollowUpRecommendations(request, mock_context)
            results.add(tuple(response.suggestions))

        assert len(results) == 1

    @patch("recommendation_service.server._get_premium_showcase_products")
    @patch("recommendation_service.server._get_cross_user_popular_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_tier1c_product_inquiry_as_only_topic(self, mock_catalog, mock_profile, mock_popular, mock_premium, servicer, mock_context):
        """Tier 1C: returning user with product inquiry history."""
        mock_catalog.return_value = _make_catalog()
        mock_profile.return_value = _make_customer_profile(
            has_history=True,
            topics_explored=["UltraWasher 8262"],
            brands_explored={"UltraWasher"},
            intents_history={"comparison", "warranty_question", "price_check"},
        )
        mock_popular.return_value = []
        mock_premium.return_value = []

        request = MagicMock()
        request.customer_id = "cust-1"
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        assert any("ultrawasher 8262" in s.lower() for s in suggestions)

    @patch("recommendation_service.server._get_premium_showcase_products")
    @patch("recommendation_service.server._get_cross_user_popular_products")
    @patch("recommendation_service.server._build_customer_profile")
    @patch("recommendation_service.server._get_product_catalog_summary")
    def test_tier1c_empty_brands_no_crash(self, mock_catalog, mock_profile, mock_popular, mock_premium, servicer, mock_context):
        """Tier 1C: warranty suggestion with empty brands_list doesn't crash."""
        mock_catalog.return_value = {
            "brands": {},
            "all_product_names": [],
            "products_by_brand": {},
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
        mock_premium.return_value = []

        request = MagicMock()
        request.customer_id = "cust-1"
        request.session_id = "sess-1"

        response = servicer.GetStartRecommendations(request, mock_context)
        suggestions = list(response.suggestions)

        # Should not crash, should return some suggestions
        assert len(suggestions) >= 2
