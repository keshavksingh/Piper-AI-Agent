"""Recommendation Service — Context-aware, template-based query suggestions."""

import json
import sys
import threading
import time
import grpc
from concurrent import futures

import psycopg2
import psycopg2.extras

sys.path.append("..")

import protos.recommendation_service_pb2 as pb2
import protos.recommendation_service_pb2_grpc as pb2_grpc
import protos.memory_service_pb2 as memory_pb2
import protos.memory_service_pb2_grpc as memory_pb2_grpc

from shared.config import Config
from shared.logging_config import setup_logging
from shared.resilience import create_grpc_channel

log = setup_logging("recommendation_service")


# ── Connections ───────────────────────────────────────────────────

def get_pg_conn():
    return psycopg2.connect(Config.DATABASE_URL)


def get_memory_stub():
    channel = create_grpc_channel(Config.MEMORY_SERVICE_ADDR)
    return memory_pb2_grpc.MemoryServiceStub(channel)


# ── Hardcoded Brand Fallback ─────────────────────────────────────

FALLBACK_BRANDS = {
    "UltraWasher": {"count": 5, "price_min": 121.0, "price_max": 333.0, "warranties": [6, 12, 24, 36]},
    "RoboCleaner": {"count": 7, "price_min": 129.0, "price_max": 499.0, "warranties": [6, 12, 18, 24]},
    "AirPurifier": {"count": 6, "price_min": 56.0, "price_max": 418.0, "warranties": [12, 18, 24, 36]},
    "NoiseCanceller": {"count": 5, "price_min": 56.0, "price_max": 407.0, "warranties": [6, 24, 36]},
    "MegaBlender": {"count": 8, "price_min": 61.0, "price_max": 336.0, "warranties": [6, 12, 18, 24, 36]},
    "EcoKettle": {"count": 5, "price_min": 87.0, "price_max": 448.0, "warranties": [6, 12]},
    "PowerDrill": {"count": 10, "price_min": 54.0, "price_max": 486.0, "warranties": [6, 12, 24, 36]},
    "SuperVac": {"count": 2, "price_min": 63.0, "price_max": 484.0, "warranties": [12, 18]},
    "ThermoBottle": {"count": 3, "price_min": 126.0, "price_max": 388.0, "warranties": [6, 12, 18]},
    "SmartLamp": {"count": 3, "price_min": 250.0, "price_max": 441.0, "warranties": [6, 18, 24]},
}


# ── All Known Intents & Tools ────────────────────────────────────

ALL_INTENTS = {"product_inquiry", "price_check", "warranty_question", "comparison"}
ALL_TOOLS = {"product_search", "price_lookup", "warranty_check", "product_compare"}

# Priority order for suggesting missing intents (most valuable first)
INTENT_PRIORITY = ["comparison", "warranty_question", "price_check", "product_inquiry"]


# ── Product Catalog Cache ────────────────────────────────────────

_catalog_cache = {"data": None, "timestamp": 0}
_catalog_lock = threading.Lock()
CATALOG_TTL = 300  # 5 minutes


def _get_product_catalog_summary():
    """Thread-safe, TTL-cached query of the products table."""
    now = time.time()
    with _catalog_lock:
        if _catalog_cache["data"] and (now - _catalog_cache["timestamp"]) < CATALOG_TTL:
            return _catalog_cache["data"]

    conn = None
    try:
        conn = get_pg_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT product_name, price, warranty_months FROM products ORDER BY product_name"
            )
            rows = cur.fetchall()

        brands = {}
        all_product_names = []
        global_min = float("inf")
        global_max = 0.0

        for row in rows:
            name = row["product_name"]
            price = float(row["price"])
            warranty = int(row["warranty_months"])
            all_product_names.append(name)

            # Extract brand (first word of product name)
            brand = name.split()[0] if name else "Unknown"
            if brand not in brands:
                brands[brand] = {"count": 0, "price_min": price, "price_max": price, "warranties": set()}
            brands[brand]["count"] += 1
            brands[brand]["price_min"] = min(brands[brand]["price_min"], price)
            brands[brand]["price_max"] = max(brands[brand]["price_max"], price)
            brands[brand]["warranties"].add(warranty)

            global_min = min(global_min, price)
            global_max = max(global_max, price)

        # Convert warranty sets to sorted lists
        for brand in brands:
            brands[brand]["warranties"] = sorted(brands[brand]["warranties"])

        catalog = {
            "brands": brands,
            "all_product_names": all_product_names,
            "price_min": global_min if global_min != float("inf") else 50.0,
            "price_max": global_max if global_max > 0 else 500.0,
        }

        with _catalog_lock:
            _catalog_cache["data"] = catalog
            _catalog_cache["timestamp"] = time.time()

        return catalog

    except Exception as e:
        log.warning("catalog_fetch_failed", error=str(e))
        # Return fallback from hardcoded brand data
        all_names = []
        for brand, info in FALLBACK_BRANDS.items():
            for i in range(info["count"]):
                all_names.append(f"{brand} {1000 + i}")

        return {
            "brands": {k: dict(v) for k, v in FALLBACK_BRANDS.items()},
            "all_product_names": all_names,
            "price_min": 50.0,
            "price_max": 500.0,
        }
    finally:
        if conn:
            conn.close()


# ── Session Context Builder ──────────────────────────────────────

def _build_session_context(session_id):
    """Query conversation_turns for the current session and extract context."""
    context = {
        "intents_used": set(),
        "products_mentioned": set(),
        "brands_mentioned": set(),
        "tools_used": set(),
        "last_intent": None,
        "last_user_query": None,
        "last_assistant_response": None,
    }

    if not session_id:
        return context

    conn = None
    try:
        catalog = _get_product_catalog_summary()
        conn = get_pg_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT role, content, intent, tool_calls "
                "FROM conversation_turns "
                "WHERE session_id = %s "
                "ORDER BY created_at ASC",
                (session_id,),
            )
            rows = cur.fetchall()

        brand_names = set(catalog["brands"].keys())
        product_names_lower = {p.lower(): p for p in catalog["all_product_names"]}

        for row in rows:
            role = row["role"]
            content = row["content"] or ""
            intent = row["intent"]
            tool_calls = row["tool_calls"]

            if intent:
                context["intents_used"].add(intent)

            if role == "user":
                context["last_user_query"] = content
                context["last_intent"] = intent
            elif role == "assistant":
                context["last_assistant_response"] = content

            # Match product names in content
            content_lower = content.lower()
            for pname_lower, pname in product_names_lower.items():
                if pname_lower in content_lower:
                    context["products_mentioned"].add(pname)

            # Match brand names in content
            for brand in brand_names:
                if brand.lower() in content_lower:
                    context["brands_mentioned"].add(brand)

            # Extract tools from tool_calls JSONB
            if tool_calls:
                calls = tool_calls if isinstance(tool_calls, list) else []
                if isinstance(tool_calls, str):
                    try:
                        calls = json.loads(tool_calls)
                    except (json.JSONDecodeError, TypeError):
                        calls = []
                for call in calls:
                    if isinstance(call, dict) and "tool" in call:
                        context["tools_used"].add(call["tool"])

        return context

    except Exception as e:
        log.warning("session_context_build_failed", error=str(e))
        return context
    finally:
        if conn:
            conn.close()


# ── Customer Profile Builder ─────────────────────────────────────

def _build_customer_profile(customer_id):
    """Query episodic memories via memory service gRPC to build customer profile."""
    profile = {
        "brands_explored": set(),
        "intents_history": set(),
        "topics_explored": [],
        "tools_used_historically": set(),
        "has_history": False,
    }

    if not customer_id:
        return profile

    try:
        memory_stub = get_memory_stub()
        episodic_resp = memory_stub.GetEpisodicMemories(
            memory_pb2.GetEpisodicRequest(customer_id=customer_id, limit=10)
        )

        if not episodic_resp.memories:
            return profile

        profile["has_history"] = True
        catalog = _get_product_catalog_summary()
        brand_names = set(catalog["brands"].keys())

        for mem in episodic_resp.memories:
            # Extract brands from key_topics
            for topic in mem.key_topics:
                if topic not in profile["topics_explored"]:
                    profile["topics_explored"].append(topic)
                for brand in brand_names:
                    if brand.lower() in topic.lower():
                        profile["brands_explored"].add(brand)

            # Extract intents and tools from metadata
            if mem.metadata:
                try:
                    meta = json.loads(mem.metadata) if isinstance(mem.metadata, str) else mem.metadata
                    if isinstance(meta, dict):
                        if "intents" in meta:
                            for intent in meta["intents"]:
                                profile["intents_history"].add(intent)
                        if "tools_used" in meta:
                            for tool in meta["tools_used"]:
                                profile["tools_used_historically"].add(tool)
                except (json.JSONDecodeError, TypeError):
                    pass

        return profile

    except Exception as e:
        log.warning("customer_profile_build_failed", error=str(e))
        return profile


# ── Gap Analysis Engine ──────────────────────────────────────────

def _gap_analysis(session_context, customer_profile):
    """Compare what user HAS done vs what they COULD do."""
    used_intents = session_context["intents_used"] | customer_profile.get("intents_history", set())
    used_brands = session_context["brands_mentioned"] | customer_profile.get("brands_explored", set())
    used_tools = session_context["tools_used"] | customer_profile.get("tools_used_historically", set())

    catalog = _get_product_catalog_summary()
    all_brands = set(catalog["brands"].keys())

    # Missing intents in priority order
    missing_intents = [i for i in INTENT_PRIORITY if i not in used_intents]

    # Unexplored brands sorted by product count (descending)
    unexplored_brands = sorted(
        [b for b in all_brands if b not in used_brands],
        key=lambda b: catalog["brands"].get(b, {}).get("count", 0),
        reverse=True,
    )

    # Missing tools
    missing_tools = ALL_TOOLS - used_tools

    # Suggested price range (midpoint of global range)
    price_mid = int((catalog["price_min"] + catalog["price_max"]) / 2)

    # Suggested warranty brand (brand with longest warranty not yet explored)
    suggested_warranty_brand = None
    for brand in unexplored_brands:
        warranties = catalog["brands"].get(brand, {}).get("warranties", [])
        if warranties and max(warranties) >= 24:
            suggested_warranty_brand = brand
            break
    if not suggested_warranty_brand and unexplored_brands:
        suggested_warranty_brand = unexplored_brands[0]

    return {
        "missing_intents": missing_intents,
        "unexplored_brands": unexplored_brands,
        "missing_tools": missing_tools,
        "suggested_price_range": price_mid,
        "suggested_warranty_brand": suggested_warranty_brand,
    }


# ── Template System ──────────────────────────────────────────────

TEMPLATES = {
    ("product_inquiry", "same_product"): [
        "What's the warranty on {product_name}?",
        "How much does {product_name} cost?",
        "Compare {product_name} with similar products",
    ],
    ("product_inquiry", "new_brand"): [
        "Show me {new_brand} products",
        "What's the most popular {new_brand} product?",
        "Compare {brand} with {new_brand}",
    ],
    ("price_check", "budget_explore"): [
        "Show me products under ${price_threshold}",
        "What's the cheapest {new_brand} product?",
        "Which brand has the best value?",
    ],
    ("price_check", "same_product"): [
        "What's the warranty on {product_name}?",
        "Compare {product_name} with cheaper alternatives",
        "Show me similar products in a different price range",
    ],
    ("warranty_question", "same_product"): [
        "How much does {product_name} cost?",
        "Compare {product_name} with other products",
        "Show me {new_brand} products with longer warranties",
    ],
    ("warranty_question", "explore"): [
        "Which {new_brand} product has the longest warranty?",
        "Show me products with at least 24 months warranty",
        "Compare warranty options across brands",
    ],
    ("comparison", "expand"): [
        "Compare {new_brand} products",
        "Which brand has the best warranty?",
        "Show me the top-rated products under ${price_threshold}",
    ],
    ("comparison", "same_product"): [
        "What's the warranty on {product_name}?",
        "How much does {product_name} cost?",
        "Show me more {brand} products",
    ],
    ("general_question", "explore"): [
        "Show me {new_brand} products",
        "Which product has the longest warranty?",
        "What's the cheapest product available?",
    ],
    ("general_question", "same_product"): [
        "Tell me more about {product_name}",
        "What's the warranty on {product_name}?",
        "How much does {product_name} cost?",
    ],
}


def _select_template_set(intent, context, gaps):
    """Pick the best template set based on current state."""
    intent = intent or "general_question"
    if intent not in {k[0] for k in TEMPLATES}:
        intent = "general_question"

    # If products were mentioned in context, use same-product templates
    if context["products_mentioned"]:
        key = (intent, "same_product")
        if key in TEMPLATES:
            return key

    # If there are unexplored brands, use new-brand or explore templates
    if gaps["unexplored_brands"]:
        for gap_type in ["new_brand", "budget_explore", "expand", "explore"]:
            key = (intent, gap_type)
            if key in TEMPLATES:
                return key

    # Fallback to explore
    key = (intent, "explore")
    if key in TEMPLATES:
        return key

    # Absolute fallback
    return ("general_question", "explore")


def _fill_templates(templates, context, catalog, gaps):
    """Fill template placeholders with real data."""
    filled = []
    # Sort for deterministic ordering
    products = sorted(context["products_mentioned"])
    brands = sorted(context["brands_mentioned"])
    unexplored = gaps.get("unexplored_brands", [])
    price_threshold = gaps.get("suggested_price_range", 200)

    product_name = products[0] if products else None
    brand = brands[0] if brands else None
    new_brand = unexplored[0] if unexplored else None

    # If no product_name, try to pick one from catalog matching brand
    if not product_name and brand:
        for pname in catalog["all_product_names"]:
            if pname.startswith(brand):
                product_name = pname
                break

    # If still no product_name, pick first from catalog
    if not product_name and catalog["all_product_names"]:
        product_name = catalog["all_product_names"][0]

    # If no brand, extract from product_name
    if not brand and product_name:
        brand = product_name.split()[0]

    # If no new_brand, pick a different brand from catalog
    if not new_brand:
        for b in sorted(catalog["brands"].keys()):
            if b != brand:
                new_brand = b
                break
        if not new_brand:
            new_brand = brand or "PowerDrill"

    # Guard: ensure brand != new_brand to avoid self-comparison
    if brand and new_brand and brand == new_brand:
        for b in sorted(catalog["brands"].keys()):
            if b != brand:
                new_brand = b
                break

    # Track whether brand and new_brand are still identical (single-brand catalog)
    same_brand = brand and new_brand and brand == new_brand

    for template in templates:
        # Skip templates that use both {brand} and {new_brand} when they resolve identically
        if same_brand and "{brand}" in template and "{new_brand}" in template:
            continue
        try:
            result = template.format(
                product_name=product_name or "this product",
                brand=brand or new_brand,
                new_brand=new_brand,
                price_threshold=price_threshold,
            )
            filled.append(result)
        except (KeyError, IndexError):
            filled.append(template)

    return filled


# ── Cross-User Popular Queries ───────────────────────────────────

def _get_cross_user_popular_queries(limit=5):
    """Aggregate popular queries from conversation_turns."""
    conn = None
    try:
        conn = get_pg_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT content, intent, COUNT(*) as freq "
                "FROM conversation_turns "
                "WHERE role = 'user' AND intent IS NOT NULL "
                "AND LENGTH(content) BETWEEN 10 AND 200 "
                "GROUP BY content, intent "
                "ORDER BY freq DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [row["content"] for row in rows]
    except Exception as e:
        log.warning("cross_user_queries_failed", error=str(e))
        return []
    finally:
        if conn:
            conn.close()


# ── Gap-Based Fallback Suggestions ───────────────────────────────

def _gap_fallbacks(gaps, catalog):
    """Generate fallback suggestions from gap analysis."""
    suggestions = []
    unexplored = gaps.get("unexplored_brands", [])
    missing_intents = gaps.get("missing_intents", [])
    price_mid = gaps.get("suggested_price_range", 200)
    brand_keys = list(catalog.get("brands", {}).keys())

    if not brand_keys and not unexplored:
        # No brand data at all — return generic suggestions
        suggestions.append(f"Show me products under ${price_mid}")
        suggestions.append("Which products come with the longest warranty?")
        suggestions.append("What products do you have?")
        return suggestions

    for intent in missing_intents:
        brand = unexplored[0] if unexplored else brand_keys[0]
        if intent == "comparison" and len(brand_keys) >= 2:
            suggestions.append(f"Compare {brand_keys[0]} with {brand_keys[1]}")
        elif intent == "warranty_question":
            suggestions.append(f"What warranty does {brand} offer?")
        elif intent == "price_check":
            suggestions.append(f"Show me products under ${price_mid}")
        elif intent == "product_inquiry":
            suggestions.append(f"Tell me about {brand} products")
        if len(suggestions) >= 3:
            break

    for brand in unexplored[:2]:
        if len(suggestions) >= 3:
            break
        s = f"Show me {brand} products"
        if s not in suggestions:
            suggestions.append(s)

    return suggestions


# ── gRPC Service Implementation ──────────────────────────────────

class RecommendationServiceServicer(pb2_grpc.RecommendationServiceServicer):

    def GetStartRecommendations(self, request, context):
        """3-tier cold start: returning user > cross-user popular > catalog-aware defaults."""
        customer_id = request.customer_id
        session_id = request.session_id

        suggestions = []
        catalog = _get_product_catalog_summary()

        # ── Tier 1: Returning user with episodic memories ──
        if customer_id:
            try:
                profile = _build_customer_profile(customer_id)
                if profile["has_history"]:
                    gaps = _gap_analysis(
                        {
                            "intents_used": set(),
                            "brands_mentioned": set(),
                            "tools_used": set(),
                            "products_mentioned": set(),
                        },
                        profile,
                    )

                    # Suggestion 1: Continue from last topic
                    if profile["topics_explored"]:
                        last_topic = profile["topics_explored"][0]
                        suggestions.append(f"Show me the latest on {last_topic}")

                    # Suggestion 2: Unexplored brand with price range
                    unexplored = gaps["unexplored_brands"]
                    if unexplored:
                        brand = unexplored[0]
                        info = catalog["brands"].get(brand, {})
                        p_min = int(info.get("price_min", 50))
                        p_max = int(info.get("price_max", 500))
                        suggestions.append(
                            f"Explore {brand} products (${p_min}-${p_max})"
                        )

                    # Suggestion 3: Missing intent with real data
                    if gaps["missing_intents"]:
                        top_intent = gaps["missing_intents"][0]
                        brands_list = list(catalog["brands"].keys())
                        if top_intent == "comparison":
                            if len(brands_list) >= 2:
                                suggestions.append(
                                    f"Compare {brands_list[0]} with {brands_list[1]}"
                                )
                        elif top_intent == "warranty_question":
                            wb = gaps["suggested_warranty_brand"] or (brands_list[0] if brands_list else None)
                            if wb:
                                suggestions.append(
                                    f"What warranty does {wb} offer?"
                                )
                            else:
                                suggestions.append(
                                    "Which products come with the longest warranty?"
                                )
                        elif top_intent == "price_check":
                            suggestions.append(
                                f"Show me products under ${gaps['suggested_price_range']}"
                            )
                        elif top_intent == "product_inquiry":
                            if unexplored:
                                suggestions.append(
                                    f"Tell me about {unexplored[0]} products"
                                )
                            elif brands_list:
                                suggestions.append(
                                    f"Tell me about {brands_list[0]} products"
                                )

                    if suggestions:
                        log.info("start_recommendations_tier1", count=len(suggestions), customer_id=customer_id)
            except Exception as e:
                log.warning("tier1_failed", error=str(e))

        # ── Tier 2: Cross-user popular queries ──
        if len(suggestions) < 3:
            try:
                popular = _get_cross_user_popular_queries(limit=5)
                for q in popular:
                    if q not in suggestions:
                        suggestions.append(q)
                    if len(suggestions) >= 5:
                        break
                if popular:
                    log.info("start_recommendations_tier2", count=len(popular))
            except Exception as e:
                log.warning("tier2_failed", error=str(e))

        # ── Tier 3: Catalog-aware defaults ──
        if len(suggestions) < 3:
            brand_names = list(catalog["brands"].keys())
            price_mid = int((catalog["price_min"] + catalog["price_max"]) / 2)

            defaults = []
            if brand_names:
                defaults.append(f"Tell me about {brand_names[0]} products")
            defaults.append(f"Show me products under ${price_mid}")
            defaults.append("Which products come with the longest warranty?")
            if len(brand_names) >= 2:
                defaults.append(f"Compare {brand_names[0]} with {brand_names[1]}")
            if len(brand_names) >= 3:
                defaults.append(f"Show me {brand_names[2]} products")

            for d in defaults:
                if d not in suggestions:
                    suggestions.append(d)
                if len(suggestions) >= 5:
                    break

            log.info("start_recommendations_tier3", count=len(suggestions))

        log.info("start_recommendations", count=len(suggestions), customer_id=customer_id)
        return pb2.RecommendationResponse(suggestions=suggestions[:5])

    def GetFollowUpRecommendations(self, request, context):
        """Context-aware follow-up suggestions using gap analysis and templates."""
        session_id = request.session_id
        last_query = request.last_query
        last_response = request.last_response
        intent = request.intent
        customer_id = getattr(request, "customer_id", "") or ""

        catalog = _get_product_catalog_summary()

        # 1. Build session context
        session_ctx = _build_session_context(session_id)

        # Override last_* from request if available (more current than DB)
        if last_query:
            session_ctx["last_user_query"] = last_query
        if intent:
            session_ctx["last_intent"] = intent
            session_ctx["intents_used"].add(intent)
        if last_response:
            session_ctx["last_assistant_response"] = last_response

        # Extract product/brand mentions from last_query and last_response
        brand_names = set(catalog["brands"].keys())
        product_names_lower = {p.lower(): p for p in catalog["all_product_names"]}

        for text in [last_query, last_response]:
            if not text:
                continue
            text_lower = text.lower()
            for pname_lower, pname in product_names_lower.items():
                if pname_lower in text_lower:
                    session_ctx["products_mentioned"].add(pname)
            for brand in brand_names:
                if brand.lower() in text_lower:
                    session_ctx["brands_mentioned"].add(brand)

        # 2. Resolve customer_id if not provided
        if not customer_id and session_id:
            try:
                conn = get_pg_conn()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT customer_id FROM sessions WHERE id = %s",
                            (session_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            customer_id = str(row[0])
                finally:
                    conn.close()
            except Exception as e:
                log.warning("customer_id_lookup_failed", error=str(e))

        # 3. Build customer profile
        profile = _build_customer_profile(customer_id)

        # 4. Run gap analysis
        gaps = _gap_analysis(session_ctx, profile)

        # 5. Select template set
        template_key = _select_template_set(intent, session_ctx, gaps)
        templates = TEMPLATES.get(template_key, TEMPLATES[("general_question", "explore")])

        # 6. Fill templates
        suggestions = _fill_templates(templates, session_ctx, catalog, gaps)

        # 7. Deduplicate, exclude echo of last query
        seen = set()
        deduped = []
        last_q_lower = (last_query or "").strip().lower()

        for s in suggestions:
            s_lower = s.strip().lower()
            if s_lower in seen:
                continue
            if s_lower == last_q_lower:
                continue
            seen.add(s_lower)
            deduped.append(s)

        suggestions = deduped

        # 8. Gap-based fallbacks if < 3 suggestions
        if len(suggestions) < 3:
            fallbacks = _gap_fallbacks(gaps, catalog)
            for fb in fallbacks:
                fb_lower = fb.strip().lower()
                if fb_lower not in seen and fb_lower != last_q_lower:
                    suggestions.append(fb)
                    seen.add(fb_lower)
                if len(suggestions) >= 3:
                    break

        # 9. Absolute fallback: catalog-aware generics
        if len(suggestions) < 3:
            brand_list = list(catalog["brands"].keys())
            generic_fallbacks = [
                f"Tell me about {brand_list[0]} products" if brand_list else "What products do you have?",
                f"Show me products under ${int((catalog['price_min'] + catalog['price_max']) / 2)}",
                "Which products come with the longest warranty?",
            ]
            for gf in generic_fallbacks:
                gf_lower = gf.strip().lower()
                if gf_lower not in seen and gf_lower != last_q_lower:
                    suggestions.append(gf)
                    seen.add(gf_lower)
                if len(suggestions) >= 3:
                    break

        log.info(
            "follow_up_recommendations",
            count=len(suggestions[:3]),
            intent=intent,
            session_id=session_id,
        )

        return pb2.RecommendationResponse(suggestions=suggestions[:3])


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
    pb2_grpc.add_RecommendationServiceServicer_to_server(
        RecommendationServiceServicer(), server
    )
    server.add_secure_port("[::]:50057", credentials)
    server.start()
    log.info("server_started", port=50057, tls=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
