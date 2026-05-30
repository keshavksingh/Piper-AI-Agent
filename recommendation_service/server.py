"""Recommendation Service — Context-aware, memory-aware query suggestions."""

import json
import sys
import threading
import time
import grpc
from concurrent import futures

import psycopg2
import psycopg2.extras
import psycopg2.pool
from google.protobuf import json_format

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

ALL_INTENTS = {
    "product_inquiry", "price_check", "warranty_question",
    "comparison", "follow_up", "session_query",
}
ALL_TOOLS = {"product_search", "price_lookup", "warranty_check", "product_compare"}


# ── Intent Strategy System ────────────────────────────────────────

INTENT_STRATEGY = {
    "product_inquiry": [
        "What's the warranty on {current_product}?",
        "How much does {current_product} cost?",
    ],
    "price_check": [
        "What's the warranty on {current_product}?",
        "Compare {current_product} with alternatives in a similar price range",
    ],
    "warranty_question": [
        "How much does {current_product} cost?",
        "Compare warranty options across brands",
    ],
    "comparison": [
        "Tell me about {current_product} features",
        "How much does {current_product} cost?",
    ],
    "session_query": [
        "Return to {current_product}",
        "Explore a new product category",
    ],
}

DEFAULT_STRATEGY = [
    "Tell me more about {current_product}",
    "What's the warranty on {current_product}?",
]


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
        products_by_brand = {}
        global_min = float("inf")
        global_max = 0.0

        for row in rows:
            name = row["product_name"]
            price = float(row["price"])
            warranty = int(row["warranty_months"])
            all_product_names.append(name)

            # Extract brand (first word of product name)
            parts = name.split() if name else []
            brand = parts[0] if parts else "Unknown"
            if brand not in brands:
                brands[brand] = {"count": 0, "price_min": price, "price_max": price, "warranties": set()}
            brands[brand]["count"] += 1
            brands[brand]["price_min"] = min(brands[brand]["price_min"], price)
            brands[brand]["price_max"] = max(brands[brand]["price_max"], price)
            brands[brand]["warranties"].add(warranty)

            if brand not in products_by_brand:
                products_by_brand[brand] = []
            products_by_brand[brand].append({"name": name, "price": price, "warranty": warranty})

            global_min = min(global_min, price)
            global_max = max(global_max, price)

        # Convert warranty sets to sorted lists
        for brand in brands:
            brands[brand]["warranties"] = sorted(brands[brand]["warranties"])

        catalog = {
            "brands": brands,
            "all_product_names": all_product_names,
            "products_by_brand": products_by_brand,
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
        products_by_brand = {}
        for brand, info in FALLBACK_BRANDS.items():
            products_by_brand[brand] = []
            for i in range(info["count"]):
                pname = f"{brand} {1000 + i}"
                all_names.append(pname)
                products_by_brand[brand].append({
                    "name": pname,
                    "price": info["price_max"],
                    "warranty": max(info["warranties"]) if info["warranties"] else 12,
                })

        return {
            "brands": {k: dict(v) for k, v in FALLBACK_BRANDS.items()},
            "all_product_names": all_names,
            "products_by_brand": products_by_brand,
            "price_min": 50.0,
            "price_max": 500.0,
        }
    finally:
        if conn:
            put_pg_conn(conn)


# ── Current Focus Extraction ──────────────────────────────────────

def _extract_current_focus(last_query, last_response, catalog):
    """Derive current_product and current_brand from the last exchange only.

    Scans response first (stronger signal), then query.
    Picks earliest-position product match for determinism.
    Falls back to brand-only if no product match.
    """
    result = {"current_product": None, "current_brand": None}

    product_names = catalog.get("all_product_names", [])
    brand_names = set(catalog.get("brands", {}).keys())

    if not product_names and not brand_names:
        return result

    # Build lowercase lookup, skip blank names
    product_names_lower = {p.lower(): p for p in product_names if p and p.strip()}

    # Scan response first (stronger signal), then query
    for text in [last_response, last_query]:
        if not text:
            continue
        text_lower = text.lower()

        # Find earliest-position product match
        best_product = None
        best_pos = len(text_lower) + 1
        for pname_lower, pname in product_names_lower.items():
            pos = text_lower.find(pname_lower)
            if pos != -1 and (
                pos < best_pos
                or (pos == best_pos and len(pname_lower) > len((best_product or "").lower()))
            ):
                best_pos = pos
                best_product = pname

        if best_product:
            result["current_product"] = best_product
            result["current_brand"] = best_product.split()[0] if best_product else None
            return result

    # Brand-only fallback: scan response then query
    for text in [last_response, last_query]:
        if not text:
            continue
        text_lower = text.lower()
        best_brand = None
        best_pos = len(text_lower) + 1
        for brand in brand_names:
            pos = text_lower.find(brand.lower())
            if pos != -1 and pos < best_pos:
                best_pos = pos
                best_brand = brand
        if best_brand:
            result["current_brand"] = best_brand
            return result

    return result


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
        "current_product": None,
        "current_brand": None,
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

        # Extract current focus from the last exchange
        focus = _extract_current_focus(
            context["last_user_query"],
            context["last_assistant_response"],
            catalog,
        )
        context["current_product"] = focus["current_product"]
        context["current_brand"] = focus["current_brand"]

        return context

    except Exception as e:
        log.warning("session_context_build_failed", error=str(e))
        return context
    finally:
        if conn:
            put_pg_conn(conn)


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
            memory_pb2.GetEpisodicRequest(customer_id=customer_id, limit=10),
            timeout=Config.GRPC_TIMEOUT_MEMORY,
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

            # Extract intents and tools from metadata (now a Struct)
            if mem.metadata.fields:
                try:
                    meta = json_format.MessageToDict(mem.metadata)
                    if isinstance(meta, dict):
                        if "intents" in meta:
                            for intent in meta["intents"]:
                                profile["intents_history"].add(intent)
                        if "tools_used" in meta:
                            for tool in meta["tools_used"]:
                                profile["tools_used_historically"].add(tool)
                except Exception:
                    pass

        return profile

    except Exception as e:
        log.warning("customer_profile_build_failed", error=str(e))
        return profile


# ── Cross-User Popular Products ───────────────────────────────────

def _get_cross_user_popular_products(catalog, limit=3):
    """Aggregate popular products by counting unique customers per product entity.

    Fetches recent user turns, extracts product names via text matching,
    counts unique customers per product, sorts by count descending.
    """
    conn = None
    try:
        product_names = catalog.get("all_product_names", [])
        if not product_names:
            return []

        conn = get_pg_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT ct.content, s.customer_id "
                "FROM conversation_turns ct "
                "JOIN sessions s ON ct.session_id = s.id "
                "WHERE ct.role = 'user' "
                "ORDER BY ct.created_at DESC LIMIT 1000"
            )
            rows = cur.fetchall()

        if not rows:
            return []

        product_names_lower = {p.lower(): p for p in product_names}
        # Count unique customers per product
        product_customers = {}  # product_name -> set of customer_ids
        for row in rows:
            content = (row["content"] or "").lower()
            customer_id = row["customer_id"] or ""
            for pname_lower, pname in product_names_lower.items():
                if pname_lower in content:
                    if pname not in product_customers:
                        product_customers[pname] = set()
                    product_customers[pname].add(customer_id)

        if not product_customers:
            return []

        # Sort by unique customer count descending, then alphabetically for determinism
        ranked = sorted(
            product_customers.items(),
            key=lambda x: (-len(x[1]), x[0]),
        )

        results = []
        for product_name, customers in ranked[:limit]:
            brand = product_name.split()[0] if product_name else ""
            results.append({
                "product": product_name,
                "brand": brand,
                "customer_count": len(customers),
            })

        return results

    except Exception as e:
        log.warning("cross_user_popular_products_failed", error=str(e))
        return []
    finally:
        if conn:
            put_pg_conn(conn)


# ── Co-occurring Products ─────────────────────────────────────────

def _get_cooccurring_products(current_product, catalog, limit=3):
    """Find products that co-occur with current_product across sessions.

    Two-step SQL:
    1. Find sessions where current_product appears (LIMIT 100)
    2. Get all content from those sessions, count co-occurring product entities
    """
    if not current_product:
        return []

    conn = None
    try:
        product_names = catalog.get("all_product_names", [])
        if not product_names:
            return []

        conn = get_pg_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Step 1: Find sessions mentioning current_product
            # Escape LIKE wildcards in product name to prevent unintended matching
            escaped = current_product.lower().replace("%", "\\%").replace("_", "\\_")
            search_term = f"%{escaped}%"
            cur.execute(
                "SELECT DISTINCT session_id FROM conversation_turns "
                "WHERE LOWER(content) LIKE %s LIMIT 100",
                (search_term,),
            )
            session_rows = cur.fetchall()

            if not session_rows:
                return []

            session_ids = [r["session_id"] for r in session_rows]

            # Step 2: Get all content from those sessions
            cur.execute(
                "SELECT content FROM conversation_turns "
                "WHERE session_id = ANY(%s)",
                (session_ids,),
            )
            content_rows = cur.fetchall()

        product_names_lower = {p.lower(): p for p in product_names}
        current_lower = current_product.lower()
        cooccurrence = {}  # product_name -> count

        for row in content_rows:
            content = (row["content"] or "").lower()
            for pname_lower, pname in product_names_lower.items():
                if pname_lower == current_lower:
                    continue
                if pname_lower in content:
                    cooccurrence[pname] = cooccurrence.get(pname, 0) + 1

        if not cooccurrence:
            return []

        ranked = sorted(
            cooccurrence.items(),
            key=lambda x: (-x[1], x[0]),
        )

        return [product for product, _ in ranked[:limit]]

    except Exception as e:
        log.warning("cooccurring_products_failed", error=str(e))
        return []
    finally:
        if conn:
            put_pg_conn(conn)


# ── Premium Showcase Products ─────────────────────────────────────

def _get_premium_showcase_products(catalog, limit=3):
    """Most expensive product per distinct brand, top N brands by price.

    Uses DB query with DISTINCT ON (brand logic), falls back to catalog summary.
    """
    conn = None
    try:
        conn = get_pg_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT DISTINCT ON (SPLIT_PART(product_name, ' ', 1)) "
                "product_name, price, SPLIT_PART(product_name, ' ', 1) AS brand "
                "FROM products "
                "ORDER BY SPLIT_PART(product_name, ' ', 1), price DESC"
            )
            rows = cur.fetchall()

        if not rows:
            return _premium_showcase_from_catalog(catalog, limit)

        # Sort by price descending, pick top N distinct brands
        sorted_rows = sorted(rows, key=lambda r: float(r["price"]), reverse=True)
        results = []
        seen_brands = set()
        for row in sorted_rows:
            brand = row["brand"]
            if brand in seen_brands:
                continue
            seen_brands.add(brand)
            results.append({
                "product": row["product_name"],
                "brand": brand,
                "price": float(row["price"]),
            })
            if len(results) >= limit:
                break

        return results

    except Exception as e:
        log.warning("premium_showcase_failed", error=str(e))
        return _premium_showcase_from_catalog(catalog, limit)
    finally:
        if conn:
            put_pg_conn(conn)


def _premium_showcase_from_catalog(catalog, limit=3):
    """Fallback: derive premium products from catalog summary brand-level data."""
    products_by_brand = catalog.get("products_by_brand", {})
    brands_info = catalog.get("brands", {})

    results = []

    if products_by_brand:
        # Pick most expensive product from each brand
        for brand, products in products_by_brand.items():
            if not products:
                continue
            top = max(products, key=lambda p: p["price"])
            results.append({
                "product": top["name"],
                "brand": brand,
                "price": top["price"],
            })
    elif brands_info:
        # Use brand-level price_max
        for brand, info in brands_info.items():
            results.append({
                "product": f"{brand} Premium",
                "brand": brand,
                "price": info.get("price_max", 0),
            })

    # Sort by price descending, pick top N distinct brands
    results.sort(key=lambda r: r["price"], reverse=True)
    seen_brands = set()
    filtered = []
    for r in results:
        if r["brand"] in seen_brands:
            continue
        seen_brands.add(r["brand"])
        filtered.append(r)
        if len(filtered) >= limit:
            break

    return filtered


# ── Price Alternative Finder ──────────────────────────────────────

def _find_price_alternative(current_product, current_brand, catalog):
    """Find a product from a different brand at a similar price point (±20%).

    Falls back to first product from a different brand in catalog.
    """
    if not current_product:
        return None

    conn = None
    try:
        conn = get_pg_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Get current product's price
            cur.execute(
                "SELECT price FROM products WHERE product_name = %s",
                (current_product,),
            )
            row = cur.fetchone()

            if not row:
                return _price_alternative_from_catalog(current_product, current_brand, catalog)

            price = float(row["price"])
            low = price * 0.8
            high = price * 1.2

            # Find products in ±20% from different brands
            if current_brand:
                cur.execute(
                    "SELECT product_name, price FROM products "
                    "WHERE price BETWEEN %s AND %s "
                    "AND NOT product_name LIKE %s "
                    "ORDER BY ABS(price - %s) ASC LIMIT 1",
                    (low, high, f"{current_brand}%", price),
                )
            else:
                # No brand to exclude — just find nearest price match that isn't the same product
                cur.execute(
                    "SELECT product_name, price FROM products "
                    "WHERE price BETWEEN %s AND %s "
                    "AND product_name != %s "
                    "ORDER BY ABS(price - %s) ASC LIMIT 1",
                    (low, high, current_product, price),
                )
            alt_row = cur.fetchone()

            if alt_row:
                return alt_row["product_name"]

    except Exception as e:
        log.warning("price_alternative_failed", error=str(e))

    finally:
        if conn:
            put_pg_conn(conn)

    return _price_alternative_from_catalog(current_product, current_brand, catalog)


def _price_alternative_from_catalog(current_product, current_brand, catalog):
    """Fallback: find first product from a different brand in catalog."""
    for pname in catalog.get("all_product_names", []):
        pbrand = pname.split()[0] if pname else ""
        if pbrand and pbrand != current_brand:
            return pname
    return None


# ── Focus-Anchored Suggestion Builder ─────────────────────────────

def _build_focus_anchored_suggestions(current_focus, intent, session_ctx, catalog, profile):
    """Build 3 context-aware suggestions anchored to current focus.

    Slot 1-2: From INTENT_STRATEGY based on effective intent.
    Slot 3: Cross-user co-occurrence or price alternative.
    Filters against episodic memory topics to avoid repeats.
    """
    current_product = current_focus.get("current_product")
    current_brand = current_focus.get("current_brand")
    suggestions = []

    # Resolve effective intent: follow_up chains resolve to last non-follow_up intent
    effective_intent = intent
    if intent == "follow_up":
        last = session_ctx.get("last_intent")
        if last and last != "follow_up":
            effective_intent = last

    # Get strategy templates
    strategy = INTENT_STRATEGY.get(effective_intent, DEFAULT_STRATEGY)

    # Fill slots 1-2 from strategy
    for template in strategy[:2]:
        if current_product:
            filled = template.replace("{current_product}", current_product)
            filled = filled.replace("{current_brand}", current_brand or "")
        elif current_brand:
            # Brand-only: adapt templates
            filled = template.replace("{current_product}", f"{current_brand} products")
            filled = filled.replace("{current_brand}", current_brand)
        else:
            continue
        suggestions.append(filled)

    # Slot 3: Cross-user co-occurrence
    slot3 = None
    if current_product:
        try:
            cooccurring = _get_cooccurring_products(current_product, catalog, limit=3)
            # Filter against session history + episodic memory
            explored_topics = set(t.lower() for t in profile.get("topics_explored", []))
            mentioned = set(p.lower() for p in session_ctx.get("products_mentioned", set()))
            for coprod in cooccurring:
                if coprod.lower() not in explored_topics and coprod.lower() not in mentioned:
                    slot3 = f"Users who asked about {current_product} also explored {coprod}"
                    break
        except Exception:
            pass

    # Slot 3 fallback: price alternative
    if not slot3 and current_product:
        try:
            alt = _find_price_alternative(current_product, current_brand, catalog)
            if alt:
                slot3 = f"Compare {current_product} with {alt}"
        except Exception:
            pass

    if slot3:
        suggestions.append(slot3)

    return suggestions


# ── Generic Catalog-Aware Fallbacks ───────────────────────────────

def _catalog_aware_generics(catalog, exclude=None):
    """Generate generic catalog-aware suggestions for padding."""
    exclude = exclude or set()
    brand_list = list(catalog.get("brands", {}).keys())
    price_mid = int((catalog.get("price_min", 50) + catalog.get("price_max", 500)) / 2)

    generics = []
    if brand_list:
        generics.append(f"Tell me about {brand_list[0]} products")
    generics.append(f"Show me products under ${price_mid}")
    generics.append("Which products come with the longest warranty?")
    if len(brand_list) >= 2:
        generics.append(f"Compare {brand_list[0]} with {brand_list[1]}")
    if len(brand_list) >= 3:
        generics.append(f"Show me {brand_list[2]} products")

    if len(generics) < 3:
        extras = [
            "What products do you have?",
            f"Show me products under ${price_mid}",
            "Which products come with the longest warranty?",
        ]
        for e in extras:
            if e not in generics:
                generics.append(e)
            if len(generics) >= 5:
                break

    exclude_lower = {e.lower() for e in exclude}
    return [g for g in generics if g.lower() not in exclude_lower]


# ── gRPC Service Implementation ──────────────────────────────────

class RecommendationServiceServicer(pb2_grpc.RecommendationServiceServicer):

    def GetStartRecommendations(self, request, context):
        """3-tier cold start: 1C (returning user) → 1A (popular products) → 1B (premium showcase) → generic."""
        customer_id = request.customer_id
        session_id = request.session_id

        suggestions = []
        catalog = _get_product_catalog_summary()

        # ── Tier 1C: Returning user with episodic memories ──
        personal_suggestion = None
        if customer_id:
            try:
                profile = _build_customer_profile(customer_id)
                if profile["has_history"] and profile["topics_explored"]:
                    last_topic = profile["topics_explored"][0]
                    personal_suggestion = f"Continue where you left off: {last_topic}"
                    suggestions.append(personal_suggestion)
                    log.info("start_recommendations_tier1c", customer_id=customer_id)
            except Exception as e:
                log.warning("tier1c_failed", error=str(e))

        # ── Tier 1A: Cross-user popular products ──
        if len(suggestions) < 3:
            try:
                popular = _get_cross_user_popular_products(catalog, limit=3)
                seen_lower = {s.lower() for s in suggestions}
                for item in popular:
                    product = item["product"]
                    brand = item["brand"]
                    suggestion = f"Tell me about {product} — our most popular {brand}"
                    if suggestion.lower() not in seen_lower:
                        suggestions.append(suggestion)
                        seen_lower.add(suggestion.lower())
                    if len(suggestions) >= 3:
                        break
                if popular:
                    log.info("start_recommendations_tier1a", count=len(popular))
            except Exception as e:
                log.warning("tier1a_failed", error=str(e))

        # ── Tier 1B: Premium showcase ──
        if len(suggestions) < 3:
            try:
                premium = _get_premium_showcase_products(catalog, limit=3)
                seen_lower = {s.lower() for s in suggestions}
                for item in premium:
                    product = item["product"]
                    brand = item["brand"]
                    price = int(item["price"])
                    suggestion = f"Check out {product} (${price}) — our premium {brand}"
                    if suggestion.lower() not in seen_lower:
                        suggestions.append(suggestion)
                        seen_lower.add(suggestion.lower())
                    if len(suggestions) >= 3:
                        break
                if premium:
                    log.info("start_recommendations_tier1b", count=len(premium))
            except Exception as e:
                log.warning("tier1b_failed", error=str(e))

        # ── Generic catalog-aware fallback ──
        if len(suggestions) < 3:
            generics = _catalog_aware_generics(catalog, exclude=set(suggestions))
            for g in generics:
                if g not in suggestions:
                    suggestions.append(g)
                if len(suggestions) >= 5:
                    break
            log.info("start_recommendations_generic", count=len(suggestions))

        log.info("start_recommendations", count=len(suggestions), customer_id=customer_id)
        return pb2.RecommendationResponse(suggestions=suggestions[:5])

    def GetFollowUpRecommendations(self, request, context):
        """Context-aware follow-up: extract focus → intent strategy → cross-user slot 3 → dedup → pad."""
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
            session_ctx["intents_used"].add(intent)
            # Preserve previous intent for follow_up resolution;
            # only overwrite last_intent when the current turn is NOT a follow_up.
            if intent != "follow_up":
                session_ctx["last_intent"] = intent
        if last_response:
            session_ctx["last_assistant_response"] = last_response

        # 2. Extract current focus from last exchange
        focus = _extract_current_focus(last_query, last_response, catalog)
        session_ctx["current_product"] = focus["current_product"]
        session_ctx["current_brand"] = focus["current_brand"]

        # Also update products_mentioned / brands_mentioned from last exchange
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

        # 3. Resolve customer_id if not provided
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
                    put_pg_conn(conn)
            except Exception as e:
                log.warning("customer_id_lookup_failed", error=str(e))

        # 4. Build customer profile
        profile = _build_customer_profile(customer_id)

        # 5. Build focus-anchored suggestions
        suggestions = _build_focus_anchored_suggestions(
            focus, intent, session_ctx, catalog, profile,
        )

        # 6. Deduplicate, exclude echo of last query
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

        # 7. Pad to 3 from catalog-aware generics if needed
        if len(suggestions) < 3:
            generics = _catalog_aware_generics(catalog, exclude=set(suggestions))
            for g in generics:
                g_lower = g.strip().lower()
                if g_lower not in seen and g_lower != last_q_lower:
                    suggestions.append(g)
                    seen.add(g_lower)
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
