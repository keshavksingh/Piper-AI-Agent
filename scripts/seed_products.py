"""Seed script — Load product data from JSON files into PostgreSQL + generate pgvector embeddings."""

import os
import sys
import json
import glob
import time
import psycopg2
import voyageai
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://piper:piper@localhost:5432/piper")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
PRODUCT_DATA_DIR = os.getenv("PRODUCT_DATA_DIR", "./product_data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "voyage-3")
BATCH_SIZE = 10


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def load_products():
    """Load all product JSON files."""
    products = []
    pattern = os.path.join(PRODUCT_DATA_DIR, "product_*.json")
    files = sorted(glob.glob(pattern))

    for filepath in files:
        with open(filepath, "r") as f:
            data = json.load(f)
            products.append(data)

    print(f"Loaded {len(products)} products from {PRODUCT_DATA_DIR}")
    return products


def insert_products(conn, products):
    """Insert products into the products table."""
    with conn.cursor() as cur:
        for p in products:
            cur.execute(
                """
                INSERT INTO products (id, product_name, description, price, manufacturing_date, warranty_months)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    product_name = EXCLUDED.product_name,
                    description = EXCLUDED.description,
                    price = EXCLUDED.price,
                    manufacturing_date = EXCLUDED.manufacturing_date,
                    warranty_months = EXCLUDED.warranty_months
                """,
                (
                    p["productid"],
                    p["productname"],
                    p.get("productdescription", ""),
                    p["productprice"],
                    p.get("productmanufacturingdate"),
                    p.get("productwarrantyinmonths", 0),
                ),
            )
    conn.commit()
    print(f"Inserted {len(products)} products into database")


def generate_and_insert_embeddings(conn, products):
    """Generate embeddings in batches and insert incrementally to survive partial failures."""
    vo_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    # Check which products already have embeddings (for resume support)
    with conn.cursor() as cur:
        cur.execute("SELECT product_id FROM product_embeddings")
        existing = {str(row[0]) for row in cur.fetchall()}

    remaining = [p for p in products if p["productid"] not in existing]
    if not remaining:
        print("All products already have embeddings. Skipping.")
        return

    print(f"  {len(existing)} already embedded, {len(remaining)} remaining")

    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    inserted = 0

    for i in range(0, len(remaining), BATCH_SIZE):
        batch_products = remaining[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        texts = [
            f"{p['productname']}: {p.get('productdescription', '')}. "
            f"Price: ${p['productprice']}. "
            f"Warranty: {p.get('productwarrantyinmonths', 0)} months."
            for p in batch_products
        ]

        print(f"  Batch {batch_num}/{total_batches} ({len(texts)} products)...")

        result = vo_client.embed(texts, model=EMBEDDING_MODEL)

        # Insert this batch immediately
        with conn.cursor() as cur:
            for product, embedding in zip(batch_products, result.embeddings):
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                cur.execute(
                    """
                    INSERT INTO product_embeddings (product_id, embedding)
                    VALUES (%s, %s::vector)
                    ON CONFLICT DO NOTHING
                    """,
                    (product["productid"], embedding_str),
                )
        conn.commit()
        inserted += len(batch_products)
        print(f"    Saved ({inserted}/{len(remaining)} total)")

        # Rate limit: free tier = 3 RPM, so wait 21s between batches
        if i + BATCH_SIZE < len(remaining):
            print(f"    Waiting 21s (free-tier rate limit)...")
            time.sleep(21)

    print(f"Generated and inserted {inserted} embeddings")


def main():
    print("=== Piper Product Seed Script ===\n")

    # Load products
    products = load_products()
    if not products:
        print("No products found. Check PRODUCT_DATA_DIR.")
        sys.exit(1)

    # Connect to database
    print(f"Connecting to database...")
    conn = get_connection()

    # Insert products
    print("Inserting products...")
    insert_products(conn, products)

    # Generate and insert embeddings (incrementally, with resume support)
    if VOYAGE_API_KEY:
        print("Generating embeddings via Voyage AI...")
        generate_and_insert_embeddings(conn, products)
    else:
        print("WARNING: VOYAGE_API_KEY not set. Skipping embedding generation.")
        print("  Run again with VOYAGE_API_KEY to generate embeddings.")

    conn.close()
    print("\nSeed complete!")


if __name__ == "__main__":
    main()
