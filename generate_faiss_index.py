import os
import json
import faiss
import numpy as np
import pickle
import voyageai
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# ------------------- Configuration -------------------
PRODUCT_DATA_DIR = "./product_data"
FAISS_INDEX_FILE = "./vectorstore/faiss_index.bin"
PRODUCT_MAPPING_FILE = "./vectorstore/product_mapping.pkl"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "voyage-3")

os.makedirs("./vectorstore", exist_ok=True)

vo_client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))

# ------------------- Step 1: Load Product Data -------------------
product_texts = []
product_metadata = []

for filename in os.listdir(PRODUCT_DATA_DIR):
    if filename.endswith(".json"):
        file_path = os.path.join(PRODUCT_DATA_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Create a text to embed
            text = f"""
            Product Name: {data.get('productname')}
            Description: {data.get('productdescription')}
            Price: {data.get('productprice')}
            Warranty (months): {data.get('productwarrantyinmonths')}
            """
            product_texts.append(text.strip())
            product_metadata.append(data)

print(f"Loaded {len(product_texts)} products.")

# ------------------- Step 2: Generate Embeddings -------------------
embeddings = []
batch_size = 10  # To avoid rate limits
print("Generating embeddings via Voyage AI...")

for i in tqdm(range(0, len(product_texts), batch_size)):
    batch_texts = product_texts[i:i+batch_size]

    result = vo_client.embed(batch_texts, model=EMBEDDING_MODEL)
    embeddings.extend(result.embeddings)

embeddings = np.array(embeddings).astype("float32")
print(f"Embeddings shape: {embeddings.shape}")

# ------------------- Step 3: Create FAISS Index -------------------
d = embeddings.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(d)  # L2 (euclidean) distance

print("Building FAISS index...")
index.add(embeddings)

# ------------------- Step 4: Save Index and Mapping -------------------
faiss.write_index(index, FAISS_INDEX_FILE)
print(f"Saved FAISS index to {FAISS_INDEX_FILE}")

with open(PRODUCT_MAPPING_FILE, "wb") as f:
    pickle.dump(product_metadata, f)
print(f"Saved product metadata mapping to {PRODUCT_MAPPING_FILE}")
