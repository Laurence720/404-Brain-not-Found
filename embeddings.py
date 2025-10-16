# sync_example.py
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings
import os
import numpy as np
from sklearn.preprocessing import normalize   # pip install scikit-learn

# --------- Configuration (override with env vars) ----------
API_KEY = os.environ.get("WATSONX_APIKEY", "vcnkhjC71t6XUFTIdvvVWNt2OuVg6XMMpUlv8pUtIS-B")
URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")  # example
PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "885fb32b-f956-49ac-b220-4666420fc225")
MODEL_ID = "intfloat/multilingual-e5-large"          # e.g. "intfloat/multilingual-e5-large"
# -------------------------------------------

# Parameter name must be api_key (not apikey).
creds = Credentials(api_key=API_KEY, url=URL)

# Create client (example uses batch/concurrency settings)
embed_client = Embeddings(
    model_id=MODEL_ID,
    credentials=creds,
    project_id=PROJECT_ID,        # If using Credentials, supply project_id or space_id as required
    batch_size=32,
    concurrency_limit=4,
    persistent_connection=True
)

# Simple chunk helper for long documents
def chunk_text(text, max_chars=1000):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start+max_chars]
        chunks.append(chunk)
        start += max_chars
    return chunks

# Example: put three documents into the vector store
docs = [
    "This is the first document. It's short.",
    "Second doc. Contains some financial terms and analysis.",
    "Very long document ... " * 300   # Example that forces chunking
]

# Chunk long docs and prepare the input list while keeping source ids
items_to_embed = []
meta = []   # Track original source (doc index, chunk index)
for i, d in enumerate(docs):
    chunks = chunk_text(d, max_chars=1200)
    for j, c in enumerate(chunks):
        items_to_embed.append(c)
        meta.append((i, j))

# Call the synchronous API (may return raw vectors or a REST-style payload)
try:
    resp = embed_client.embed_documents(items_to_embed)  # or .embed_documents(items_to_embed, params=...)
except Exception as e:
    print("embed_documents failed:", e)
    raise

# Some SDKs return list[list[float]], others wrap vectors under data/embedding
vectors = None
if isinstance(resp, dict):
    # Try to extract embeddings from common response shapes
    if "data" in resp and isinstance(resp["data"], list):
        extracted = []
        for item in resp["data"]:
            # Accept either raw vectors or {'embedding': [...]} dicts
            if isinstance(item, dict) and "embedding" in item:
                extracted.append(item["embedding"])
            else:
                extracted.append(item)
        vectors = np.array(extracted)
    elif "embeddings" in resp and isinstance(resp["embeddings"], list):
        vectors = np.array(resp["embeddings"])
    else:
        # Fallback: coerce entire response into an array (may raise)
        try:
            vectors = np.array(resp)
        except Exception as e:
            print("Unable to parse embed_documents dict structure:", resp)
            raise
elif isinstance(resp, list):
    vectors = np.array(resp)
else:
    # Fallback
    try:
        vectors = np.array(resp)
    except Exception as e:
        print("Unable to convert response to vector array; response type:", type(resp))
        raise

# Common post-processing: L2 normalize for cosine similarity
vectors = normalize(vectors, norm='l2', axis=1)

# Simple in-memory "vector store" example (real projects should persist to FAISS/Milvus/Weaviate/etc.)
vector_store = {
    "vectors": vectors,
    "meta": meta,
    "source_texts": items_to_embed
}

# Similarity search example (cosine similarity)
def retrieve(query, k=3):
    q_vec = embed_client.embed_query(query)  # single query -> vector
    # embed_query may return a dict or list; handle both cases
    if isinstance(q_vec, dict) and "embedding" in q_vec:
        q_vec = q_vec["embedding"]
    q_vec = normalize(np.array([q_vec]), axis=1)[0]
    sims = vectors @ q_vec   # since vectors already L2-normalized, dot = cosine
    top_idx = np.argsort(-sims)[:k]
    return [(i, float(sims[i]), meta[i], items_to_embed[i]) for i in top_idx]

print("Top matches for 'financial risk model':")
print(retrieve("financial risk model", k=2))
