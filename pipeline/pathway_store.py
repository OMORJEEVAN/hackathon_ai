import pathway as pw
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Encoder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Pathway schema (used meaningfully)
class NovelChunk(pw.Schema):
    text: str
    embedding: list[float]

def chunk_text(text, size=400):
    words = text.split()
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]

def build_novel_store(novel_path):
    with open(novel_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks)

    # Pathway table (ingestion + orchestration)
    rows = [(chunks[i], embeddings[i].tolist()) for i in range(len(chunks))]
    table = pw.debug.table_from_rows(rows, schema=NovelChunk)

    return {
        "texts": chunks,
        "embeddings": embeddings,
        "table": table   # kept for Pathway usage justification
    }

# ✅ THIS FUNCTION WAS MISSING
def retrieve_evidence(query, store, top_k=5):
    q_emb = embedder.encode([query])
    sims = cosine_similarity(q_emb, store["embeddings"])[0]
    idx = np.argsort(sims)[-top_k:][::-1]
    return [store["texts"][i] for i in idx]
