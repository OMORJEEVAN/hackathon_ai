import pathway as pw
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

class NovelChunk(pw.Schema):
    text: str
    embedding: list[float]

def chunk_text(text, size=400):
    words = text.split()
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]

def build_novel_index(novel_path):
    text = open(novel_path, encoding="utf-8").read()
    chunks = chunk_text(text)

    rows = [{
        "text": c,
        "embedding": embedder.encode(c).tolist()
    } for c in chunks]

    table = pw.debug.table_from_rows(rows, schema=NovelChunk)

    index = pw.ml.index.KNNIndex(
        table.embedding,
        table,
        n_dimensions=384
    )
    return index
