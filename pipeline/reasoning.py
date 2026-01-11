import torch
from sentence_transformers import SentenceTransformer
from model.consistency_model import ConsistencyModel
from pipeline.pathway_store import retrieve_evidence

# ✅ Use the SAME encoder used in training
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_model():
    model = ConsistencyModel(input_dim=384)
    model.load_state_dict(
        torch.load("model/saved_model/model.pt", map_location="cpu")
    )
    model.eval()
    return model

def evaluate_claim(claim_text, store, model):
    # ---- Embed the CLAIM (not the evidence) ----
    emb = embedder.encode([claim_text])
    emb = torch.tensor(emb, dtype=torch.float32)

    with torch.no_grad():
        score = model(emb).item()

    label = 1 if score >= 0.5 else 0

    # ---- Retrieve evidence separately ----
    passages = retrieve_evidence(claim_text, store, top_k=5)

    rationale = {
        "decision": "Consistent" if label else "Contradicted",
        "confidence": round(score, 3),
        "evidence_passages": passages
    }

    return label, rationale