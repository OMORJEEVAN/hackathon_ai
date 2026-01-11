import torch
from transformers import AutoTokenizer, AutoModel
from model.consistency_model import ConsistencyClassifier

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
encoder = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def embed(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        return encoder(**inputs).last_hidden_state.mean(dim=1)

model = ConsistencyClassifier()
model.load_state_dict(
    torch.load("model/saved_model/model.pt", map_location="cpu")
)
model.eval()

def evaluate_claim(claim, index):
    query_vec = embed(claim).numpy()[0]
    results = index.search(query_vec, k=5)

    passages = [r["text"] for r in results]
    combined_context = " ".join(passages)

    score = model(embed(combined_context)).item()
    label = 1 if score >= 0.5 else 0

    rationale = {
        "decision": "Consistent" if label else "Contradicted",
        "confidence": round(score, 3),
        "evidence_passages": passages
    }
    return label, rationale