import os
import pandas as pd
from pipeline.pathway_store import build_novel_store
from pipeline.reasoning import evaluate_claim, load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NOVEL_PATH ="data/In search of the castaways.txt"
CSV_PATH =  "data/train.csv"
MODEL_PATH = "model/saved_model/model.pt"
OUTPUT_PATH = "test.csv"

print(" Building novel store...")
store = build_novel_store(NOVEL_PATH)

print(" Loading trained model...")
model = load_model(MODEL_PATH)

df = pd.read_csv(CSV_PATH)

results = []

print(f" Running inference on {len(df)} rows...\n")

for _, row in df.iterrows():
    claim_text = (
        f"Character: {row['char']}. "
        f"{row['caption']} {row['content']}"
    )

    label, rationale = evaluate_claim(claim_text, store, model)

    results.append({
        "character": row["char"],
        "claim": row["caption"],
        "judgment": "Consistent" if label else "Contradicted",
        "confidence": rationale["confidence"],
        "evidence": " || ".join(rationale["evidence_passages"][:2])
    })

#  SAVE TO test.csv (NO LABEL)
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_PATH, index=False)

print(f" Inference complete.")
print(f" Results saved to: {OUTPUT_PATH}")