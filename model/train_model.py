import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from consistency_model import ConsistencyClassifier

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

df = pd.read_csv("data/train.csv")

X, y = [], []

for _, row in df.iterrows():
    combined = (
        f"Character: {row['char']}. "
        f"{row['caption']} {row['content']}"
    )
    X.append(embed(combined))
    y.append(1 if row["label"] == "consistent" else 0)

X = torch.cat(X)
y = torch.tensor(y).float().unsqueeze(1)

model = ConsistencyClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCELoss()

for epoch in range(5):
    optimizer.zero_grad()
    preds = model(X)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model/saved_model/model.pt")
print(" Model trained and saved.")