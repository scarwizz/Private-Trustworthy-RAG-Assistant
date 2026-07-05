from transformers import AutoTokenizer, AutoModel
import torch
from chromadb import EphemeralClient

# --- Load embedding model ---
model_id = "BAAI/bge-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

# --- Create test embedding ---
text = "Demand"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    embedding = model(**inputs).last_hidden_state.mean(dim=1).tolist()

# --- Use Chroma EphemeralClient (in-memory) ---
client = EphemeralClient()
collection = client.create_collection("test_collection")

# --- Add mock docs ---
collection.add(
    embeddings=[embedding[0], embedding[0]],
    documents=[
        "Doc A: Testing Chroma in-memory query.",
        "Doc B: Demand response project details."
    ],
    ids=["1", "2"],
)

print("✅ Added mock docs")

# --- Query ---
results = collection.query(query_embeddings=embedding, n_results=2)
for i, doc in enumerate(results["documents"][0]):
    print(f"\n--- CHUNK {i+1} ---\n{doc}")
