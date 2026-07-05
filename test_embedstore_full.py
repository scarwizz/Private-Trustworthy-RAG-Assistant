# test_embedstore_full.py
"""
✅ Full RAG pipeline test:
- Embed + store with google/embedding-gemma-300M
- Retrieve from Chroma
- Generate with Flan-T5
"""

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
from src.data.embed_store import EmbedStore

# ---------------------------------------------------------------------
# Step 1: Initialize embed store
# ---------------------------------------------------------------------
print("\n🚀 Testing EmbedStore + RAG Pipeline...\n")

store = EmbedStore(use_chroma=True)
print("✅ EmbedStore initialized successfully.\n")

# ---------------------------------------------------------------------
# Step 2: Load embedding model (Google Embedding Gemma)
# ---------------------------------------------------------------------
model_id = "google/BAAI/bge-base-en-v1.5"
print(f"📦 Loading embedding model: {model_id}")
embed_tokenizer = AutoTokenizer.from_pretrained(model_id)
embed_model = AutoModel.from_pretrained(model_id)

# Helper for embedding
def get_embedding(text: str):
    inputs = embed_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        emb = embed_model(**inputs).last_hidden_state.mean(dim=1)
    return emb[0].tolist()

# ---------------------------------------------------------------------
# Step 3: Add mock documents
# ---------------------------------------------------------------------
docs = [
    "Demand response helps balance electricity supply and demand by adjusting usage during peak times.",
    "Machine learning models can predict energy consumption patterns using historical data.",
    "Renewable energy integration requires smart grid technologies and adaptive demand management."
]

records = []
for i, text in enumerate(docs):
    emb = get_embedding(text)
    records.append({
        "id": f"doc_{i}",
        "embedding": emb,
        "metadata": {"source": "test_corpus"},
        "text": text
    })

store.upsert_chunks(records)
print("✅ Added test documents to vector store.\n")

# ---------------------------------------------------------------------
# Step 4: Query similar chunks
# ---------------------------------------------------------------------
query_text = "How can electricity demand be balanced during peak load?"
query_emb = get_embedding(query_text)
results = store.query(query_emb, n_results=3)

print("🔎 Retrieved relevant chunks:")
for i, doc in enumerate(results["documents"][0]):
    print(f"\n--- CHUNK {i+1} ---\n{doc[:200]}")

# ---------------------------------------------------------------------
# Step 5: Load generator model (Flan-T5)
# ---------------------------------------------------------------------
gen_model_id = "google/flan-t5-base"
print(f"\n🧠 Loading generator model: {gen_model_id}")
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_id)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_id)

# ---------------------------------------------------------------------
# Step 6: Generate contextual answer
# ---------------------------------------------------------------------
context = " ".join(results["documents"][0])
prompt = f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
inputs = gen_tokenizer(prompt, return_tensors="pt")
outputs = gen_model.generate(**inputs, max_new_tokens=150)
answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n💬 Generated Answer:\n", answer)
print("\n✅ RAG pipeline working successfully!\n")
