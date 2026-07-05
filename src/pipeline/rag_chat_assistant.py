"""
RAG Chat Assistant — local + private
------------------------------------
Uses:
 - google/embedding-gemma-300M for embeddings
 - google/flan-t5-base for generation
 - ChromaDB for retrieval
 - Persistent chat memory across sessions
"""

import os
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
from src.data.embed_store import EmbedStore
from src.utils.config import Config

# ========================
# 🔧 SETUP
# ========================
cfg = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEMORY_FILE = "chat_memory.json"

# --- Assistant Persona ---
ASSISTANT_PERSONA = """
You are Arti, Arpit's helpful AI assistant. 
You answer questions clearly, concisely, and refer to Arpit's documents when relevant. 
Avoid hallucinating — if unsure, say "I’m not fully sure, but here’s what I found."
"""

# ========================
# 🧠 LOAD MODELS
# ========================
print("🚀 Loading Embedding Gemma + Flan-T5 models...")
embed_model_id = cfg.gemma_model_path  # google/embedding-gemma-300M
gen_model_id = cfg.llm_model_path      # google/flan-t5-base

embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
embed_model = AutoModel.from_pretrained(embed_model_id).to(DEVICE)

gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_id)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_id).to(DEVICE)

# ========================
# 💾 LOAD CHROMA STORE
# ========================
print("🧠 Initializing Chroma store...")
store = EmbedStore(use_chroma=True)
existing_docs = store.client.col.count()
print(f"✅ Chroma loaded with {existing_docs} chunks.")

# ========================
# 💬 CHAT MEMORY
# ========================
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        chat_history = json.load(f)
    print(f"💾 Loaded {len(chat_history)} previous conversations.")
else:
    chat_history = []

# ========================
# 🔍 EMBEDDING + RETRIEVAL
# ========================
def embed_text(text: str):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        emb = embed_model(**inputs).last_hidden_state.mean(dim=1).cpu().tolist()[0]
    return emb

# ========================
# 🧠 GENERATION
# ========================
def generate_with_flan(prompt: str) -> str:
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = gen_model.generate(**inputs, max_new_tokens=200)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ========================
# 🧩 CHAT LOOP
# ========================
print("\n🤖 RAG Chat Assistant ready! Type 'exit' to quit.\n")

while True:
    try:
        query = input("🧑 You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye, see you next time!")
            break

        # Retrieve relevant context
        query_emb = embed_text(query)
        results = store.query(query_emb, n_results=4)
        retrieved_docs = results.get("documents", [[]])[0]
        context_text = "\n".join(retrieved_docs)

        # Keep short memory context (last 3 exchanges)
        short_history = chat_history[-3:]
        history_text = "\n".join(
            [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in short_history]
        )

        # Build final prompt
        prompt = f"""{ASSISTANT_PERSONA}

Conversation so far:
{history_text}

Relevant document context:
{context_text}

User: {query}
Assistant:"""

        # Generate response
        answer = generate_with_flan(prompt)
        print(f"🤖 {answer}\n")

        # Save to memory
        chat_history.append({"user": query, "assistant": answer})
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, indent=2)

    except KeyboardInterrupt:
        print("\n👋 Session ended.")
        break
