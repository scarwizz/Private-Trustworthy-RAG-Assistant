


"""
rag_gemma_flan_pipeline.py
--------------------------------------
Unified RAG pipeline runner integrating:
 - Gemma embeddings (local)
 - Flan-T5 generator (local)
 - ChromaDB vector store
 - Explainable pipeline with source confidence and context tracing
"""

import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import chromadb
from PyPDF2 import PdfReader
from docx import Document
import numpy as np

from src.core.memory import ChatMemory
from src.core.pipeline_explainable import RAGPipeline

# === CONFIG ===
DATA_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL_PATH = "D:/TAI/models/BAAI/bge-base-en-v1.5"   # local Gemma embedding modelembeddinggemma-300m
# EMBEDDING_MODEL_ID = "BAAI/bge-base-en-v1.5"

GEN_MODEL = "google/flan-t5-base"                        # Flan-T5 generator model

# # === LOAD MODELS ===
# print("🚀 Loading Embedding Gemma + Flan-T5 models...")
# embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH, local_files_only=True)
# embed_model = AutoModel.from_pretrained(EMBED_MODEL_PATH, local_files_only=True)
print("🚀 Loading BGE embedding model + Flan-T5 generator...")

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("D:/TAI/models/bge-base-en-v1.5")

gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, local_files_only=True)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL, local_files_only=True)

# === INITIALIZE CHROMA ===
print("🧠 Initializing Chroma store...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("cv_docs")

# === LOAD DOCS ===
def load_documents(folder):
    docs = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        text = ""
        if fname.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
        elif fname.endswith(".pdf"):
            for page in PdfReader(fpath).pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
        elif fname.endswith(".docx"):
            doc = Document(fpath)
            text = "\n".join(p.text for p in doc.paragraphs)
        if text.strip():
            docs.append(text)
    return docs

# === CHUNKER ===
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# === EMBED FUNCTION ===
# def embed_text(text):
#     """Generate a single embedding vector using Gemma embedding model."""
#     inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         emb = embed_model(**inputs).last_hidden_state.mean(dim=1)
#     return emb[0].cpu().tolist()

def embed_text(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        emb = embed_model(**inputs).last_hidden_state.mean(dim=1)
    vec = emb[0].cpu().numpy()
    vec /= np.linalg.norm(vec) + 1e-12  # normalize to unit vector
    return vec.tolist()


# === DEBUG: Check embedding dimension consistency ===
def check_embedding_integrity():
    print("\n🔍 Checking Chroma embedding integrity...")
    try:
        count = collection.count()
        print(f"📦 Collection contains {count} entries.")
        sample = collection.peek()
        stored_dim = None
        if sample and "embeddings" in sample:
            emb_data = sample["embeddings"]
    # Ensure it's a non-empty list/array
            if isinstance(emb_data, (list, tuple)) and len(emb_data) > 0:
                first_vec = emb_data[0]
        # Handle possible numpy arrays
                if hasattr(first_vec, "__len__"):
                    stored_dim = len(first_vec)
                    print(f"📏 Stored embedding dimension: {stored_dim}")
                else:
                    print("⚠️ First embedding has no measurable length.")
            else:
                print("⚠️ No embeddings found in peek().")
        else:
            print("⚠️ Peek() returned no embeddings.")

    except Exception as e:
        print(f"❌ Failed to inspect Chroma collection: {e}")
        stored_dim = None
        count = 0

    # Compute Gemma model output dimension
    try:
        dummy_input = embed_tokenizer("test", return_tensors="pt")
        with torch.no_grad():
            dummy_emb = embed_model(**dummy_input).last_hidden_state.mean(dim=1)
        gemma_dim = dummy_emb.shape[1]
        print(f"🧠 Gemma embedding model dimension: {gemma_dim}")
    except Exception as e:
        print(f"❌ Could not compute Gemma embedding dimension: {e}")
        gemma_dim = None

    # Compare
    if stored_dim and gemma_dim and stored_dim != gemma_dim:
        print(f"\n⚠️ Dimension mismatch detected! (Chroma: {stored_dim} vs Gemma: {gemma_dim})")
        print("This means your database was built with a different embedding model.")
        choice = input("🧹 Rebuild ChromaDB now? (y/n): ").strip().lower()
        if choice == "y":
            rebuild_chromadb()
        else:
            print("⚠️ Continuing with possibly incompatible embeddings — retrieval may fail.")
    elif not stored_dim and gemma_dim:
        print("🧹 No valid embeddings found, reindexing from scratch.")
        rebuild_chromadb()
    else:
        print("✅ Embedding dimensions match. No rebuild needed.\n")

# === REBUILD CHROMA DATABASE ===
def rebuild_chromadb():
    """Clean and rebuild the Chroma collection using Gemma embeddings."""
    global collection
    try:
        print("🧹 Cleaning old Chroma collection...")
        client.delete_collection("cv_docs")
    except Exception:
        print("⚠️ No old collection to delete.")
    collection = client.get_or_create_collection("cv_docs")

    docs = load_documents(DATA_DIR)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))
    print(f"📄 Loaded and chunked {len(all_chunks)} chunks from {len(docs)} files.")

    # print("🔢 Embedding chunks using BGE...")
    # embeddings = embed_model.encode(all_chunks, normalize_embeddings=True)
    print("🔢 Embedding chunks using BGE...")
    embeddings = embed_model.encode(all_chunks, normalize_embeddings=True)


    ids = [f"id_{i}" for i in range(len(all_chunks))]

    collection.add(documents=all_chunks, embeddings=embeddings, ids=ids)
    print("✅ Rebuilt and reindexed ChromaDB successfully!\n")

# === INDEX DOCS (normal flow) ===
def index_documents():
    """Index docs only if collection empty."""
    existing = collection.count()
    if existing > 0:
        print(f"✅ Chroma already contains {existing} documents, skipping re-index.")
        return
    rebuild_chromadb()

# === MEMORY SETUP ===
memory = ChatMemory()

# === PIPELINE ADAPTER ===
class LocalFlanGenerator:
    """Wrapper around Flan-T5 so it's compatible with RAGPipeline interface."""
    def generate(self, query, contexts, max_new_tokens=250):
        joined_context = "\n".join(contexts)
        prompt = (
    "You are an assistant that answers ONLY using the provided context. "
    "If the context does not contain the answer, respond: "
    "'Not found in knowledge base.'\n\n"
    f"Context:\n{joined_context}\n\nQuestion: {query}\n\nAnswer:"
)
        inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = gen_model.generate(**inputs, max_new_tokens=max_new_tokens)
        return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Instantiate the generator + pipeline
generator = LocalFlanGenerator()
# rag_pipeline = RAGPipeline(generator=generator)
rag_pipeline = RAGPipeline(
    store=None,  # will use the existing collection directly below
    generator=generator
)
# Manually override the store to use the same Chroma collection
rag_pipeline.store.collection = collection

# === MAIN RUNNER ===
if __name__ == "__main__":
    print("📥 Checking vector store integrity...")
    check_embedding_integrity()

    print("📥 Indexing (if needed)...")
    index_documents()

    print("\n🧠 RAG assistant ready with explainability! Type 'exit' to quit.\n")

    while True:
        query = input("❓ Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting chat. Goodbye!")
            break

        print(f"💬 Query: {query}")
        result = rag_pipeline.answer_query(query, top_k=3, debug=True)

        # Save in memory
        memory.add(query, result["answer"])

        # Display formatted explainable output
        print("\n🤖 Answer:", result["answer"])
        print("\n📚 Sources:")
        for src in result["sources"]:
            print(f"  - {src['doc']} ({src['score']:.2f}): {src['snippet'][:200]}...")
        print("\n🧠 Context used:", result["context_used"][:300], "...\n")
