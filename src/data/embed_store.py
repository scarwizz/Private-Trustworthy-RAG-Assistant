# # src/data/embed_store.py
# """
# EmbedStore: unified local vector store.
# - Primary backend: ChromaDB (with FAISS if available)
# - Fallback: standalone FAISS index if Chroma not installed
# Fully offline & persistent.
# """

# from typing import List, Dict, Optional
# import os
# import json
# from pathlib import Path
# os.environ["CHROMA_TELEMETRY"] = "False"
# os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"


# # Optional Chroma
# try:
#     import chromadb
#     from chromadb.config import Settings 
#     # as ChromaSettings
#     CHROMA_AVAILABLE = True
# except Exception:
#     CHROMA_AVAILABLE = False

# # FAISS
# try:
#     import faiss
#     import numpy as np
#     FAISS_AVAILABLE = True
# except Exception:
#     FAISS_AVAILABLE = False
# # from chromadb.config import Settings


# # --- CHROMA + FAISS HYBRID BACKEND ---
# class LocalChromaManager:
#     """
#     Modern Chroma manager (compatible with Chroma >= 0.5).
#     Uses FAISS internally for similarity indexing if available.
#     """
#     # def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "tai_documents"):
#     #     if not CHROMA_AVAILABLE:
#     #         raise ImportError("chromadb not installed")

#     #     os.makedirs(persist_directory, exist_ok=True)

#     #     try:
#     #         from chromadb import PersistentClient
#     #         self.client = PersistentClient(path=persist_directory)
#     #         print(f"✅ Initialized Chroma PersistentClient at {persist_directory}")
#     #     except Exception as e:
#     #         raise RuntimeError(f"Failed to initialize Chroma PersistentClient: {e}")

#     #     # get or create a collection
#     #     try:
#     #         self.col = self.client.get_or_create_collection(name=collection_name)
#     #         print(f"✅ Using Chroma collection: {collection_name}")
#     #     except Exception as e:
#     #         raise RuntimeError(f"Failed to create/get Chroma collection: {e}")

#     # def __init__(self, persist_directory="./chroma_db"):
#     #     try:
#     #         self.client = chromadb.Client(
#     #             Settings(persist_directory=persist_directory)
#     #         )
#     #         self.col = self.client.get_or_create_collection(name="tai_documents")
#     #         print(f"✅ Initialized Chroma Persistent Client at {persist_directory}")
#     #     except Exception as e:
#     #         raise RuntimeError(f"Failed to initialize Chroma Client: {e}")
#     def __init__(self, persist_directory="./chroma_db"):
#         try:
#             # ✅ New correct Chroma API (v0.6+)
#             self.client = chromadb.PersistentClient(path=persist_directory)
#             self.col = self.client.get_or_create_collection("tai_documents")
#             print(f"✅ Initialized new Chroma PersistentClient at {persist_directory}")
#         except Exception as e:
#             raise RuntimeError(f"Failed to initialize Chroma PersistentClient: {e}")


#     def upsert(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict], documents: List[str]):
#         self.col.add(
#             ids=ids,
#             embeddings=embeddings,
#             metadatas=metadatas,
#             documents=documents
#         )

#     def query(self, query_embedding: List[float], n_results: int = 5):
#         if isinstance(query_embedding, list) and isinstance(query_embedding[0], list):
#             query_embeddings = query_embedding  # already a list of lists
#         else:
#             query_embeddings = [query_embedding]  # wrap single vector

#         res = self.col.query(
#         query_embeddings=query_embeddings,
#         n_results=n_results
#     )

#         return res


# # --- PURE FAISS BACKEND (fallback only) ---
# class FaissManager:
#     def __init__(self, persist_directory: str = "./faiss_index", dim: int = 768):
#         if not FAISS_AVAILABLE:
#             raise ImportError("faiss not installed")
#         self.persist_directory = Path(persist_directory)
#         self.persist_directory.mkdir(parents=True, exist_ok=True)
#         self.index_file = self.persist_directory / "index.faiss"
#         self.meta_file = self.persist_directory / "meta.json"
#         self.dim = int(dim)

#         # Load or initialize index
#         if self.index_file.exists() and self.meta_file.exists():
#             self.index = faiss.read_index(str(self.index_file))
#             with open(self.meta_file, "r", encoding="utf-8") as f:
#                 self.meta = json.load(f)
#         else:
#             self.index = faiss.IndexFlatIP(self.dim)
#             self.meta = {}

#     def upsert(self, ids: List[str], embeddings: List[List[float]],
#                metadatas: List[dict], documents: List[str]):
#         arr = np.array(embeddings).astype('float32')
#         faiss.normalize_L2(arr)
#         start_idx = len(self.meta)
#         self.index.add(arr)
#         for i, _id in enumerate(ids):
#             self.meta[str(start_idx + i)] = {
#                 "id": _id,
#                 "metadata": metadatas[i],
#                 "document": documents[i]
#             }
#         faiss.write_index(self.index, str(self.index_file))
#         with open(self.meta_file, "w", encoding="utf-8") as f:
#             json.dump(self.meta, f, ensure_ascii=False, indent=2)

#     def query(self, query_embedding: List[float], n_results: int = 5):
#         vec = np.array([query_embedding]).astype('float32')
#         faiss.normalize_L2(vec)
#         D, I = self.index.search(vec, n_results)
#         results = []
#         for score, idx in zip(D[0], I[0]):
#             if idx < 0:
#                 continue
#             rec = self.meta.get(str(int(idx)))
#             results.append({"score": float(score), "record": rec})
#         return results


# # --- UNIFIED WRAPPER ---
# class EmbedStore:
#     def __init__(self,
#                  use_chroma: bool = True,
#                  chroma_dir: Optional[str] = None,
#                  faiss_dir: Optional[str] = None,
#                  embedding_dim: int = 768):
#         """
#         Automatically uses:
#         - Chroma + FAISS hybrid (preferred)
#         - pure FAISS (fallback)
#         """
#         self.use_chroma = use_chroma and CHROMA_AVAILABLE
#         if self.use_chroma:
#             chroma_dir = chroma_dir or "./chroma_db"
#             self.client = LocalChromaManager(persist_directory=chroma_dir)
#         else:
#             faiss_dir = faiss_dir or "./faiss_index"
#             self.client = FaissManager(persist_directory=faiss_dir, dim=embedding_dim)

#         # Alias for direct Chroma access
#         if hasattr(self.client, "col"):
#             self.col = self.client.col


#     def upsert_chunks(self, chunk_records: List[Dict]):
#         ids = [r["id"] for r in chunk_records]
#         embs = [r["embedding"] for r in chunk_records]
#         metas = [r["metadata"] for r in chunk_records]
#         docs = [r["text"] for r in chunk_records]
#         self.client.upsert(ids=ids, embeddings=embs, metadatas=metas, documents=docs)

#     def query(self, query_embedding: List[float], n_results: int = 5):
#         return self.client.query(query_embedding, n_results=n_results)


# # --- TEST HARNESS (optional) ---
# if __name__ == "__main__":
#     import numpy as np
#     print("🔍 Testing EmbedStore...")
#     store = EmbedStore(use_chroma=True, embedding_dim=768)
#     dummy_vecs = np.random.rand(5, 768).tolist()
#     docs = [f"Doc {i}" for i in range(5)]
#     metas = [{"source": "unit_test"} for _ in range(5)]
#     ids = [f"id_{i}" for i in range(5)]
#     store.upsert_chunks([
#         {"id": ids[i], "embedding": dummy_vecs[i], "metadata": metas[i], "text": docs[i]}
#         for i in range(5)
#     ])
#     result = store.query(dummy_vecs[0], n_results=3)
#     print("✅ Query result:", result)
"""
EmbedStore — Unified local vector store for TAI.
✅ Supports new Chroma v0.6+ architecture (no legacy configs)
✅ Falls back to FAISS if Chroma unavailable
✅ Uses local BGE embeddings for RAG indexing
✅ Fully offline & persistent
"""

from typing import List, Dict, Optional, Any
import os, json, argparse
from pathlib import Path

# Disable Chroma telemetry for privacy
os.environ["CHROMA_TELEMETRY"] = "False"

# Try dependencies
try:
    import chromadb
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# ---------------------------------------------------------------------
# ✨ Embedding Generator (BAAI/bge-base-en-v1.5)
# ---------------------------------------------------------------------
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Wrapper around local SentenceTransformer embedding model."""

    def __init__(self, model_path: str = "D:/TAI/models/bge-base-en-v1.5"):
        print(f"🧩 Loading embedding model: {model_path}")
        self.model = SentenceTransformer(model_path)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------
# 🧠 Modern Chroma backend
# ---------------------------------------------------------------------
class LocalChromaManager:
    """Chroma client for v0.6+ persistent architecture."""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "tai_documents"):
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb not installed")

        os.makedirs(persist_directory, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            print(f"✅ Initialized new Chroma PersistentClient at {persist_directory}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chroma Client: {e}")

        try:
            self.col = self.client.get_or_create_collection(name=collection_name)
            print(f"✅ Using Chroma collection: {collection_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to create/get Chroma collection: {e}")

    def upsert(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict], documents: List[str]):
        self.col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, query_embedding: List[float], n_results: int = 5):
        if not isinstance(query_embedding[0], list):
            query_embedding = [query_embedding]
        return self.col.query(query_embeddings=query_embedding, n_results=n_results)


# ---------------------------------------------------------------------
# 🧩 FAISS fallback
# ---------------------------------------------------------------------
class FaissManager:
    """FAISS index used if Chroma unavailable."""

    def __init__(self, persist_directory: str = "./faiss_index", dim: int = 768):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss not installed")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.index_file = self.persist_directory / "index.faiss"
        self.meta_file = self.persist_directory / "meta.json"
        self.dim = int(dim)

        if self.index_file.exists() and self.meta_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.meta = {}

    def upsert(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict], documents: List[str]):
        arr = np.array(embeddings).astype("float32")
        faiss.normalize_L2(arr)
        start_idx = len(self.meta)
        self.index.add(arr)
        for i, _id in enumerate(ids):
            self.meta[str(start_idx + i)] = {
                "id": _id,
                "metadata": metadatas[i],
                "document": documents[i],
            }
        faiss.write_index(self.index, str(self.index_file))
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

    def query(self, query_embedding: List[float], n_results: int = 5):
        vec = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(vec)
        D, I = self.index.search(vec, n_results)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            rec = self.meta.get(str(int(idx)))
            results.append({"score": float(score), "record": rec})
        return results


# ---------------------------------------------------------------------
# 🧰 Unified Wrapper
# ---------------------------------------------------------------------
class EmbedStore:
    """Unified local vector store — auto-picks Chroma or FAISS."""

    def __init__(
        self,
        use_chroma: bool = True,
        chroma_dir: Optional[str] = None,
        faiss_dir: Optional[str] = None,
        embedding_dim: int = 768,
        embed_model_path: str = "D:/TAI/models/bge-base-en-v1.5",
    ):
        self.use_chroma = use_chroma and CHROMA_AVAILABLE
        self.generator = EmbeddingGenerator(embed_model_path)

        if self.use_chroma:
            chroma_dir = chroma_dir or "./chroma_db"
            self.client = LocalChromaManager(persist_directory=chroma_dir)
        else:
            faiss_dir = faiss_dir or "./faiss_index"
            self.client = FaissManager(persist_directory=faiss_dir, dim=embedding_dim)

        if hasattr(self.client, "col"):
            self.col = self.client.col

    # Add plain texts directly
    def upsert_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        embeddings = self.generator.encode(texts)
        ids = [f"id_{i}" for i in range(len(texts))]
        metadatas = metadatas or [{} for _ in texts]
        self.client.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
        print(f"✅ Inserted {len(texts)} documents using BGE embeddings.")

    # Search via text query
    def query_text(self, query: str, n_results: int = 5):
        q_emb = self.generator.encode([query])[0]
        return self.client.query(q_emb, n_results=n_results)

    # Manual chunk upsert (if you already have embeddings)
    def upsert_chunks(self, chunk_records: List[Dict]):
        ids = [r["id"] for r in chunk_records]
        embs = [r["embedding"] for r in chunk_records]
        metas = [r["metadata"] for r in chunk_records]
        docs = [r["text"] for r in chunk_records]
        self.client.upsert(ids=ids, embeddings=embs, metadatas=metas, documents=docs)

    def query(self, query_embedding: List[float], n_results: int = 5):
        return self.client.query(query_embedding, n_results=n_results)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Retrieve all documents and their metadata from the vector store.
        Returns a list of dictionaries, each containing at least 'text' and 'metadata' keys.
        """
        if hasattr(self.client, 'col'):  # ChromaDB backend
            # Use ChromaDB's get method to fetch all documents and metadata
            results = self.client.col.get(include=['documents', 'metadatas'])
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])
            return [{"text": doc, "metadata": meta} for doc, meta in zip(documents, metadatas)]
        else:  # FAISS backend
            # FAISS backend stores metadata in self.client.meta
            texts = []
            metadatas = []
            for record in self.client.meta.values():
                texts.append(record["document"])
                metadatas.append(record["metadata"])
            return [{"text": text, "metadata": meta} for text, meta in zip(texts, metadatas)]

    def embed_text(self, text: str) -> List[float]:
        """
        Compute the embedding for a single text string using the internal embedding model.
        Returns a list of floats representing the embedding vector.
        """
        # The generator expects a list of texts
        embeddings = self.generator.encode([text])
        return embeddings[0] if embeddings else []


# ---------------------------------------------------------------------
# 🧮 CLI Utility — Reindex or Test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmbedStore — build or test your local vector store")
    parser.add_argument("--reindex", action="store_true", help="Rebuild vector store from local docs directory")
    parser.add_argument("--docs", type=str, default="./docs", help="Path to documents folder")
    args = parser.parse_args()

    print("🚀 Initializing EmbedStore (Chroma v0.6+)...")
    store = EmbedStore(use_chroma=True)

    if args.reindex:
        print(f"🔄 Reindexing from: {args.docs}")
        texts, metas = [], []
        for root, _, files in os.walk(args.docs):
            for file in files:
                if file.endswith((".txt", ".md")):
                    path = os.path.join(root, file)
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                        texts.append(text[:2000])  # truncate to avoid massive docs
                        metas.append({"source": path})
        if texts:
            store.upsert_texts(texts, metas)
            print(f"✅ Indexed {len(texts)} documents from {args.docs}")
        else:
            print("⚠️ No documents found to index.")
    else:
        # Quick test
        docs = [f"This is document {i}" for i in range(5)]
        store.upsert_texts(docs, metadatas=[{"source": "unit_test"} for _ in docs])
        result = store.query_text("Which document talks about 3?", n_results=3)
        print("✅ Query result:", result)
