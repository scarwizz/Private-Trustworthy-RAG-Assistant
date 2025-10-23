# src/data/embed_store.py
"""
EmbedStore: uses Chroma (if available) else FAISS fallback.
- Upsert chunks: expects records with id, embedding (list[float]), metadata (dict), text (str)
- Query: returns results from chosen backend
"""
from typing import List, Dict, Optional
import os
import json
from pathlib import Path

# optional Chroma
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

# FAISS
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

class LocalChromaManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb not installed")
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.Client(ChromaSettings(persist_directory=persist_directory))
        self.col = self.client.get_or_create_collection(name="tai_documents")

    def upsert(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict], documents: List[str]):
        # chroma expects python lists for embeddings
        self.col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, query_embedding: List[float], n_results: int = 5):
        res = self.col.query(query_embeddings=[query_embedding], n_results=n_results)
        return res

class FaissManager:
    def __init__(self, persist_directory: str = "./faiss_index", dim: int = 1536):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss not installed")
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.index_file = self.persist_directory / "index.faiss"
        self.meta_file = self.persist_directory / "meta.json"
        self.dim = int(dim)
        # use IndexFlatIP with normalized vectors
        if self.index_file.exists() and self.meta_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            # meta is mapping str(int_index) -> record
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.meta = {}  # idx (str) -> record
            # if index is empty, no vectors yet

    def upsert(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict], documents: List[str]):
        arr = np.array(embeddings).astype('float32')
        # normalize vectors for IP similarity
        faiss.normalize_L2(arr)
        start_idx = len(self.meta)
        self.index.add(arr)
        for i, _id in enumerate(ids):
            idx = start_idx + i
            self.meta[str(idx)] = {
                "id": _id,
                "metadata": metadatas[i],
                "document": documents[i]
            }
        # persist
        faiss.write_index(self.index, str(self.index_file))
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def query(self, query_embedding: List[float], n_results: int = 5):
        vec = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(vec)
        D, I = self.index.search(vec, n_results)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            rec = self.meta.get(str(int(idx)), None)
            results.append({"score": float(score), "record": rec})
        return results

class EmbedStore:
    def __init__(self, use_chroma: bool = True, chroma_dir: Optional[str] = None, faiss_dir: Optional[str] = None, embedding_dim: int = 1536):
        self.use_chroma = use_chroma and CHROMA_AVAILABLE
        if self.use_chroma:
            chroma_dir = chroma_dir or "./chroma_db"
            self.client = LocalChromaManager(persist_directory=chroma_dir)
        else:
            faiss_dir = faiss_dir or "./faiss_index"
            self.client = FaissManager(persist_directory=faiss_dir, dim=embedding_dim)

    def upsert_chunks(self, chunk_records: List[Dict]):
        ids = [r["id"] for r in chunk_records]
        embs = [r["embedding"] for r in chunk_records]
        metas = [r["metadata"] for r in chunk_records]
        docs = [r["text"] for r in chunk_records]
        self.client.upsert(ids=ids, embeddings=embs, metadatas=metas, documents=docs)

    def query(self, query_embedding: List[float], n_results: int = 5):
        return self.client.query(query_embedding, n_results=n_results)
