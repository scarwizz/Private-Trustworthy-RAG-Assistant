"""
Phase 3 — Retrieval + Generation Pipeline
Connects retriever (GemmaEmbeddings + EmbedStore) with generator (Flan-T5).
"""

from typing import List, Dict
from src.data.embed_store import EmbedStore
from src.models.embeddings import GemmaEmbeddings
from src.models.generator import FlanT5Generator  # You’ll define or import this
from src.utils.logger import get_logger
from src.utils.config import Config


logger = get_logger("retrieval_generation")

class RAGPipeline:
    def __init__(
        self,
        embedder: GemmaEmbeddings,
        store: EmbedStore,
        generator: FlanT5Generator,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict]:
        """Embed query and fetch top-k most relevant chunks."""
        query_emb = self.embedder.embed_texts([query])[0]
        results = self.store.query(query_emb, n_results=self.top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]

    def generate(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Feed retrieved context + query into Flan-T5 generator."""
        context = "\n\n".join([d["text"] for d in retrieved_docs])
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            f"Answer concisely based on the context above:"
        )
        response = self.generator.generate(prompt)
        return response.strip()

    def run(self, query: str) -> Dict[str, str]:
        """Full retrieval + generation flow."""
        logger.info(f"Running RAG pipeline for query: {query}")
        retrieved = self.retrieve(query)
        answer = self.generate(query, retrieved)
        return {
            "query": query,
            "answer": answer,
            "sources": [r["metadata"] for r in retrieved],
        }

# Example CLI usage
if __name__ == "__main__":
    import argparse
    from src.utils.config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", required=True)
    parser.add_argument("--use_chroma", action="store_true", default=False)
    parser.add_argument("--chroma_dir", default="./chroma_db")
    parser.add_argument("--faiss_dir", default="./faiss_index")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    cfg = Config()
    embedder = GemmaEmbeddings(model_name=cfg.gemma_model_path, local_only=True)
    store = EmbedStore(
    use_chroma=True,
    chroma_dir="./chroma_db",
    faiss_dir="./faiss_index",
    embedding_dim=1536
   ) 
    generator = FlanT5Generator(model_name=cfg.llm_model_path, device=None)

    rag = RAGPipeline(embedder, store, generator, top_k=args.top_k)
    result = rag.run(args.query)

    print("\n🧠 Query:", result["query"])
    print("💬 Answer:", result["answer"])
    print("📚 Sources:", result["sources"])
