"""
High-level RAG pipeline with explainability features.
Upgraded for Phase 3: Advanced RAG Architecture with:
  - Query Transformation (HyDE + Routing)
  - Hybrid Search (Dense + Sparse with RRF)
  - Cross-Encoder Re-Ranking

Provides:
  - Source citations (document name + snippet)
  - Confidence scores (normalized 0–1)
  - Context preview for transparency
  - Structured output dict: {answer, sources, context_used, debug}

Compatible with:
  - EmbedStore (Chroma-backed)
  - GemmaEmbeddings (local embedding model)
  - LocalGGUFLLM (GGUF quantized LLM via llama-cpp-python)
  - New: QueryTransformer, HybridSearcher, CrossEncoderReranker modules

Handles retrieval gracefully and includes full error fallbacks.
"""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import logging

from src.utils.metrics import calculate_confidence_scores
from src.utils.logger import get_logger
from src.utils.config import Config
from src.data.embed_store import EmbedStore
from src.models.embeddings import GemmaEmbeddings
from src.models.llm_gguf import LocalGGUFLLM
from src.query.transformer import QueryTransformer
from src.retrieval.hybrid_searcher import HybridSearcher
from src.retrieval.reranker import CrossEncoderReranker

logger = get_logger("core.pipeline_explainable")
cfg = Config()


def _make_snippet(chunk_text: str, query: str, window: int = 200) -> str:
    """Create a short snippet centered around query occurrence (if found)."""
    if not chunk_text:
        return ""
    q = query.lower().strip()
    txt = chunk_text
    if q:
        idx = txt.lower().find(q)
        if idx >= 0:
            start = max(0, idx - window // 3)
            end = min(len(txt), idx + len(q) + (window - window // 3))
            snippet = txt[start:end].strip()
            if len(snippet) > window:
                snippet = snippet[:window].rsplit(" ", 1)[0] + "…"
            return snippet
    # fallback: first window chars
    snippet = txt.strip().replace("\n", " ")[:window]
    if len(txt) > window:
        snippet = snippet.rsplit(" ", 1)[0] + "…"
    return snippet


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline with explainability and safety.
    Implements advanced RAG pipeline: Query Transform -> Hybrid Search -> Re-Rank -> Generate.

    Example:
        pipeline = RAGPipeline()
        result = pipeline.answer_query("What is demand?", top_k=3)
    """

    def __init__(
        self,
        store: Optional[EmbedStore] = None,
        embedder: Optional[GemmaEmbeddings] = None,
        generator: Optional[Any] = None,
    ):
        """
        Initialize the RAG pipeline with core and advanced components.

        Args:
            store: EmbedStore instance for vector storage and retrieval.
                   If None, creates a new Chroma-backed EmbedStore.
            embedder: GemmaEmbeddings instance for text embedding.
                      If None, creates from config.
            generator: Language model instance for answer generation.
                       If None, creates LocalGGUFLLM from config.
        """
        # Core components (backward compatible)
        self.store = store or EmbedStore(use_chroma=True, chroma_dir=cfg.paths["VECTOR_DB_DIR"])
        self.embedder = embedder or GemmaEmbeddings(
            model_name=cfg.models["EMBEDDING_MODEL"], local_only=True
        )
        self.generator = generator or LocalGGUFLLM(
            model_path=cfg.models["GGUF_MODEL_PATH"],
            n_ctx=cfg.models["GGUF_N_CTX"],
            n_batch=cfg.models["GGUF_N_BATCH"],
            n_threads=cfg.models["GGUF_N_THREADS"],
            n_gpu_layers=cfg.models["GGUF_N_GPU_LAYERS"],
            temperature=cfg.models["GGUF_TEMPERATURE"],
            top_p=cfg.models["GGUF_TOP_P"],
            top_k=cfg.models["GGUF_TOP_K"],
            repeat_penalty=cfg.models["GGUF_REPEAT_PENALTY"],
            verbose=cfg.models.get("GGUF_VERBOSE", False)
        )
        self.top_k = (
            cfg.system.get("TOP_K_RETRIEVAL", 3)
            if hasattr(cfg, "system")
            else cfg.CONFIG["SYSTEM"]["TOP_K_RETRIEVAL"]
        )

        # Advanced RAG components (Phase 3)
        self.query_transformer = QueryTransformer(llm=self.generator)
        self.hybrid_searcher = HybridSearcher(dense_retriever=self.store)
        self.reranker = CrossEncoderReranker()

        logger.info("RAGPipeline initialized with advanced retrieval pipeline")

    # === Internal helpers (kept for compatibility and potential fallback) ===
    def _embed_query(self, query: str) -> List[float]:
        """Embed query using GemmaEmbeddings."""
        try:
            return self.embedder.embed_text(query)
        except Exception as e:
            logger.exception("❌ Failed to embed query: %s", e)
            raise

    def _retrieve(self, query_emb: List[float], top_k: int) -> Dict[str, Any]:
        """Query the vector store with proper fallbacks."""
        try:
            return self.store.query(query_emb, n_results=top_k)
        except TypeError:
            return self.store.query(query_emb, top_k=top_k)
        except Exception as e:
            logger.exception("❌ Vector store query failed: %s", e)
            raise

    def _collect_chunk_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Re-embed retrieved chunk texts for accurate similarity scoring."""
        emb_list = []
        for c in chunks:
            try:
                emb = self.embedder.embed_text(c)
            except Exception:
                emb = [0.0] * getattr(self.embedder, "dim", 768)
            emb_list.append(emb)
        return emb_list

    # === Public method (refactored for Phase 3) ===
    def answer_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Main API: Implements advanced RAG pipeline.
        Flow: Query Transform -> Hybrid Search -> Re-Rank -> Generate -> Explain.

        Args:
            query: User's question
            top_k: Number of final results to return (after re-ranking)
            return_sources: Whether to include source documents in output
            debug: Whether to include debug information

        Returns:
            Dictionary with answer, sources, context_used, and debug info.
        """
        if top_k is None:
            top_k = self.top_k

        out: Dict[str, Any] = {
            "answer": "",
            "sources": [],
            "context_used": "",
            "debug": {}
        }

        try:
            # === Step 1: Query Transformation ===
            hyde_query, query_route = self.query_transformer.transform(query)
            if debug:
                out["debug"]["hyde_query"] = hyde_query
                out["debug"]["query_route"] = query_route

            # === Step 2: Hybrid Search (Dense + Sparse) ===
            # Retrieve larger set for re-ranking
            retrieval_size = max(top_k * 4, 20)  # Get more candidates for re-ranking
            hybrid_results = self.hybrid_searcher.search(
                dense_query=hyde_query,   # Use HyDE for dense retrieval
                sparse_query=query,       # Use original query for sparse (BM25)
                top_k=retrieval_size
            )

            if debug:
                out["debug"]["hybrid_search_count"] = len(hybrid_results)
                if hybrid_results:
                    out["debug"]["hybrid_top_score"] = hybrid_results[0].get("rrf_score", 0.0)

            # === Step 3: Cross-Encoder Re-Ranking ===
            reranked_results = self.reranker.rerank(
                query=query,              # Use original query for re-ranking
                documents=hybrid_results,
                top_k=top_k
            )

            if debug:
                out["debug"]["reranked_count"] = len(reranked_results)
                if reranked_results:
                    out["debug"]["rerank_top_score"] = reranked_results[0].get("rerank_score", 0.0)

            # === Step 4: Prepare Context and Sources ===
            if not reranked_results:
                out["answer"] = "No relevant information found in the knowledge base."
                return out

            # Extract text and metadata for context generation
            context_texts = [doc["text"] for doc in reranked_results]
            context_used = "\n\n".join(context_texts)
            out["context_used"] = context_used

            # Build source metadata for citation
            sources = []
            for i, doc in enumerate(reranked_results):
                meta = doc.get("metadata", {})
                docname = (
                    meta.get("source_filename")
                    or meta.get("filename")
                    or meta.get("source")
                    or doc.get("id", f"doc_{i}")
                )
                # Create snippet from the document text (highlighting query terms)
                snippet = _make_snippet(doc["text"], query, window=300)
                # Use appropriate score based on availability
                score = doc.get("rerank_score", doc.get("rrf_score", 0.0))
                sources.append({
                    "doc": docname,
                    "snippet": snippet,
                    "score": float(score),
                    "raw_metadata": meta
                })
            out["sources"] = sources

            # === Step 5: Generate Final Answer ===
            if self.generator:
                try:
                    answer_text = self.generator.generate(
                        query=query,
                        contexts=context_texts,  # Use the reranked document texts
                        max_new_tokens=256,
                        temperature=0.1,  # Lower temperature for factual accuracy
                        top_p=0.9
                    )
                except Exception as e:
                    logger.exception("⚠️ Generator failed: %s", e)
                    out["debug"]["generator_error"] = str(e)
                    # Fallback to concatenated context
                    answer_text = " ".join(context_texts[:2])  # Use first two contexts
            else:
                answer_text = " ".join(context_texts[:2])  # Simple fallback

            out["answer"] = answer_text.strip()

            # === Step 6: Compute Confidence Scores (optional) ===
            if self.embedder and len(context_texts) > 0:
                try:
                    # Embed the query and context sentences for confidence
                    query_emb = self._embed_query(query)
                    context_embeddings = self.embedder.embed_texts(context_texts)
                    confidences = calculate_confidence_scores(
                        query_emb, context_embeddings
                    )
                    # Update source scores with confidence
                    for i, (src, conf) in enumerate(zip(sources, confidences)):
                        if i < len(sources):
                            # Blend rerank score with confidence for final score
                            alpha = 0.7  # Weight for rerank score
                            beta = 0.3   # Weight for confidence
                            original_score = src["score"]
                            blended_score = alpha * original_score + beta * float(conf)
                            sources[i]["score"] = float(blended_score)
                        # Re-sort by blended score
                        if i < len(sources)-1 and sources[i]["score"] < sources[i+1]["score"]:
                            # Simple bubble - in practice we'd re-sort but keep it simple
                            pass
                except Exception as e:
                    logger.warning(f"Could not compute confidence scores: {e}")
                    # Keep existing scores

            # Update sources with final scores
            out["sources"] = sorted(sources, key=lambda x: x["score"], reverse=True)

            # === Step 7: Debug Info ===
            if debug:
                out["debug"].update({
                    "retrieved_count": len(hybrid_results) if hybrid_results else 0,
                    "reranked_count": len(reranked_results) if reranked_results else 0,
                    "final_sources_count": len(sources),
                    "query_embedding_norm": float(np.linalg.norm(self._embed_query(query))) if self.embedder else 0.0
                })

        except Exception as e:
            logger.exception("❌ Error in answer_query")
            out["answer"] = f"⚠️ Error processing query: {str(e)}"
            out["debug"]["error"] = str(e)

        return out