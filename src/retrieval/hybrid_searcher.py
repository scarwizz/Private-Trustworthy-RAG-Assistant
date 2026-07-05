# src/retrieval/hybrid_searcher.py
"""
Hybrid search module combining dense (vector) and sparse (BM25) retrieval.
Uses Reciprocal Rank Fusion (RRF) to combine results.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from src.data.embed_store import EmbedStore
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("retrieval.hybrid_searcher")


class HybridSearcher:
    """
    Combines dense vector search (ChromaDB) and sparse BM25 search.
    Fuses results using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: EmbedStore,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60
    ):
        """
        Initialize the hybrid searcher.

        Args:
            dense_retriever: Instance of EmbedStore for dense retrieval
            bm25_k1: BM25 parameter k1
            bm25_b: BM25 parameter b
            rrf_k: RRF parameter (typically 60)
        """
        self.dense_retriever = dense_retriever
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.rrf_k = rrf_k
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_texts: List[str] = []
        self.corpus_metadatas: List[Dict[str, Any]] = []
        self._initialized = False

    def _initialize_bm25(self) -> None:
        """Build BM25 index from all documents in the dense retriever."""
        if self._initialized:
            return

        logger.info("Building BM25 index from dense retriever...")
        # Get all documents from the dense retriever
        # Note: This requires EmbedStore to have a get_all_documents method
        documents = self.dense_retriever.get_all_documents()

        if not documents:
            logger.warning("No documents found for BM25 index initialization")
            return

        self.corpus_texts = [doc["text"] for doc in documents]
        self.corpus_metadatas = [doc["metadata"] for doc in documents]

        # Tokenize corpus for BM25 (simple whitespace tokenization for now)
        tokenized_corpus = [text.lower().split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.bm25_k1,
            b=self.bm25_b
        )
        self._initialized = True
        logger.info(f"BM25 index initialized with {len(self.corpus_texts)} documents")

    def search(
        self,
        dense_query: str,
        sparse_query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using dense and sparse retrievers, fuse with RRF.

        Args:
            dense_query: Query text for dense retrieval (e.g., HyDE transformed query)
            sparse_query: Query text for sparse retrieval (e.g., original query)
            top_k: Number of final results to return

        Returns:
            List of result dictionaries, each containing:
                - text: The chunk text
                - metadata: Associated metadata
                - dense_score: Normalized dense retrieval score (higher is better)
                - sparse_score: BM25 score (higher is better)
                - rrf_score: Fused RRF score (higher is better)
                - rank: Final rank after fusion
        """
        # Ensure BM25 index is initialized
        self._initialize_bm25()
        if not self._initialized or self.bm25 is None:
            logger.warning("BM25 index not available, falling back to dense only")
            # Fallback to dense only
            dense_results = self.dense_retriever.query(
                self.dense_retriever.embed_text(dense_query),
                n_results=top_k
            )
            return self._format_dense_results(dense_results)

        # Get dense results
        dense_embedding = self.dense_retriever.embed_text(dense_query)
        dense_results = self.dense_retriever.query(
            dense_embedding,
            n_results=50  # Retrieve more for better fusion
        )
        dense_hits = self._format_dense_results(dense_results)

        # Get sparse results
        tokenized_query = sparse_query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Get top indices from BM25 scores
        top_indices = np.argsort(bm25_scores)[::-1][:50]  # Top 50 for fusion
        sparse_hits = []
        for idx in top_indices:
            if idx < len(self.corpus_texts):
                sparse_hits.append({
                    "text": self.corpus_texts[idx],
                    "metadata": self.corpus_metadatas[idx],
                    "sparse_score": float(bm25_scores[idx]),
                    "dense_score": 0.0,  # Will be filled if also in dense results
                    "index": int(idx)
                })

        # Combine scores using RRF
        fused_scores = self._reciprocal_rank_fusion(dense_hits, sparse_hits)

        # Sort by fused score and return top_k
        fused_scores.sort(key=lambda x: x["rrf_score"], reverse=True)
        results = fused_scores[:top_k]

        # Add final rank
        for i, result in enumerate(results):
            result["rank"] = i + 1

        return results

    def _format_dense_results(
        self,
        raw_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert raw ChromaDB results to our standard format."""
        hits = []
        # Handle ChromaDB format (dict with lists)
        if isinstance(raw_results, dict):
            ids = raw_results.get("ids", [[]])[0] if isinstance(raw_results.get("ids"), list) else []
            documents = raw_results.get("documents", [[]])[0] if isinstance(raw_results.get("documents"), list) else []
            metadatas = raw_results.get("metadatas", [[]])[0] if isinstance(raw_results.get("metadatas"), list) else []
            distances = raw_results.get("distances", [[]])[0] if isinstance(raw_results.get("distances"), list) else []
            for i in range(len(documents)):
                # Convert distance to similarity score (higher is better)
                # Chroma returns L2 distance by default? We'll use 1/(1+distance) as a similarity measure
                dist = distances[i] if i < len(distances) else 0.0
                similarity = 1.0 / (1.0 + dist) if dist is not None else 0.0
                hits.append({
                    "text": documents[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "dense_score": similarity,
                    "sparse_score": 0.0,  # Will be updated if also in sparse results
                    "index": -1  # Not applicable for dense-only
                })
        # Handle FAISS format (list of dicts with 'score' and 'record')
        elif isinstance(raw_results, list):
            for item in raw_results:
                score = item.get("score", 0.0)
                record = item.get("record", {})
                text = record.get("document", "")
                metadata = record.get("metadata", {})
                # For FAISS, score is already a similarity (higher better)
                similarity = float(score)
                hits.append({
                    "text": text,
                    "metadata": metadata,
                    "dense_score": similarity,
                    "sparse_score": 0.0,
                    "index": -1
                })
        else:
            logger.warning(f"Unexpected raw_results type: {type(raw_results)}")
        return hits

    def _reciprocal_rank_fusion(
        self,
        dense_hits: List[Dict[str, Any]],
        sparse_hits: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Fuse dense and sparse results using Reciprocal Rank Fusion.

        Args:
            dense_hits: List of dense retrieval results (each with 'text' and optional 'index')
            sparse_hits: List of sparse retrieval results (each with 'text' and 'index' in corpus)

        Returns:
            List of fused results with combined scores
        """
        # Create a mapping from document text to combined result
        fused_map: Dict[str, Dict[str, Any]] = {}

        # Process dense hits
        for rank, hit in enumerate(dense_hits, start=1):
            text = hit["text"]
            if text not in fused_map:
                fused_map[text] = {
                    "text": text,
                    "metadata": hit["metadata"],
                    "dense_score": hit["dense_score"],
                    "sparse_score": 0.0,
                    "dense_rank": rank,
                    "sparse_rank": None
                }
            else:
                current_dense_rank = fused_map[text].get("dense_rank")
                if current_dense_rank is None:
                    current_dense_rank = float('inf')
                if rank < current_dense_rank:
                    fused_map[text]["dense_score"] = hit["dense_score"]
                    fused_map[text]["dense_rank"] = rank

        # Process sparse hits
        for rank, hit in enumerate(sparse_hits, start=1):
            text = hit["text"]
            if text not in fused_map:
                fused_map[text] = {
                    "text": text,
                    "metadata": hit["metadata"],
                    "dense_score": 0.0,
                    "sparse_score": hit["sparse_score"],
                    "dense_rank": None,
                    "sparse_rank": rank
                }
            else:
                current_sparse_rank = fused_map[text].get("sparse_rank")
                if current_sparse_rank is None:
                    current_sparse_rank = float('inf')
                if rank < current_sparse_rank:
                    fused_map[text]["sparse_score"] = hit["sparse_score"]
                    fused_map[text]["sparse_rank"] = rank

        # Calculate RRF scores
        results = []
        for text, data in fused_map.items():
            dense_rank = data.get("dense_rank")
            if dense_rank is None:
                dense_rank = float('inf')
            sparse_rank = data.get("sparse_rank")
            if sparse_rank is None:
                sparse_rank = float('inf')

            rrf_score = 0.0
            if dense_rank != float('inf'):
                rrf_score += 1.0 / (self.rrf_k + dense_rank)
            if sparse_rank != float('inf'):
                rrf_score += 1.0 / (self.rrf_k + sparse_rank)

            results.append({
                "text": text,
                "metadata": data["metadata"],
                "dense_score": data["dense_score"],
                "sparse_score": data["sparse_score"],
                "rrf_score": rrf_score
            })

        return results