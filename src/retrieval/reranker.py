# src/retrieval/reranker.py
"""
Cross-encoder re-ranking module for the Private RAG Assistant.
Uses a cross-encoder model to re-score retrieved documents.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from typing import Optional, List, Dict, Any
from sentence_transformers import CrossEncoder
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger("retrieval.reranker")


class CrossEncoderReranker:
    """
    Re-ranks retrieved documents using a cross-encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 32
    ):
        """
        Initialize the cross-encoder re-ranker.

        Args:
            model_name: Name or path of the cross-encoder model
            max_length: Maximum sequence length for the model
            batch_size: Batch size for scoring
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model: Optional[CrossEncoder] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length
            )
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {self.model_name}: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on their relevance to the query using cross-encoder.

        Args:
            query: User query string
            documents: List of document dictionaries, each containing at least 'text' and 'metadata'
            top_k: Number of top results to return after re-ranking. If None, returns all.

        Returns:
            List of re-ranked document dictionaries, each with an added 'rerank_score' field.
            The list is sorted by 'rerank_score' in descending order.
        """
        if not documents:
            return []

        if self.model is None:
            logger.warning("Cross-encoder model not loaded, returning original order")
            return documents

        # Prepare pairs of (query, document text)
        pairs = [[query, doc["text"]] for doc in documents]

        # Get relevance scores from the cross-encoder
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
        except Exception as e:
            logger.error(f"Error during cross-encoder prediction: {e}")
            return documents  # Fallback to original order

        # Add scores to documents
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            scored_docs.append(doc_copy)

        # Sort by re-rank score descending
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            return scored_docs[:top_k]
        return scored_docs