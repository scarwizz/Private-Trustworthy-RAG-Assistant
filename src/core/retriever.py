# src/core/retriever.py
"""
Retriever implementation that uses the project's Gemma embeddings wrapper and the local
EmbedStore (Chroma/FAISS) to return top-k chunks for a text query.

Implements:
    class EmbedStoreRetriever:
        retrieve(query: str, top_k: int) -> List[Dict]
"""
from typing import List, Dict, Optional
import logging

from src.models.embeddings import GemmaEmbeddings
from src.data.embed_store import EmbedStore
from src.utils.config import CONFIG

try:
    # optional encryptor hook (kept optional)
    from src.security.encryptor import maybe_encrypt_metadata  # type: ignore
except Exception:
    maybe_encrypt_metadata = None  # type: ignore

logger = logging.getLogger(__name__)


class EmbedStoreRetriever:
    """
    Retriever that converts a query into an embedding using GemmaEmbeddings,
    then queries the EmbedStore (Chroma/FAISS) for the top-k most similar chunks.

    Returned list items are normalized dicts with keys:
        id, text, metadata, score
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        store: Optional[EmbedStore] = None,
        device: Optional[str] = None,
    ):
        model_name = embedding_model_name or CONFIG["MODELS"]["EMBEDDING_MODEL"]
        self.embedder = GemmaEmbeddings(model_name, device=device, local_only=True)
        # If an external store instance is given use it; otherwise create one using config
        if store is not None:
            self.store = store
        else:
            self.store = EmbedStore(
                use_chroma=True,
                chroma_dir=CONFIG["PATHS"]["VECTOR_DB_DIR"],
                faiss_dir=CONFIG["PATHS"].get("FAISS_DIR", None),
                embedding_dim=CONFIG.get("SYSTEM", {}).get("EMBEDDING_DIM", 768),
            )
            # Turn off telemetry if Chroma supports it (defensive).
            try:
                self.store.set_telemetry(False)
            except Exception:
                # Not all store wrappers expose telemetry toggle - ignore if absent.
                pass

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve top_k chunks for `query`.

        :param query: user query string
        :param top_k: number of top results to return (defaults to CONFIG["SYSTEM"]["TOP_K_RETRIEVAL"])
        :returns: list of dicts with keys: id, text, metadata, score
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string.")

        top_k = int(top_k) if top_k is not None else CONFIG["SYSTEM"].get("TOP_K_RETRIEVAL", 5)

        # Compute embedding (wrap in try/except so we return helpful errors)
        try:
            q_emb = self.embedder.embed_texts([query])[0]
        except Exception as e:
            logger.exception("Failed to create query embedding.")
            raise RuntimeError(f"Failed to create query embedding: {e}") from e

        # Query the store
        try:
            raw = self.store.query(q_emb, n_results=top_k)
        except Exception as e:
            logger.exception("EmbedStore.query failed.")
            raise RuntimeError(f"Embed store query failed: {e}") from e

        # Normalize different store backends (Chroma-style -> {ids, documents, metadatas, distances})
        results: List[Dict] = []
        # Defensive checks for common shapes
        try:
            ids = raw.get("ids") or raw.get("ids_list") or []
            # Chroma returns nested lists for batched queries; flatten if necessary
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
        except Exception:
            ids = []

        try:
            texts = raw.get("documents") or raw.get("texts") or raw.get("documents_list") or []
            if isinstance(texts, list) and texts and isinstance(texts[0], list):
                texts = texts[0]
        except Exception:
            texts = []

        try:
            metadatas = raw.get("metadatas") or raw.get("metadata") or []
            if isinstance(metadatas, list) and metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]
        except Exception:
            metadatas = []

        try:
            distances = raw.get("distances") or raw.get("scores") or raw.get("dist") or []
            if isinstance(distances, list) and distances and isinstance(distances[0], list):
                distances = distances[0]
        except Exception:
            distances = []

        # If store returned fewer pieces, pad with placeholders gracefully
        max_len = max(len(ids), len(texts), len(metadatas), len(distances))

        for i in range(max_len):
            rec_id = ids[i] if i < len(ids) else None
            text = texts[i] if i < len(texts) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}
            score = distances[i] if i < len(distances) else None

            # Optionally encrypt metadata if project uses encryption hook (we keep raw by default)
            if maybe_encrypt_metadata:
                try:
                    metadata = maybe_encrypt_metadata(metadata)
                except Exception:
                    # If encryption fails, ignore and return raw metadata (non-fatal)
                    logger.warning("maybe_encrypt_metadata failed; returning raw metadata.")

            results.append(
                {
                    "id": rec_id,
                    "text": text,
                    "metadata": metadata,
                    "score": float(score) if score is not None else None,
                }
            )

        # Sort by score descending if higher is better, otherwise (for distances) smaller is better.
        # We heuristically check: if scores look like distances (>=0, higher = worse) we invert.
        def _sort_key(r):
            s = r.get("score")
            # missing numeric score -> very low priority
            if s is None:
                return -float("inf")
            # If score appears to be a distance (values > 1 or commonly > 0.01), treat smaller better.
            # We'll detect whether smaller-is-better by comparing against a threshold.
            if s > 1.0 or s > 0.05:
                # smaller is better -> sort descending by -score
                return -s
            # else assume similarity where larger is better
            return s

        # Keep deterministic ordering in ties by stable sort
        results_sorted = sorted(results, key=_sort_key, reverse=False)

        # Return only top_k results (in case we expanded earlier)
        return results_sorted[:top_k]
