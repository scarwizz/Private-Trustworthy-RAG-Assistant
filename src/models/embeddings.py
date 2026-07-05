# src/models/embeddings.py
"""
GemmaEmbeddings - offline-ready wrapper for google/embeddinggemma-300m
BAAI/bge-base-en-v1.5

Features:
- Uses Hugging Face transformers locally (local_files_only=True)
- Provides both `embed_texts(list[str]) -> List[List[float]]` and
  `embed_text(str) -> List[float]` for single queries
- Performs mean pooling and L2 normalization
- Supports batching to prevent OOM during large embedding runs
"""

from typing import List, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class GemmaEmbeddings:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
        local_only: bool = True,
        batch_size: int = 32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_only = local_only
        self.batch_size = batch_size

        # Load tokenizer and model from local cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=self.local_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=self.local_only).to(self.device)
        self.model.eval()

        # Try to infer embedding dimension
        try:
            self.dim = int(self.model.config.hidden_size)
        except Exception:
            self.dim = None

    # ------------------------------------------------------
    # 🔹 Internal batch embedder
    # ------------------------------------------------------
    @torch.inference_mode()
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for a batch of texts using mean pooling
        and normalize each vector to unit length.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # mean pooling with mask
        if "attention_mask" in encoded:
            mask = encoded["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        else:
            pooled = last_hidden.mean(dim=1)

        # Move to CPU and convert to numpy
        arr = pooled.detach().cpu().numpy()

        # 🔹 Normalize each embedding to unit vector (L2 normalization)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        normalized = arr / norms

        return normalized.tolist()

    # ------------------------------------------------------
    # 🔹 Public multi-text embedding
    # ------------------------------------------------------
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for multiple texts.
        Handles batching to avoid OOM.
        """
        results = []
        n = len(texts)
        if n == 0:
            return []

        bs = max(1, int(self.batch_size))
        for i in range(0, n, bs):
            batch = texts[i : i + bs]
            batch_emb = self._embed_batch(batch)
            results.extend(batch_emb)

        # Set embedding dimension if unknown
        if self.dim is None and len(results) > 0:
            self.dim = len(results[0])

        return results

    # ------------------------------------------------------
    # 🔹 Public single-text wrapper
    # ------------------------------------------------------
    def embed_text(self, text: str) -> List[float]:
        """
        Compute embedding for a single text (compatibility wrapper).
        Equivalent to embed_texts([text])[0].
        """
        if not isinstance(text, str):
            raise ValueError("Input to embed_text() must be a string.")
        emb = self.embed_texts([text])[0]

        # Ensure normalization (redundant but safe)
        vec = np.array(emb, dtype=float)
        vec /= np.linalg.norm(vec) + 1e-12
        return vec.tolist()