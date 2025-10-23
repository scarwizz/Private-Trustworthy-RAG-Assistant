# src/models/embeddings.py
"""
GemmaEmbeddings - offline-ready wrapper for google/embeddinggemma-300m
Uses Hugging Face transformers locally (local_files_only=True).
Provides embed_texts(list[str]) -> List[List[float]] and embed_text(str) -> List[float].
Batching included to prevent OOM.
"""
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import math

class GemmaEmbeddings:
    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        device: Optional[str] = None,
        local_only: bool = True,
        batch_size: int = 32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_only = local_only
        self.batch_size = batch_size

        # Load tokenizer and model from local cache (requires model cached beforehand)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=self.local_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=self.local_only).to(self.device)
        self.model.eval()

        # infer embedding dimension
        # some models expose config.hidden_size; fallback to model output last_hidden_state size at runtime
        try:
            self.dim = int(self.model.config.hidden_size)
        except Exception:
            self.dim = None

    @torch.inference_mode()
    def _embed_batch(self, texts: List[str]):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024,  # safe limit; adjust if model supports more
        ).to(self.device)

        outputs = self.model(**encoded)
        # mean pooling over token embeddings (simple and usually decent)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        # compute attention mask mean to avoid padding effect if available
        if hasattr(encoded, "attention_mask"):
            mask = encoded["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        else:
            pooled = last_hidden.mean(dim=1)

        arr = pooled.detach().cpu().numpy()
        return arr.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # batch the texts
        results = []
        n = len(texts)
        if n == 0:
            return []
        bs = max(1, int(self.batch_size))
        for i in range(0, n, bs):
            batch = texts[i : i + bs]
            batch_emb = self._embed_batch(batch)
            results.extend(batch_emb)
        # set dim if unknown
        if self.dim is None and len(results) > 0:
            self.dim = len(results[0])
        return results

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
