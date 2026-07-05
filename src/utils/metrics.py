"""
src/utils/metrics.py

Helper utilities for similarity and confidence calculations used by the RAG pipeline.

Functions:
- compute_cosine_similarity(vec1, vec2)
- calculate_confidence_scores(query_embedding, chunk_embeddings)

Notes:
- Inputs may be Python lists (as produced by many embedder APIs). Functions
  convert to numpy arrays internally.
- calculate_confidence_scores returns scores normalized to [0.0, 1.0].
- Defensive: handles zero vectors and degenerate cases.
"""
from typing import List
import math

try:
    import numpy as np
except Exception:  # pragma: no cover
    # numpy is a standard scientific dependency; if missing, raise informative error
    raise RuntimeError("numpy is required for src.utils.metrics. Install via `pip install numpy`.")


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: list/iterable of floats
        vec2: list/iterable of floats

    Returns:
        cosine similarity as float in [-1.0, 1.0]. If either vector is zero-vector,
        returns 0.0 (safe fallback).
    """
    a = np.asarray(vec1, dtype=float)
    b = np.asarray(vec2, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    # defensive shapes
    if a.ndim != 1 or b.ndim != 1:
        a = a.ravel()
        b = b.ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    # numerical safety clamp
    if math.isnan(sim):
        return 0.0
    return max(-1.0, min(1.0, sim))


def calculate_confidence_scores(query_embedding: List[float], chunk_embeddings: List[List[float]]) -> List[float]:
    """
    Given a query embedding and a list of chunk embeddings, compute cosine similarities
    and normalize them to [0, 1] as confidence scores.

    Normalization strategy:
      - compute raw cosine similarities (range [-1,1])
      - shift to [0,2] by adding +1
      - divide by 2 to map to [0,1]
      - if all scores identical (degenerate), return equal probabilities (or raw scaled values)

    Args:
        query_embedding: single vector (list of floats)
        chunk_embeddings: list of vectors

    Returns:
        List[float] of normalized scores in [0.0, 1.0], same length as chunk_embeddings.
    """
    if not chunk_embeddings:
        return []

    # compute raw sims
    raw_sims = []
    for emb in chunk_embeddings:
        sim = compute_cosine_similarity(query_embedding, emb)
        raw_sims.append(sim)

    # map from [-1,1] -> [0,1]
    mapped = [(s + 1.0) / 2.0 for s in raw_sims]

    # If all values equal (or nearly), return them directly (already in [0,1])
    arr = np.asarray(mapped, dtype=float)
    if np.allclose(arr, arr[0]):
        # slightly adjust to sum to 1? We prefer keeping absolute confidences, not probabilities.
        return [float(x) for x in arr]

    # Otherwise, optionally rescale to 0-1 by min-max
    minv = float(np.min(arr))
    maxv = float(np.max(arr))
    if maxv - minv <= 1e-12:
        return [float(x) for x in arr]
    normalized = [(x - minv) / (maxv - minv) for x in arr]
    return [float(x) for x in normalized]
