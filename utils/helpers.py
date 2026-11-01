"""
utils/helpers.py — Shared utility functions.

Provides text-processing helpers, similarity computation, and token
estimation used across multiple modules.
"""

import re
import numpy as np
from typing import List


def clean_text(text: str) -> str:
    """Remove extra whitespace, control characters, and normalise unicode."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_entity_name(name: str) -> str:
    """Lowercase, strip, and collapse whitespace for matching."""
    return re.sub(r'\s+', ' ', name.strip().lower())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Cosine similarity of *query* (1-D) against every row of *corpus* (2-D)."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    corpus_norms = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    return corpus_norms @ query_norm


def estimate_tokens(text: str) -> int:
    """Rough token estimate (≈ word-count × 1.3 for sub-word tokenizers)."""
    return max(1, int(len(text.split()) * 1.3))


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50,
               min_length: int = 30) -> List[str]:
    """
    Split *text* into word-level chunks with overlap.

    Parameters
    ----------
    chunk_size : target words per chunk.
    overlap    : words shared between consecutive chunks.
    min_length : minimum words to keep a chunk.

    Returns
    -------
    List of chunk strings.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if len(words) >= min_length else []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        if len(chunk.split()) >= min_length:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text to *max_len* characters with an ellipsis."""
    return text if len(text) <= max_len else text[:max_len - 3] + '...'


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Division that returns *default* instead of raising on zero."""
    return a / b if b != 0 else default
