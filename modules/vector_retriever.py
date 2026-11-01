"""
modules/vector_retriever.py — Vector Index and Initial Retrieval (Module 3).

Generates embeddings for document chunks and knowledge-graph entities using
Sentence Transformers (all-MiniLM-L6-v2).  Provides cosine-similarity based
retrieval for initial anchor-node selection.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional

import numpy as np
import networkx as nx

import config
from utils.helpers import cosine_similarity_matrix, normalize_entity_name


# ──────────────────────────────────────────────────────────────
# Embedding Model Wrapper
# ──────────────────────────────────────────────────────────────

class VectorRetriever:
    """
    Manages embedding generation and similarity-based retrieval.

    Attributes
    ----------
    model       : SentenceTransformer model.
    chunk_texts : list of chunk text strings (parallel to chunk_embeds).
    chunk_ids   : list of chunk IDs (parallel to chunk_embeds).
    chunk_embeds: (N, D) numpy array of chunk embeddings.
    entity_ids  : list of normalised entity IDs.
    entity_texts: list of entity description strings.
    entity_embeds: (M, D) numpy array of entity embeddings.
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self._dim = self.model.get_sentence_embedding_dimension()

        # Indexed data (populated by build_index)
        self.chunk_texts: List[str] = []
        self.chunk_ids: List[str] = []
        self.chunk_embeds: Optional[np.ndarray] = None

        self.entity_ids: List[str] = []
        self.entity_texts: List[str] = []
        self.entity_embeds: Optional[np.ndarray] = None

    # ── Encoding ──────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Encode a single string and return a 1-D numpy vector."""
        return self.model.encode(text, convert_to_numpy=True,
                                 show_progress_bar=False)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings and return (N, D) array."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return self.model.encode(texts, convert_to_numpy=True,
                                 show_progress_bar=False, batch_size=64)

    # ── Index Construction ────────────────────────────────────

    def build_index(self, chunks, graph: nx.DiGraph) -> None:
        """
        Build the embedding index from document chunks and graph nodes.

        Parameters
        ----------
        chunks : list of TextChunk objects.
        graph  : NetworkX DiGraph with node attribute 'description'.
        """
        # Chunk embeddings
        self.chunk_texts = [c.text for c in chunks]
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.chunk_embeds = self.encode_batch(self.chunk_texts)

        # Entity embeddings — use "name: description" as the text
        self.entity_ids = []
        self.entity_texts = []
        for nid, data in graph.nodes(data=True):
            name = data.get('name', nid)
            desc = data.get('description', '')
            self.entity_ids.append(nid)
            self.entity_texts.append(f"{name}: {desc}")
        self.entity_embeds = self.encode_batch(self.entity_texts)

    # ── Retrieval ─────────────────────────────────────────────

    def retrieve_chunks(self, query: str, top_k: int = 5
                        ) -> List[Tuple[str, str, float]]:
        """
        Return the *top_k* most relevant (chunk_id, text, score) tuples
        for the given query.
        """
        if self.chunk_embeds is None or len(self.chunk_embeds) == 0:
            return []
        q_emb = self.encode(query)
        scores = cosine_similarity_matrix(q_emb, self.chunk_embeds)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.chunk_ids[i], self.chunk_texts[i], float(scores[i]))
            for i in top_indices
            if scores[i] >= config.SIMILARITY_THRESHOLD
        ]

    def retrieve_entities(self, query: str, top_k: int = 5
                          ) -> List[Tuple[str, float]]:
        """
        Return the *top_k* most relevant (entity_id, score) tuples.
        These serve as initial anchor nodes for graph retrieval.
        """
        if self.entity_embeds is None or len(self.entity_embeds) == 0:
            return []
        q_emb = self.encode(query)
        scores = cosine_similarity_matrix(q_emb, self.entity_embeds)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.entity_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] >= config.SIMILARITY_THRESHOLD
        ]

    def get_entity_score(self, query_embedding: np.ndarray,
                         entity_id: str) -> float:
        """Cosine similarity between a query embedding and a single entity."""
        if self.entity_embeds is None:
            return 0.0
        try:
            idx = self.entity_ids.index(entity_id)
        except ValueError:
            return 0.0
        ent_emb = self.entity_embeds[idx]
        norm_q = np.linalg.norm(query_embedding)
        norm_e = np.linalg.norm(ent_emb)
        if norm_q == 0 or norm_e == 0:
            return 0.0
        return float(np.dot(query_embedding, ent_emb) / (norm_q * norm_e))

    def get_entity_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Return the embedding vector for a given entity ID."""
        try:
            idx = self.entity_ids.index(entity_id)
            return self.entity_embeds[idx]
        except (ValueError, IndexError):
            return None
