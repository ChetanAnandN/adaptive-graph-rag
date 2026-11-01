"""
modules/context_builder.py — Context Construction (Module 6).

After the RL agent chooses STOP, this module:
1.  Collects the selected graph nodes and their relationships.
2.  Ranks evidence by query relevance.
3.  Removes duplicates.
4.  Constructs a compact context string suitable for LLM prompting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from modules.vector_retriever import VectorRetriever
from modules.adaptive_retriever import RetrievalResult
from utils.helpers import estimate_tokens, truncate, normalize_entity_name
import config


@dataclass
class ContextBlock:
    """One piece of evidence assembled for the LLM context."""
    entity_name: str
    description: str
    source_document: str
    chunk_ids: List[str]
    relevance_score: float


@dataclass
class BuiltContext:
    """Full context package ready for answer generation."""
    context_text: str
    entities: List[ContextBlock]
    relationships: List[Dict]
    num_entities: int = 0
    num_relationships: int = 0
    estimated_tokens: int = 0
    source_documents: List[str] = field(default_factory=list)


def build_context(retrieval_result: RetrievalResult,
                  graph: nx.DiGraph,
                  retriever: VectorRetriever,
                  max_tokens: int = config.MAX_CONTEXT_TOKENS,
                  ) -> BuiltContext:
    """
    Construct a compact, ranked context from the retrieval result.

    Steps
    -----
    1. Gather node descriptions, ranked by query relevance.
    2. Gather edges between expanded nodes.
    3. Deduplicate by normalised entity name.
    4. Trim to ``max_tokens`` budget.
    5. Format as a structured text block.
    """
    query = retrieval_result.query
    expanded = retrieval_result.expanded_nodes

    if not expanded:
        return BuiltContext(
            context_text="No evidence was retrieved.",
            entities=[], relationships=[],
        )

    query_emb = retriever.encode(query)

    # ── Collect & rank entities ────────────────────────────────
    blocks: List[ContextBlock] = []
    seen_names = set()
    for nid in expanded:
        if nid not in graph:
            continue
        data = graph.nodes[nid]
        name = data.get('name', nid)
        norm = normalize_entity_name(name)
        if norm in seen_names:
            continue
        seen_names.add(norm)

        score = retriever.get_entity_score(query_emb, nid)
        blocks.append(ContextBlock(
            entity_name=name,
            description=data.get('description', ''),
            source_document=data.get('source_document', 'unknown'),
            chunk_ids=data.get('chunk_ids', []),
            relevance_score=score,
        ))

    # Sort by descending relevance
    blocks.sort(key=lambda b: b.relevance_score, reverse=True)

    # ── Collect relationships between expanded nodes ──────────
    rel_list: List[Dict] = []
    expanded_set = set(expanded)
    seen_edges = set()
    for src, tgt, edata in graph.edges(data=True):
        if src in expanded_set and tgt in expanded_set:
            key = (src, edata.get('relation', ''), tgt)
            if key not in seen_edges:
                seen_edges.add(key)
                src_name = graph.nodes[src].get('name', src)
                tgt_name = graph.nodes[tgt].get('name', tgt)
                rel_list.append({
                    'source': src_name,
                    'target': tgt_name,
                    'relation': edata.get('relation', 'related'),
                    'evidence': edata.get('evidence', ''),
                    'source_document': edata.get('source_document', ''),
                })

    # ── Build context text (trim to budget) ───────────────────
    parts: List[str] = []
    running_tokens = 0

    parts.append("=== Retrieved Entities ===")
    for blk in blocks:
        snippet = f"• {blk.entity_name}: {blk.description}"
        snippet_tokens = estimate_tokens(snippet)
        if running_tokens + snippet_tokens > max_tokens:
            break
        parts.append(snippet)
        running_tokens += snippet_tokens

    if rel_list:
        parts.append("\n=== Relationships ===")
        for r in rel_list:
            snippet = f"• {r['source']} —[{r['relation']}]→ {r['target']}"
            snippet_tokens = estimate_tokens(snippet)
            if running_tokens + snippet_tokens > max_tokens:
                break
            parts.append(snippet)
            running_tokens += snippet_tokens

    context_text = '\n'.join(parts)

    # Source documents
    src_docs = sorted({b.source_document for b in blocks})

    return BuiltContext(
        context_text=context_text,
        entities=blocks,
        relationships=rel_list,
        num_entities=len(blocks),
        num_relationships=len(rel_list),
        estimated_tokens=estimate_tokens(context_text),
        source_documents=src_docs,
    )
