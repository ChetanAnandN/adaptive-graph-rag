"""
modules/graph_builder.py — Knowledge Graph Construction (Module 2b).

Builds a NetworkX DiGraph from extracted entities and relationships.
Each node stores entity metadata; each edge stores the relationship type
and source evidence.  Also provides visualisation helpers (Pyvis HTML and
Matplotlib fallback).
"""

from __future__ import annotations

import os
import json
import tempfile
from typing import List, Dict, Optional, Tuple

import networkx as nx

from modules.entity_extractor import Entity, Relationship
from utils.helpers import normalize_entity_name


# ──────────────────────────────────────────────────────────────
# Graph Construction
# ──────────────────────────────────────────────────────────────

def build_knowledge_graph(entities: List[Entity],
                          relationships: List[Relationship]) -> nx.DiGraph:
    """
    Construct a directed knowledge graph.

    Node attributes
    ----------------
    - name          : display name of the entity
    - description   : first sentence mentioning the entity
    - source_document: originating file
    - chunk_ids     : list of chunk identifiers
    - entity_type   : concept | method | metric | model | acronym

    Edge attributes
    ----------------
    - relation      : relationship label (is_a, uses, contains, …)
    - evidence      : sentence where the relationship was observed
    - chunk_id      : originating chunk
    - source_document: originating file
    """
    G = nx.DiGraph()

    # ── Add nodes ──────────────────────────────────────────────
    entity_lookup: Dict[str, str] = {}  # normalised → canonical name
    for ent in entities:
        nid = ent.normalized
        entity_lookup[nid] = ent.name
        G.add_node(nid,
                   name=ent.name,
                   description=ent.description,
                   source_document=ent.source_document,
                   chunk_ids=ent.chunk_ids,
                   entity_type=ent.entity_type)

    # ── Add edges ──────────────────────────────────────────────
    for rel in relationships:
        src = normalize_entity_name(rel.source)
        tgt = normalize_entity_name(rel.target)

        # Only add edge if both endpoints exist as nodes
        if src in G.nodes and tgt in G.nodes and src != tgt:
            G.add_edge(src, tgt,
                       relation=rel.relation,
                       evidence=rel.evidence,
                       chunk_id=rel.chunk_id,
                       source_document=rel.source_document)

    return G


def get_graph_stats(G: nx.DiGraph) -> Dict:
    """Return basic statistics about the knowledge graph."""
    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'num_connected_components': nx.number_weakly_connected_components(G),
        'avg_degree': sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1),
    }


def get_node_info(G: nx.DiGraph, node_id: str) -> Optional[Dict]:
    """Return attribute dict for a node, or None if not found."""
    if node_id in G.nodes:
        return dict(G.nodes[node_id])
    return None


def get_neighbors(G: nx.DiGraph, node_id: str) -> List[str]:
    """Return successors + predecessors (treat as undirected neighbors)."""
    neighbours = set()
    if node_id in G:
        neighbours.update(G.successors(node_id))
        neighbours.update(G.predecessors(node_id))
    neighbours.discard(node_id)
    return list(neighbours)


def get_edge_info(G: nx.DiGraph, src: str, tgt: str) -> Optional[Dict]:
    """Return edge data between two nodes."""
    if G.has_edge(src, tgt):
        return dict(G.edges[src, tgt])
    if G.has_edge(tgt, src):
        return dict(G.edges[tgt, src])
    return None


# ──────────────────────────────────────────────────────────────
# Visualization — Pyvis (interactive HTML)
# ──────────────────────────────────────────────────────────────

def graph_to_pyvis_html(G: nx.DiGraph, height: str = "600px") -> str:
    """
    Render the knowledge graph as an interactive HTML string via Pyvis.
    Falls back to a matplotlib-based PNG if pyvis is unavailable.
    """
    try:
        from pyvis.network import Network

        net = Network(height=height, width="100%", directed=True,
                      bgcolor="#0e1117", font_color="#fafafa")
        net.barnes_hut(gravity=-3000, central_gravity=0.3,
                       spring_length=150, spring_strength=0.01)

        # Colour palette for entity types
        _COLORS = {
            'concept': '#6366f1',   # indigo
            'method': '#22d3ee',    # cyan
            'metric': '#f59e0b',    # amber
            'model': '#ec4899',     # pink
            'acronym': '#a78bfa',   # violet
        }

        for node_id, data in G.nodes(data=True):
            label = data.get('name', node_id)
            etype = data.get('entity_type', 'concept')
            color = _COLORS.get(etype, '#6366f1')
            title = (f"<b>{label}</b><br>"
                     f"Type: {etype}<br>"
                     f"Source: {data.get('source_document', 'N/A')}<br>"
                     f"Chunks: {', '.join(data.get('chunk_ids', []))}<br><br>"
                     f"{data.get('description', '')[:200]}")
            net.add_node(node_id, label=label, color=color,
                         title=title, size=20)

        _EDGE_COLORS = {
            'is_a': '#94a3b8',
            'is_type_of': '#94a3b8',
            'uses': '#22d3ee',
            'contains': '#f59e0b',
            'produces': '#10b981',
            'mitigates': '#ef4444',
            'extends': '#8b5cf6',
            'based_on': '#06b6d4',
            'used_for': '#f97316',
            'combines': '#d946ef',
            'co_occurs': '#475569',
        }

        for src, tgt, data in G.edges(data=True):
            rel = data.get('relation', 'related')
            color = _EDGE_COLORS.get(rel, '#475569')
            title = (f"<b>{rel}</b><br>"
                     f"Evidence: {data.get('evidence', '')[:150]}")
            net.add_edge(src, tgt, title=title, label=rel,
                         color=color, arrows='to')

        # Generate HTML string
        html = net.generate_html()
        # Patch: ensure the HTML works in an iframe
        html = html.replace("'", '"')
        return html

    except ImportError:
        return _graph_to_matplotlib_html(G)


def _graph_to_matplotlib_html(G: nx.DiGraph) -> str:
    """Fallback: render graph with matplotlib and return an <img> tag."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#6366f1',
                           node_size=300, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: G.nodes[n].get('name', n)[:20] for n in G},
                            font_size=7, font_color='white')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#475569',
                           arrows=True, arrowsize=10, alpha=0.6)

    ax.set_axis_off()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="width:100%"/>'
