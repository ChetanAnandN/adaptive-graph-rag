"""
modules/evaluation.py — Baseline Comparison and Evaluation (Module 8).

Compares fixed-depth graph retrieval against RL-based adaptive retrieval
using the labelled demo questions.

Metrics
-------
- Number of expanded graph nodes
- Average graph depth
- Retrieval latency (seconds)
- Retrieved-token count
- Supporting-fact recall (coverage of ground-truth supporting entities)
- Estimated retrieval cost (expansion-based)

All metrics are computed from real runs — nothing is invented or hard-coded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import networkx as nx
from stable_baselines3 import PPO

import config
from modules.vector_retriever import VectorRetriever
from modules.adaptive_retriever import (
    rl_retrieve, fixed_depth_retrieve, RetrievalResult,
)
from utils.helpers import normalize_entity_name


@dataclass
class QuestionMetrics:
    """Metrics for a single question under a specific retrieval method."""
    question: str
    method: str
    num_expanded: int = 0
    depth: int = 0
    latency_s: float = 0.0
    token_count: int = 0
    recall: float = 0.0          # supporting-fact coverage
    cost: float = 0.0            # normalised expansion cost


@dataclass
class EvaluationReport:
    """Aggregated evaluation across all demo questions."""
    rl_metrics: List[QuestionMetrics] = field(default_factory=list)
    fixed_metrics: Dict[int, List[QuestionMetrics]] = field(default_factory=dict)
    summary: Dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────

def _compute_recall(expanded_nodes: List[str],
                    supporting_entities: List[str],
                    graph: nx.DiGraph) -> float:
    """
    Fraction of ground-truth supporting entities covered by expanded nodes.
    Uses normalised substring matching.
    """
    if not supporting_entities:
        return 1.0

    expanded_names = set()
    for nid in expanded_nodes:
        if nid in graph:
            expanded_names.add(normalize_entity_name(
                graph.nodes[nid].get('name', nid)))
        expanded_names.add(normalize_entity_name(nid))

    covered = 0
    for sup in supporting_entities:
        sup_norm = normalize_entity_name(sup)
        for en in expanded_names:
            if sup_norm in en or en in sup_norm:
                covered += 1
                break

    return covered / len(supporting_entities)


def evaluate(demo_questions: List[Dict],
             graph: nx.DiGraph,
             retriever: VectorRetriever,
             model: Optional[PPO] = None,
             fixed_depths: List[int] = None,
             ) -> EvaluationReport:
    """
    Run both retrieval methods on every demo question and compute metrics.

    Parameters
    ----------
    model         : trained PPO model (None → skip RL evaluation).
    fixed_depths  : list of BFS depths for the baseline (default: [1, 2, 3]).
    """
    if fixed_depths is None:
        fixed_depths = config.FIXED_DEPTH_LEVELS

    report = EvaluationReport()

    for q in demo_questions:
        question = q["question"]
        supporting = q.get("supporting_entities", [])

        # ── RL retrieval ──────────────────────────────────────
        if model is not None:
            t0 = time.perf_counter()
            rl_result = rl_retrieve(question, graph, retriever, model)
            latency = time.perf_counter() - t0

            recall = _compute_recall(rl_result.expanded_nodes, supporting, graph)
            report.rl_metrics.append(QuestionMetrics(
                question=question,
                method="rl",
                num_expanded=len(rl_result.expanded_nodes),
                depth=rl_result.total_depth,
                latency_s=latency,
                token_count=rl_result.token_estimate,
                recall=recall,
                cost=len(rl_result.expanded_nodes) / max(config.MAX_EXPANSIONS, 1),
            ))

        # ── Fixed-depth baselines ─────────────────────────────
        for depth in fixed_depths:
            t0 = time.perf_counter()
            fd_result = fixed_depth_retrieve(
                question, graph, retriever, depth=depth)
            latency = time.perf_counter() - t0

            recall = _compute_recall(fd_result.expanded_nodes, supporting, graph)
            if depth not in report.fixed_metrics:
                report.fixed_metrics[depth] = []
            report.fixed_metrics[depth].append(QuestionMetrics(
                question=question,
                method=f"fixed_depth_{depth}",
                num_expanded=len(fd_result.expanded_nodes),
                depth=depth,
                latency_s=latency,
                token_count=fd_result.token_estimate,
                recall=recall,
                cost=len(fd_result.expanded_nodes) / max(config.MAX_EXPANSIONS, 1),
            ))

    # ── Summary statistics ────────────────────────────────────
    def _avg(metrics: List[QuestionMetrics], attr: str) -> float:
        vals = [getattr(m, attr) for m in metrics]
        return float(np.mean(vals)) if vals else 0.0

    summary = {}
    if report.rl_metrics:
        summary['rl'] = {
            'avg_expanded': _avg(report.rl_metrics, 'num_expanded'),
            'avg_depth': _avg(report.rl_metrics, 'depth'),
            'avg_latency': _avg(report.rl_metrics, 'latency_s'),
            'avg_tokens': _avg(report.rl_metrics, 'token_count'),
            'avg_recall': _avg(report.rl_metrics, 'recall'),
            'avg_cost': _avg(report.rl_metrics, 'cost'),
        }
    for depth, metrics in report.fixed_metrics.items():
        summary[f'fixed_{depth}'] = {
            'avg_expanded': _avg(metrics, 'num_expanded'),
            'avg_depth': _avg(metrics, 'depth'),
            'avg_latency': _avg(metrics, 'latency_s'),
            'avg_tokens': _avg(metrics, 'token_count'),
            'avg_recall': _avg(metrics, 'recall'),
            'avg_cost': _avg(metrics, 'cost'),
        }

    report.summary = summary
    return report
