"""
modules/adaptive_retriever.py — RL-Based Adaptive Graph Retrieval (Module 4b).

Uses the trained PPO agent (or a fallback fixed-depth strategy) to perform
graph exploration.  Records every step so the UI can visualise the agent's
EXPAND / STOP decisions and the exploration path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx
from stable_baselines3 import PPO

import config
from modules.rl_environment import GraphRetrievalEnv
from modules.vector_retriever import VectorRetriever
from utils.helpers import normalize_entity_name, estimate_tokens


# ──────────────────────────────────────────────────────────────
# Data-classes for retrieval results
# ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalStep:
    """One step of the RL agent's decision process."""
    step_idx: int
    action: str            # "EXPAND" or "STOP" or "FORCED_STOP"
    node_expanded: Optional[str] = None
    node_name: Optional[str] = None
    relevance_score: float = 0.0
    observation: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """Full output of an adaptive retrieval episode."""
    query: str
    initial_anchors: List[str]
    expanded_nodes: List[str]
    steps: List[RetrievalStep] = field(default_factory=list)
    total_depth: int = 0
    token_estimate: int = 0
    final_relevance: float = 0.0
    method: str = "rl"     # "rl" or "fixed_depth"


# ──────────────────────────────────────────────────────────────
# RL-Based Retrieval
# ──────────────────────────────────────────────────────────────

def rl_retrieve(query: str,
                graph: nx.DiGraph,
                retriever: VectorRetriever,
                model: PPO,
                num_anchors: int = config.NUM_INITIAL_ANCHORS,
                ) -> RetrievalResult:
    """
    Run the trained PPO agent to adaptively retrieve graph evidence.

    1. Find initial anchor nodes via vector similarity.
    2. Set up the RL environment in *inference* mode (no ground truth).
    3. Roll out the policy — the agent decides EXPAND or STOP at each step.
    4. Record every step for the UI.
    """
    # Initial anchors
    anchor_tuples = retriever.retrieve_entities(query, top_k=num_anchors)
    anchor_ids = [a[0] for a in anchor_tuples]

    if not anchor_ids:
        return RetrievalResult(
            query=query, initial_anchors=[], expanded_nodes=[],
            method="rl",
        )

    # Build a lightweight env for inference (no training data needed)
    env = GraphRetrievalEnv(
        graph=graph, retriever=retriever,
        demo_questions=[{"question": query, "supporting_entities": []}],
    )
    obs, _ = env.reset()
    env.set_query(query, anchor_ids)
    obs = env._get_obs()

    steps: List[RetrievalStep] = []
    done = False
    step_idx = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        action_label = "STOP" if action == 1 else "EXPAND"
        if truncated and action != 1:
            action_label = "FORCED_STOP"

        node_exp = None
        node_name = None
        rel_score = 0.0
        if action_label == "EXPAND" and env._history:
            last_act, last_node = env._history[-1]
            if last_node:
                node_exp = last_node
                node_name = graph.nodes[last_node].get('name', last_node) if last_node in graph else last_node
                rel_score = retriever.get_entity_score(env._query_embedding, last_node)

        steps.append(RetrievalStep(
            step_idx=step_idx,
            action=action_label,
            node_expanded=node_exp,
            node_name=node_name,
            relevance_score=rel_score,
            observation=obs.tolist(),
        ))
        step_idx += 1

    expanded = list(env._expanded)
    tok_est = sum(estimate_tokens(graph.nodes[n].get('description', ''))
                  for n in expanded if n in graph)

    return RetrievalResult(
        query=query,
        initial_anchors=anchor_ids,
        expanded_nodes=expanded,
        steps=steps,
        total_depth=env._depth,
        token_estimate=tok_est,
        final_relevance=env._compute_relevance(),
        method="rl",
    )


# ──────────────────────────────────────────────────────────────
# Fixed-Depth Baseline Retrieval
# ──────────────────────────────────────────────────────────────

def fixed_depth_retrieve(query: str,
                         graph: nx.DiGraph,
                         retriever: VectorRetriever,
                         depth: int = 2,
                         num_anchors: int = config.NUM_INITIAL_ANCHORS,
                         ) -> RetrievalResult:
    """
    Baseline: BFS expansion from anchor nodes to a fixed depth.

    Expands *all* neighbours up to ``depth`` hops — no RL decisions.
    """
    anchor_tuples = retriever.retrieve_entities(query, top_k=num_anchors)
    anchor_ids = [a[0] for a in anchor_tuples]

    if not anchor_ids:
        return RetrievalResult(
            query=query, initial_anchors=[], expanded_nodes=[],
            method="fixed_depth",
        )

    expanded: Set[str] = set(anchor_ids)
    frontier: Set[str] = set(anchor_ids)
    steps: List[RetrievalStep] = []
    step_idx = 0

    for d in range(depth):
        next_frontier: Set[str] = set()
        for node in frontier:
            neighbours = set()
            if node in graph:
                neighbours.update(graph.successors(node))
                neighbours.update(graph.predecessors(node))
            for nbr in neighbours:
                if nbr not in expanded:
                    expanded.add(nbr)
                    next_frontier.add(nbr)
                    name = graph.nodes[nbr].get('name', nbr) if nbr in graph else nbr
                    steps.append(RetrievalStep(
                        step_idx=step_idx,
                        action="EXPAND",
                        node_expanded=nbr,
                        node_name=name,
                        relevance_score=retriever.get_entity_score(
                            retriever.encode(query), nbr),
                    ))
                    step_idx += 1
        frontier = next_frontier
        if not frontier:
            break

    steps.append(RetrievalStep(step_idx=step_idx, action="STOP"))

    expanded_list = list(expanded)
    query_emb = retriever.encode(query)
    tok_est = sum(estimate_tokens(graph.nodes[n].get('description', ''))
                  for n in expanded_list if n in graph)
    relevance_scores = [retriever.get_entity_score(query_emb, n)
                        for n in expanded_list]
    avg_rel = float(np.mean(relevance_scores)) if relevance_scores else 0.0

    return RetrievalResult(
        query=query,
        initial_anchors=anchor_ids,
        expanded_nodes=expanded_list,
        steps=steps,
        total_depth=depth,
        token_estimate=tok_est,
        final_relevance=avg_rel,
        method="fixed_depth",
    )
