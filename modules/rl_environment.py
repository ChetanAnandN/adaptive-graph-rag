"""
modules/rl_environment.py — Reinforcement Learning Environment (Modules 4 & 5).

A Gymnasium-compatible environment for adaptive graph retrieval.

The RL agent decides whether to EXPAND the retrieval frontier (explore a
neighbouring node) or STOP (terminate exploration and send evidence to the
answer generator).  The reward balances *retrieval quality* against
*retrieval cost*.

Observation Space (7 continuous features, all normalised to ≈ [0, 1])
----------------------------------------------------------------------
0. query_relevance        – cosine similarity of aggregated evidence to query
1. norm_depth             – current depth / MAX_DEPTH
2. norm_expanded          – # expanded nodes / MAX_EXPANSIONS
3. norm_candidates        – # candidate nodes / 10  (capped at 1)
4. norm_token_count       – estimated tokens / MAX_CONTEXT_TOKENS
5. next_candidate_score   – relevance of the best available candidate
6. relevance_delta        – change in relevance after last expansion (clipped)

Action Space
------------
0 = EXPAND   — add the most-relevant neighbouring candidate to the evidence set
1 = STOP     — end exploration; current evidence is forwarded downstream

Reward (calculated at the terminal step)
-----------------------------------------
R = α × Q  −  β × C  +  Σ(step_penalties)

Q = supporting-evidence coverage (fraction of ground-truth entities covered)
C = normalised expansion cost = # expanded / MAX_EXPANSIONS

NOTE: Ground-truth supporting entities are used **only** inside the reward
      computation during training.  They are **never** part of the agent's
      observation vector.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx

import config
from utils.helpers import normalize_entity_name, estimate_tokens
from modules.vector_retriever import VectorRetriever


class GraphRetrievalEnv(gym.Env):
    """
    Gymnasium env for RL-based adaptive graph retrieval.

    Parameters
    ----------
    graph           : The knowledge graph (NetworkX DiGraph).
    retriever       : VectorRetriever (already indexed).
    demo_questions  : list of dicts with 'question' and 'supporting_entities'.
    alpha           : quality weight in reward.
    beta            : cost weight in reward.
    max_depth       : hard ceiling on exploration depth.
    max_expansions  : hard ceiling on total expanded nodes.
    max_tokens      : budget for estimated context tokens.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 graph: nx.DiGraph,
                 retriever: VectorRetriever,
                 demo_questions: List[Dict],
                 alpha: float = config.ALPHA,
                 beta: float = config.BETA,
                 max_depth: int = config.MAX_DEPTH,
                 max_expansions: int = config.MAX_EXPANSIONS,
                 max_tokens: int = config.MAX_CONTEXT_TOKENS):
        super().__init__()

        self.graph = graph
        self.retriever = retriever
        self.demo_questions = demo_questions

        # Reward parameters
        self.alpha = alpha
        self.beta = beta
        self.step_penalty = config.STEP_PENALTY

        # Exploration limits
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        self.max_tokens = max_tokens

        # ── Spaces ─────────────────────────────────────────────
        self.action_space = spaces.Discrete(2)            # EXPAND=0, STOP=1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(config.RL_STATE_DIM,),
            dtype=np.float32,
        )

        # ── Episode state (set in reset) ──────────────────────
        self._query_text: str = ""
        self._query_embedding: Optional[np.ndarray] = None
        self._supporting: Set[str] = set()     # ground-truth (reward only)
        self._expanded: Set[str] = set()        # already-explored nodes
        self._candidates: List[Tuple[str, float]] = []  # (node_id, score)
        self._depth: int = 0
        self._prev_relevance: float = 0.0
        self._relevance_delta: float = 0.0
        self._cumulative_penalty: float = 0.0
        self._history: List[Tuple[str, Optional[str]]] = []

    # ──────────────────────────────────────────────────────────
    #  Gymnasium API
    # ──────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Sample a demo question
        idx = self.np_random.integers(0, len(self.demo_questions))
        q = self.demo_questions[idx]

        self._query_text = q["question"]
        self._query_embedding = self.retriever.encode(self._query_text)

        # Ground-truth supporting entities (normalised) — reward only
        self._supporting = {
            normalize_entity_name(e) for e in q.get("supporting_entities", [])
        }

        # Initial anchor nodes (top-k by embedding similarity)
        anchors = self.retriever.retrieve_entities(
            self._query_text, top_k=config.NUM_INITIAL_ANCHORS
        )
        self._expanded = {a[0] for a in anchors}
        if not self._expanded:
            # Fallback: pick random nodes if retriever found nothing
            nodes = list(self.graph.nodes)
            if nodes:
                chosen = self.np_random.choice(
                    nodes, size=min(config.NUM_INITIAL_ANCHORS, len(nodes)),
                    replace=False)
                self._expanded = set(chosen)

        self._depth = 0
        self._prev_relevance = self._compute_relevance()
        self._relevance_delta = 0.0
        self._cumulative_penalty = 0.0
        self._history = []
        self._update_candidates()

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        truncated = False

        if action == 0:  # ── EXPAND ─────────────────────────
            if self._candidates:
                best_id, best_score = self._candidates[0]
                self._expanded.add(best_id)
                self._depth += 1
                self._update_candidates()

                new_rel = self._compute_relevance()
                self._relevance_delta = new_rel - self._prev_relevance
                self._prev_relevance = new_rel
                self._cumulative_penalty += self.step_penalty
                self._history.append(("EXPAND", best_id))
            else:
                # No candidates left → force STOP
                action = 1

        if action == 1:  # ── STOP ───────────────────────────
            self._history.append(("STOP", None))
            reward = self._compute_reward()
            return self._get_obs(), reward, True, False, self._build_info()

        # Check hard limits (→ forced termination)
        if (self._depth >= self.max_depth or
                len(self._expanded) >= self.max_expansions or
                self._estimate_token_count() >= self.max_tokens):
            self._history.append(("FORCED_STOP", None))
            reward = self._compute_reward()
            return self._get_obs(), reward, False, True, self._build_info()

        return self._get_obs(), 0.0, False, False, self._build_info()

    # ──────────────────────────────────────────────────────────
    #  Observation
    # ──────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        relevance = self._compute_relevance()
        num_cand = len(self._candidates)
        next_score = self._candidates[0][1] if self._candidates else 0.0
        token_frac = self._estimate_token_count() / max(self.max_tokens, 1)

        obs = np.array([
            np.clip(relevance, 0.0, 1.0),                        # 0
            np.clip(self._depth / max(self.max_depth, 1), 0, 1), # 1
            np.clip(len(self._expanded) / max(self.max_expansions, 1), 0, 1),  # 2
            np.clip(num_cand / 10.0, 0, 1),                      # 3
            np.clip(token_frac, 0, 1),                            # 4
            np.clip(next_score, 0, 1),                            # 5
            np.clip(self._relevance_delta, -1, 1),                # 6
        ], dtype=np.float32)
        return obs

    # ──────────────────────────────────────────────────────────
    #  Internals
    # ──────────────────────────────────────────────────────────

    def _update_candidates(self):
        """Refresh the sorted candidate list (neighbours of expanded set)."""
        candidates: Dict[str, float] = {}
        for node in self._expanded:
            for nbr in self._get_graph_neighbors(node):
                if nbr not in self._expanded and nbr not in candidates:
                    score = self.retriever.get_entity_score(
                        self._query_embedding, nbr)
                    candidates[nbr] = score
        # Sort by descending relevance
        self._candidates = sorted(candidates.items(),
                                  key=lambda x: x[1], reverse=True)

    def _get_graph_neighbors(self, node_id: str) -> List[str]:
        """Successors + predecessors (undirected neighbourhood)."""
        nbrs = set()
        if node_id in self.graph:
            nbrs.update(self.graph.successors(node_id))
            nbrs.update(self.graph.predecessors(node_id))
        nbrs.discard(node_id)
        return list(nbrs)

    def _compute_relevance(self) -> float:
        """Average cosine similarity of expanded entities to query."""
        if not self._expanded or self._query_embedding is None:
            return 0.0
        scores = []
        for nid in self._expanded:
            s = self.retriever.get_entity_score(self._query_embedding, nid)
            scores.append(s)
        return float(np.mean(scores)) if scores else 0.0

    def _estimate_token_count(self) -> int:
        """Estimated tokens in the evidence from expanded nodes."""
        total = 0
        for nid in self._expanded:
            desc = self.graph.nodes[nid].get('description', '') if nid in self.graph else ''
            total += estimate_tokens(desc)
        return total

    def _compute_reward(self) -> float:
        """
        Terminal reward:  R = α × Q  −  β × C  + cumulative step penalties.

        Q = fraction of ground-truth supporting entities covered.
        C = expansion cost (normalised).
        """
        # ── Quality: coverage of supporting entities ──────────
        if not self._supporting:
            quality = 1.0  # no ground truth → assume full quality
        else:
            covered = 0
            expanded_names = {normalize_entity_name(
                self.graph.nodes[n].get('name', n))
                for n in self._expanded if n in self.graph}
            for sup in self._supporting:
                # Check exact and substring match
                for en in expanded_names:
                    if sup in en or en in sup:
                        covered += 1
                        break
            quality = covered / len(self._supporting)

        # ── Cost: normalised expansion count ──────────────────
        cost = len(self._expanded) / max(self.max_expansions, 1)

        reward = self.alpha * quality - self.beta * cost + self._cumulative_penalty
        return float(reward)

    def _build_info(self) -> Dict[str, Any]:
        return {
            'expanded_nodes': list(self._expanded),
            'depth': self._depth,
            'history': list(self._history),
            'num_candidates': len(self._candidates),
            'token_estimate': self._estimate_token_count(),
        }

    # ──────────────────────────────────────────────────────────
    #  Inference helper (used outside training)
    # ──────────────────────────────────────────────────────────

    def set_query(self, query: str, initial_anchors: List[str]):
        """
        Configure the environment for inference with a specific query
        and pre-computed anchors (bypasses demo-question sampling).
        """
        self._query_text = query
        self._query_embedding = self.retriever.encode(query)
        self._supporting = set()          # no ground truth at inference
        self._expanded = set(initial_anchors)
        self._depth = 0
        self._prev_relevance = self._compute_relevance()
        self._relevance_delta = 0.0
        self._cumulative_penalty = 0.0
        self._history = []
        self._update_candidates()
