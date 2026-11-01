"""
modules/train_agent.py — PPO Training Loop (Module 5 continued).

Provides helpers for:
- Creating the RL environment.
- Training a PPO policy with Stable-Baselines3.
- Saving / loading checkpoints.
- Tracking training metrics for display.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import config
from modules.rl_environment import GraphRetrievalEnv
from modules.vector_retriever import VectorRetriever


# ──────────────────────────────────────────────────────────────
# Training Callback — collects per-episode metrics
# ──────────────────────────────────────────────────────────────

class TrainingMetricsCallback(BaseCallback):
    """
    Custom SB3 callback that records episode rewards and lengths.
    These metrics are displayed on the RL Agent page after training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self._current_rewards: float = 0.0
        self._current_length: int = 0

    def _on_step(self) -> bool:
        # Accumulate within-episode reward
        # SB3 stores rewards in self.locals['rewards']
        # infos contains dones
        infos = self.locals.get('infos', [{}])
        rewards = self.locals.get('rewards', [0.0])
        dones = self.locals.get('dones', [False])

        for i, done in enumerate(dones):
            self._current_rewards += rewards[i]
            self._current_length += 1
            if done:
                self.episode_rewards.append(self._current_rewards)
                self.episode_lengths.append(self._current_length)
                self._current_rewards = 0.0
                self._current_length = 0

        return True  # continue training


# ──────────────────────────────────────────────────────────────
# Training Helpers
# ──────────────────────────────────────────────────────────────

def load_demo_questions() -> List[Dict]:
    """Load labelled demo questions from the data directory."""
    if not os.path.isfile(config.DEMO_QUESTIONS_PATH):
        return []
    with open(config.DEMO_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_training_env(graph: nx.DiGraph,
                        retriever: VectorRetriever,
                        demo_questions: Optional[List[Dict]] = None
                        ) -> GraphRetrievalEnv:
    """Instantiate a ``GraphRetrievalEnv`` ready for training."""
    if demo_questions is None:
        demo_questions = load_demo_questions()
    if not demo_questions:
        raise ValueError("No demo questions available for training.")

    env = GraphRetrievalEnv(
        graph=graph,
        retriever=retriever,
        demo_questions=demo_questions,
        alpha=config.ALPHA,
        beta=config.BETA,
        max_depth=config.MAX_DEPTH,
        max_expansions=config.MAX_EXPANSIONS,
        max_tokens=config.MAX_CONTEXT_TOKENS,
    )
    return env


def train_ppo(graph: nx.DiGraph,
              retriever: VectorRetriever,
              total_timesteps: int = config.PPO_TOTAL_TIMESTEPS,
              demo_questions: Optional[List[Dict]] = None,
              ) -> Tuple[PPO, TrainingMetricsCallback]:
    """
    Train a PPO agent on the graph-retrieval environment.

    Returns
    -------
    (model, callback) — the trained PPO model and the metrics callback.
    """
    env = create_training_env(graph, retriever, demo_questions)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.PPO_LEARNING_RATE,
        n_steps=config.PPO_N_STEPS,
        batch_size=config.PPO_BATCH_SIZE,
        n_epochs=config.PPO_N_EPOCHS,
        gamma=config.PPO_GAMMA,
        gae_lambda=config.PPO_GAE_LAMBDA,
        clip_range=config.PPO_CLIP_RANGE,
        policy_kwargs=config.PPO_POLICY_KWARGS,
        verbose=1,
        seed=42,
    )

    callback = TrainingMetricsCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save checkpoint
    save_checkpoint(model)

    return model, callback


def save_checkpoint(model: PPO, path: Optional[str] = None) -> str:
    """Persist the PPO model to disk."""
    path = path or config.PPO_CHECKPOINT_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    return path


def load_checkpoint(path: Optional[str] = None) -> Optional[PPO]:
    """Load a previously saved PPO checkpoint.  Returns None if missing."""
    path = path or config.PPO_CHECKPOINT_PATH
    zip_path = path if path.endswith('.zip') else path + '.zip'
    if not os.path.isfile(zip_path):
        return None
    return PPO.load(path)


def is_trained() -> bool:
    """Check whether a PPO checkpoint exists on disk."""
    path = config.PPO_CHECKPOINT_PATH
    zip_path = path if path.endswith('.zip') else path + '.zip'
    return os.path.isfile(zip_path)
