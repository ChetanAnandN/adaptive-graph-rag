"""
config.py — Central configuration for the Adaptive Graph RAG system.

All tuneable hyper-parameters live here so they can be adjusted from a single
location without touching module code.
"""

import os

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAMPLE_DOCS_DIR = os.path.join(DATA_DIR, "sample_documents")
DEMO_QUESTIONS_PATH = os.path.join(DATA_DIR, "demo_questions.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PPO_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "ppo_graph_retriever")

# ──────────────────────────────────────────────────────────────
# Document Processing
# ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 300          # Target words per chunk
CHUNK_OVERLAP = 50        # Overlap words between consecutive chunks
MIN_CHUNK_LENGTH = 30     # Minimum words to keep a chunk

# ──────────────────────────────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────
NUM_INITIAL_ANCHORS = 3   # Number of initial anchor nodes retrieved
SIMILARITY_THRESHOLD = 0.2  # Minimum cosine similarity to consider

# ──────────────────────────────────────────────────────────────
# Knowledge Graph
# ──────────────────────────────────────────────────────────────
MAX_ENTITY_NAME_LENGTH = 80
MIN_ENTITY_NAME_LENGTH = 2

# ──────────────────────────────────────────────────────────────
# Reinforcement Learning — Environment
# ──────────────────────────────────────────────────────────────
RL_STATE_DIM = 7          # Observation dimensions (see rl_environment.py)
RL_ACTION_DIM = 2         # 0 = EXPAND, 1 = STOP
MAX_DEPTH = 10            # Maximum exploration depth before forced stop
MAX_EXPANSIONS = 20       # Maximum nodes the agent may expand
MAX_CONTEXT_TOKENS = 2048 # Budget cap for retrieved-token estimate

# ──────────────────────────────────────────────────────────────
# Reinforcement Learning — Reward
# ──────────────────────────────────────────────────────────────
ALPHA = 2.0               # Weight for retrieval quality (coverage)
BETA = 0.5                # Weight for retrieval cost (expansions)
STEP_PENALTY = -0.02      # Small per-step cost to discourage idle looping

# ──────────────────────────────────────────────────────────────
# Reinforcement Learning — PPO Training
# ──────────────────────────────────────────────────────────────
PPO_TOTAL_TIMESTEPS = 20_000   # Total environment steps during training
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 256              # Steps before each PPO update
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_POLICY_KWARGS = dict(net_arch=[64, 64])  # Two hidden layers

# ──────────────────────────────────────────────────────────────
# Fixed-Depth Baseline
# ──────────────────────────────────────────────────────────────
FIXED_DEPTH_LEVELS = [1, 2, 3]  # BFS depths compared in evaluation

# ──────────────────────────────────────────────────────────────
# Answer Generation
# ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 30       # seconds
