# Reinforcement Learning-Based Adaptive Graph Retrieval for RAG

A research prototype demonstrating how a PPO-trained RL agent can **dynamically
decide** how deep to explore a knowledge graph before generating an answer —
balancing retrieval quality against retrieval cost.

---

## Quick Start (Windows)

### 1. Create a virtual environment

```powershell
cd adaptive-graph-rag
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
# Install PyTorch CPU (if you don't have a GPU setup)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install all other requirements
pip install -r requirements.txt
```

### 3. Launch the application

```powershell
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**.

---

## Demo Walkthrough

| Step | Page | Action |
|------|------|--------|
| 1 | **Document Processing** | Click **Load sample documents** |
| 2 | **Knowledge Graph** | Click **Build Knowledge Graph** |
| 3 | **RL Agent** | Click **Train PPO Agent** (≈ 1–3 min) |
| 4 | **Query & Retrieval** | Type a question → **Run RL Retrieval** |
| 5 | **Answer Generation** | Type a question → **Generate Answer** |
| 6 | **Evaluation** | Click **Run Evaluation** to compare methods |

---

## Project Structure

```
adaptive-graph-rag/
│
├── app.py                         # Streamlit frontend (7 pages)
├── config.py                      # All hyper-parameters
├── requirements.txt
├── README.md
│
├── data/
│   ├── sample_documents/          # Built-in corpus (3 documents)
│   │   ├── neural_networks.txt
│   │   ├── machine_learning.txt
│   │   └── natural_language_processing.txt
│   └── demo_questions.json        # Labelled questions for training
│
├── modules/
│   ├── document_processor.py      # PDF / TXT / DOCX extraction & chunking
│   ├── entity_extractor.py        # Rule-based entity & relationship extraction
│   ├── graph_builder.py           # NetworkX graph construction & Pyvis viz
│   ├── vector_retriever.py        # Sentence Transformer embeddings & retrieval
│   ├── rl_environment.py          # Gymnasium env (EXPAND / STOP)
│   ├── train_agent.py             # PPO training loop (Stable-Baselines3)
│   ├── adaptive_retriever.py      # RL-based & fixed-depth retrieval
│   ├── context_builder.py         # Evidence ranking & context construction
│   ├── answer_generator.py        # Ollama LLM + extractive fallback
│   └── evaluation.py              # Baseline comparison & metrics
│
├── models/                        # Saved PPO checkpoints (auto-created)
│
└── utils/
    └── helpers.py                 # Text processing & similarity utilities
```

---

## Module Descriptions

### Module 1 — Document Processor (`document_processor.py`)
Extracts text from PDF (PyMuPDF), TXT, and DOCX (python-docx) files.
Cleans, normalises, and splits text into overlapping word-level chunks.
Each chunk carries a unique ID and its source document name.

### Module 2 — Entity Extractor (`entity_extractor.py`)
Uses capitalisation patterns, acronym detection, and syntactic templates to
extract entities and relationships from text.  No external NER model needed.

### Module 3 — Graph Builder (`graph_builder.py`)
Constructs a NetworkX directed graph from extracted entities (nodes) and
relationships (edges).  Renders an interactive Pyvis visualisation.

### Module 4 — Vector Retriever (`vector_retriever.py`)
Generates embeddings with `all-MiniLM-L6-v2` (Sentence Transformers).
Supports cosine-similarity based retrieval for chunks and entities.

### Module 5 — RL Environment (`rl_environment.py`)
A Gymnasium environment where the agent observes a 7-dimensional state
(query relevance, depth, expansions, candidates, tokens, next-candidate
score, relevance delta) and chooses EXPAND or STOP.

### Module 6 — Train Agent (`train_agent.py`)
Trains PPO (Stable-Baselines3) on the graph-retrieval environment using
labelled demo questions.  Saves/loads checkpoints automatically.

### Module 7 — Adaptive Retriever (`adaptive_retriever.py`)
Rolls out the trained PPO policy at inference time, recording each decision
step.  Also implements a fixed-depth BFS baseline for comparison.

### Module 8 — Context Builder (`context_builder.py`)
Collects expanded nodes, ranks by relevance, deduplicates, and trims to a
configurable token budget to construct a compact context for the LLM.

### Module 9 — Answer Generator (`answer_generator.py`)
Sends the context + question to Ollama (if available).  Falls back to a
clearly labelled extractive answer assembled from retrieved evidence.

### Module 10 — Evaluation (`evaluation.py`)
Runs both retrieval methods on every demo question and computes real
metrics: nodes expanded, depth, latency, token count, and supporting-fact
recall.

---

## RL Formulation

| Component | Detail |
|-----------|--------|
| **State** | 7-D continuous vector (normalised to [0, 1]) |
| **Actions** | `EXPAND` (0) — explore next candidate; `STOP` (1) — terminate |
| **Reward** | `R = α·Q − β·C + Σ(step_penalties)` |
| **Q** | Fraction of ground-truth entities covered (training only) |
| **C** | Normalised expansion cost |
| **Algorithm** | PPO with MLP policy `[64, 64]` |

> Ground-truth supporting entities are used **only in the reward function
> during training**.  The agent's observation vector contains **no**
> ground-truth information.

---

## Optional: Ollama Setup

For LLM-based answer generation, install and run Ollama:

```powershell
# Install from https://ollama.com
ollama pull llama3
ollama serve
```

If Ollama is not running, the system automatically falls back to
extractive answers built from the retrieved evidence.

---

## Configuration

All hyper-parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 300 | Target words per chunk |
| `NUM_INITIAL_ANCHORS` | 3 | Anchor nodes for initial retrieval |
| `ALPHA` | 2.0 | Quality weight in RL reward |
| `BETA` | 0.5 | Cost weight in RL reward |
| `PPO_TOTAL_TIMESTEPS` | 20 000 | Training environment steps |
| `MAX_DEPTH` | 10 | Max exploration depth |
| `MAX_EXPANSIONS` | 20 | Max expanded nodes |

---

## License

Academic research prototype — for educational use.
