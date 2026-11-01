"""
app.py — Streamlit Frontend for Adaptive Graph RAG.

Seven-page academic dashboard:
  1. Overview          – project description, workflow, training status
  2. Document Processing – upload / load docs, preprocessing, chunks
  3. Knowledge Graph    – interactive graph visualisation
  4. Query & Retrieval  – question input, anchor nodes, similarity scores
  5. RL Agent           – training, policy status, EXPAND/STOP decisions
  6. Answer Generation  – evidence, context, generated answer
  7. Evaluation         – fixed-depth vs RL comparison tables & charts
"""

from __future__ import annotations

import json
import os
import sys
import time

import streamlit as st
import numpy as np
import pandas as pd

# ── Ensure project root is on the path ─────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.document_processor import (
    load_sample_documents, process_uploaded_file, TextChunk, ProcessedDocument,
)
from modules.entity_extractor import extract_from_chunks
from modules.graph_builder import (
    build_knowledge_graph, get_graph_stats, graph_to_pyvis_html,
    get_node_info, get_neighbors,
)
from modules.vector_retriever import VectorRetriever
from modules.train_agent import (
    train_ppo, load_checkpoint, is_trained, load_demo_questions,
)
from modules.adaptive_retriever import rl_retrieve, fixed_depth_retrieve
from modules.context_builder import build_context
from modules.answer_generator import generate_answer, check_ollama_status
from modules.evaluation import evaluate


# ══════════════════════════════════════════════════════════════
#  Page Config & Styling
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Adaptive Graph RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ──────────────────────────────────────────
st.markdown("""
<style>
/* ── Base overrides ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label {
    color: #e2e8f0 !important;
}
.stApp { background-color: #0f172a; }

/* ── Metric cards ───────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 16px;
}
div[data-testid="stMetric"] label { color: #94a3b8 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
}

/* ── Status badges ──────────────────────────────────────── */
.badge-trained {
    display: inline-block;
    background: #065f46; color: #6ee7b7;
    padding: 4px 14px; border-radius: 20px;
    font-weight: 600; font-size: 0.85rem;
}
.badge-untrained {
    display: inline-block;
    background: #7f1d1d; color: #fca5a5;
    padding: 4px 14px; border-radius: 20px;
    font-weight: 600; font-size: 0.85rem;
}

/* ── Section headers ────────────────────────────────────── */
.section-header {
    border-left: 4px solid #6366f1;
    padding-left: 12px;
    margin: 24px 0 12px 0;
}

/* ── Step cards ──────────────────────────────────────────── */
.step-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.expand-action { border-left: 3px solid #22d3ee; }
.stop-action   { border-left: 3px solid #f59e0b; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  Session State Initialisation
# ══════════════════════════════════════════════════════════════

_DEFAULTS = {
    'documents': [],
    'all_chunks': [],
    'entities': [],
    'relationships': [],
    'graph': None,
    'retriever': None,
    'rl_model': None,
    'rl_trained': False,
    'training_callback': None,
    'retrieval_result': None,
    'context': None,
    'answer': None,
    'eval_report': None,
    'demo_questions': None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.demo_questions is None:
    st.session_state.demo_questions = load_demo_questions()


# ══════════════════════════════════════════════════════════════
#  Sidebar Navigation
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔬 Adaptive Graph RAG")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📄 Document Processing",
         "🕸️ Knowledge Graph",
         "🔍 Query & Retrieval",
         "🤖 RL Agent",
         "💬 Answer Generation",
         "📊 Evaluation"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Quick status indicators
    n_docs = len(st.session_state.documents)
    n_chunks = len(st.session_state.all_chunks)
    g = st.session_state.graph
    n_nodes = g.number_of_nodes() if g else 0
    n_edges = g.number_of_edges() if g else 0

    st.markdown("##### Pipeline Status")
    st.markdown(f"📄 Documents: **{n_docs}**")
    st.markdown(f"📦 Chunks: **{n_chunks}**")
    st.markdown(f"🕸️ Nodes / Edges: **{n_nodes}** / **{n_edges}**")
    trained = st.session_state.rl_trained or is_trained()
    if trained:
        st.markdown('<span class="badge-trained">✓ Agent Trained</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-untrained">✗ Not Trained</span>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  Page 1: Overview
# ══════════════════════════════════════════════════════════════

def page_overview():
    st.markdown("# 🔬 RL-Based Adaptive Graph Retrieval for RAG")
    st.markdown(
        "A reinforcement-learning prototype that dynamically decides **how "
        "deep** to explore a knowledge graph before generating an answer."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", len(st.session_state.documents))
    col2.metric("Chunks", len(st.session_state.all_chunks))
    g = st.session_state.graph
    col3.metric("Graph Nodes", g.number_of_nodes() if g else 0)
    trained = st.session_state.rl_trained or is_trained()
    col4.metric("RL Status", "Trained ✓" if trained else "Not trained")

    st.markdown("---")
    st.markdown(
        '<div class="section-header"><h3>System Workflow</h3></div>',
        unsafe_allow_html=True,
    )

    steps = [
        ("📄", "Document Upload", "Upload PDF / TXT / DOCX documents"),
        ("⚙️", "Preprocessing", "Clean, normalise, and chunk text"),
        ("🕸️", "Graph Construction", "Extract entities & relationships → NetworkX"),
        ("📐", "Vector Indexing", "Embed chunks & entities (all-MiniLM-L6-v2)"),
        ("🔍", "Initial Retrieval", "Cosine-similarity anchor-node selection"),
        ("🤖", "RL Exploration", "PPO agent decides EXPAND or STOP"),
        ("📋", "Context Building", "Rank, deduplicate, trim evidence"),
        ("💬", "Answer Generation", "Ollama LLM or extractive fallback"),
    ]

    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(
                f'<div class="step-card">'
                f'<span style="font-size:1.6rem">{icon}</span><br>'
                f'<b>{title}</b><br>'
                f'<span style="color:#94a3b8;font-size:0.85rem">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        '<div class="section-header"><h3>RL Formulation</h3></div>',
        unsafe_allow_html=True,
    )
    st.latex(r"R = \alpha \cdot Q - \beta \cdot C")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "| Symbol | Meaning |\n"
            "|--------|---------|\n"
            "| Q | Retrieval quality (evidence coverage) |\n"
            "| C | Retrieval cost (normalised expansions) |\n"
            f"| α | Relevance weight = **{config.ALPHA}** |\n"
            f"| β | Cost weight = **{config.BETA}** |"
        )
    with c2:
        st.markdown(
            "**Observation vector (7-D):**\n"
            "1. Query relevance\n"
            "2. Normalised depth\n"
            "3. Normalised expansions\n"
            "4. Candidate count\n"
            "5. Token-budget fraction\n"
            "6. Next-candidate relevance\n"
            "7. Relevance delta"
        )


# ══════════════════════════════════════════════════════════════
#  Page 2: Document Processing
# ══════════════════════════════════════════════════════════════

def page_documents():
    st.markdown("# 📄 Document Processing")

    tab_upload, tab_sample = st.tabs(["Upload Documents", "Load Sample Corpus"])

    # ── Upload tab ─────────────────────────────────────────────
    with tab_upload:
        files = st.file_uploader(
            "Upload PDF, TXT, or DOCX files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
        )
        if files and st.button("Process uploaded files", key="proc_upload"):
            with st.spinner("Extracting and chunking…"):
                for f in files:
                    try:
                        doc = process_uploaded_file(f)
                        st.session_state.documents.append(doc)
                        st.session_state.all_chunks.extend(doc.chunks)
                        st.success(f"✅ {doc.filename}: {len(doc.chunks)} chunks")
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")

    # ── Sample corpus tab ──────────────────────────────────────
    with tab_sample:
        st.markdown(
            "Load the built-in sample documents (Neural Networks, Machine "
            "Learning, NLP)."
        )
        if st.button("Load sample documents", key="load_sample"):
            with st.spinner("Loading and chunking sample corpus…"):
                docs = load_sample_documents()
                if not docs:
                    st.error("No sample documents found.")
                else:
                    st.session_state.documents = docs
                    st.session_state.all_chunks = [
                        c for d in docs for c in d.chunks
                    ]
                    for d in docs:
                        st.success(f"✅ {d.filename}: {len(d.chunks)} chunks")

    # ── Display chunks ─────────────────────────────────────────
    if st.session_state.all_chunks:
        st.markdown("---")
        st.markdown(
            '<div class="section-header"><h3>Extracted Chunks</h3></div>',
            unsafe_allow_html=True,
        )
        chunks = st.session_state.all_chunks
        df = pd.DataFrame([
            {
                "Chunk ID": c.chunk_id,
                "Source": c.source_document,
                "Words": c.word_count,
                "Preview": c.text[:120] + "…",
            }
            for c in chunks
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.info(f"Total chunks: **{len(chunks)}** across "
                f"**{len(st.session_state.documents)}** documents")


# ══════════════════════════════════════════════════════════════
#  Page 3: Knowledge Graph
# ══════════════════════════════════════════════════════════════

def page_knowledge_graph():
    st.markdown("# 🕸️ Knowledge Graph")

    if not st.session_state.all_chunks:
        st.warning("⚠️ No chunks available. Go to **Document Processing** first.")
        return

    if st.button("🔨 Build Knowledge Graph", key="build_graph"):
        with st.spinner("Extracting entities and relationships…"):
            entities, relationships = extract_from_chunks(
                st.session_state.all_chunks)
            st.session_state.entities = entities
            st.session_state.relationships = relationships

        with st.spinner("Constructing graph…"):
            G = build_knowledge_graph(entities, relationships)
            st.session_state.graph = G

        with st.spinner("Building vector index…"):
            retriever = VectorRetriever()
            retriever.build_index(st.session_state.all_chunks, G)
            st.session_state.retriever = retriever

        st.success(
            f"✅ Graph built: **{G.number_of_nodes()}** nodes, "
            f"**{G.number_of_edges()}** edges"
        )

    G = st.session_state.graph
    if G is None:
        st.info("Click **Build Knowledge Graph** to proceed.")
        return

    # ── Stats ──────────────────────────────────────────────────
    stats = get_graph_stats(G)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", stats['num_nodes'])
    c2.metric("Edges", stats['num_edges'])
    c3.metric("Density", f"{stats['density']:.4f}")
    c4.metric("Components", stats['num_connected_components'])

    # ── Visualisation ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header"><h3>Graph Visualisation</h3></div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Rendering graph…"):
        html = graph_to_pyvis_html(G, height="550px")
        st.components.v1.html(html, height=580, scrolling=True)

    # ── Entity list ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header"><h3>Entities</h3></div>',
        unsafe_allow_html=True,
    )
    ent_data = []
    for nid, data in G.nodes(data=True):
        ent_data.append({
            'Entity': data.get('name', nid),
            'Type': data.get('entity_type', ''),
            'Source': data.get('source_document', ''),
            'Chunks': ', '.join(data.get('chunk_ids', [])[:3]),
        })
    if ent_data:
        st.dataframe(pd.DataFrame(ent_data), use_container_width=True,
                      hide_index=True)

    # ── Relationship list ──────────────────────────────────────
    st.markdown(
        '<div class="section-header"><h3>Relationships</h3></div>',
        unsafe_allow_html=True,
    )
    edge_data = []
    for src, tgt, data in G.edges(data=True):
        src_name = G.nodes[src].get('name', src)
        tgt_name = G.nodes[tgt].get('name', tgt)
        edge_data.append({
            'Source': src_name,
            'Relation': data.get('relation', ''),
            'Target': tgt_name,
            'Evidence': data.get('evidence', '')[:100],
        })
    if edge_data:
        st.dataframe(pd.DataFrame(edge_data), use_container_width=True,
                      hide_index=True)


# ══════════════════════════════════════════════════════════════
#  Page 4: Query & Retrieval
# ══════════════════════════════════════════════════════════════

def page_query_retrieval():
    st.markdown("# 🔍 Query & Retrieval")

    if st.session_state.retriever is None:
        st.warning("⚠️ Build the knowledge graph first (Page 3).")
        return

    retriever = st.session_state.retriever
    G = st.session_state.graph

    num_anchors = st.slider("Number of initial anchors", 1, 10,
                            config.NUM_INITIAL_ANCHORS, key="n_anchors")

    query = st.text_input("Enter your question:",
                          placeholder="e.g. How does backpropagation work?",
                          key="query_input")

    if query:
        with st.spinner("Retrieving anchor nodes…"):
            anchor_results = retriever.retrieve_entities(query, top_k=num_anchors)

        if not anchor_results:
            st.warning("No relevant anchor nodes found. Try a different query.")
            return

        st.markdown(
            '<div class="section-header"><h3>Initial Anchor Nodes</h3></div>',
            unsafe_allow_html=True,
        )
        anchor_df = []
        for eid, score in anchor_results:
            name = G.nodes[eid].get('name', eid) if eid in G else eid
            desc = G.nodes[eid].get('description', '')[:120] if eid in G else ''
            anchor_df.append({
                'Entity': name,
                'Similarity': f"{score:.4f}",
                'Description': desc,
            })
        st.dataframe(pd.DataFrame(anchor_df), use_container_width=True,
                      hide_index=True)

        # ── Run retrieval (RL or fixed-depth) ──────────────────
        st.markdown("---")
        model = st.session_state.rl_model
        if model is None and is_trained():
            model = load_checkpoint()
            st.session_state.rl_model = model
            st.session_state.rl_trained = True

        use_rl = st.session_state.rl_trained and model is not None

        col_rl, col_fd = st.columns(2)
        with col_rl:
            run_rl = st.button("🤖 Run RL Retrieval",
                               disabled=not use_rl, key="run_rl")
        with col_fd:
            fd_depth = st.selectbox("Fixed depth", [1, 2, 3], index=1,
                                    key="fd_depth")
            run_fd = st.button("📏 Run Fixed-Depth Retrieval", key="run_fd")

        if run_rl and model:
            with st.spinner("RL agent exploring graph…"):
                result = rl_retrieve(query, G, retriever, model,
                                     num_anchors=num_anchors)
                st.session_state.retrieval_result = result
                _display_retrieval_result(result, G)

        if run_fd:
            with st.spinner("Fixed-depth BFS…"):
                result = fixed_depth_retrieve(query, G, retriever,
                                              depth=fd_depth,
                                              num_anchors=num_anchors)
                st.session_state.retrieval_result = result
                _display_retrieval_result(result, G)

        if not use_rl:
            st.info("Train the RL agent on the **RL Agent** page to enable "
                    "adaptive retrieval.")


def _display_retrieval_result(result, G):
    """Shared display for both RL and fixed-depth retrieval results."""
    st.markdown(
        '<div class="section-header"><h3>Retrieval Result</h3></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes Expanded", len(result.expanded_nodes))
    c2.metric("Depth", result.total_depth)
    c3.metric("Est. Tokens", result.token_estimate)
    c4.metric("Avg Relevance", f"{result.final_relevance:.3f}")

    # Decision history
    if result.steps:
        st.markdown("**Agent Decision History:**")
        for step in result.steps:
            action = step.action
            css = "expand-action" if action == "EXPAND" else "stop-action"
            node_info = f" → **{step.node_name}** (score: {step.relevance_score:.3f})" if step.node_name else ""
            st.markdown(
                f'<div class="step-card {css}">'
                f'Step {step.step_idx}: <b>{action}</b>{node_info}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Expanded nodes list
    st.markdown("**Expanded Nodes:**")
    for nid in result.expanded_nodes:
        name = G.nodes[nid].get('name', nid) if nid in G else nid
        st.markdown(f"- {name}")


# ══════════════════════════════════════════════════════════════
#  Page 5: RL Agent
# ══════════════════════════════════════════════════════════════

def page_rl_agent():
    st.markdown("# 🤖 RL Agent — PPO Training")

    if st.session_state.graph is None or st.session_state.retriever is None:
        st.warning("⚠️ Build the knowledge graph and vector index first.")
        return

    # ── Training status ────────────────────────────────────────
    trained = st.session_state.rl_trained or is_trained()
    if trained:
        st.markdown('<span class="badge-trained">✓ PPO Agent Trained</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-untrained">✗ PPO Agent Not Trained</span>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Training configuration ─────────────────────────────────
    st.markdown(
        '<div class="section-header"><h3>Training Configuration</h3></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        total_steps = st.number_input(
            "Total timesteps", min_value=1000, max_value=200000,
            value=config.PPO_TOTAL_TIMESTEPS, step=5000, key="train_steps")
    with c2:
        st.markdown(f"**Learning rate:** {config.PPO_LEARNING_RATE}")
        st.markdown(f"**Network:** {config.PPO_POLICY_KWARGS['net_arch']}")
    with c3:
        st.markdown(f"**α (quality):** {config.ALPHA}")
        st.markdown(f"**β (cost):** {config.BETA}")
        st.markdown(f"**Demo questions:** {len(st.session_state.demo_questions)}")

    # ── Train button ───────────────────────────────────────────
    if st.button("🚀 Train PPO Agent", key="train_btn", type="primary"):
        progress_bar = st.progress(0, text="Initialising training…")
        status_text = st.empty()

        try:
            status_text.info("Setting up environment and PPO model…")
            progress_bar.progress(10, text="Training in progress…")

            model, callback = train_ppo(
                graph=st.session_state.graph,
                retriever=st.session_state.retriever,
                total_timesteps=total_steps,
                demo_questions=st.session_state.demo_questions,
            )

            st.session_state.rl_model = model
            st.session_state.rl_trained = True
            st.session_state.training_callback = callback

            progress_bar.progress(100, text="Training complete!")
            status_text.success(
                f"✅ Training complete — {len(callback.episode_rewards)} "
                f"episodes, checkpoint saved."
            )
        except Exception as e:
            progress_bar.empty()
            status_text.error(f"❌ Training failed: {e}")
            raise

    # ── Load existing checkpoint ───────────────────────────────
    if not st.session_state.rl_trained and is_trained():
        if st.button("📂 Load saved checkpoint", key="load_ckpt"):
            model = load_checkpoint()
            if model:
                st.session_state.rl_model = model
                st.session_state.rl_trained = True
                st.success("✅ Checkpoint loaded.")
            else:
                st.error("Could not load checkpoint.")

    # ── Training metrics ───────────────────────────────────────
    cb = st.session_state.training_callback
    if cb and cb.episode_rewards:
        st.markdown("---")
        st.markdown(
            '<div class="section-header"><h3>Training Metrics</h3></div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Episodes", len(cb.episode_rewards))
        c2.metric("Mean Reward", f"{np.mean(cb.episode_rewards):.3f}")
        c3.metric("Mean Episode Len", f"{np.mean(cb.episode_lengths):.1f}")

        # Reward curve
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#0f172a')

        # Smooth rewards with a rolling window
        rewards = np.array(cb.episode_rewards)
        window = min(20, len(rewards) // 3 + 1)
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        else:
            smoothed = rewards

        ax1.set_facecolor('#1e293b')
        ax1.plot(rewards, alpha=0.3, color='#6366f1', linewidth=0.8)
        ax1.plot(range(len(smoothed)), smoothed, color='#22d3ee', linewidth=2)
        ax1.set_xlabel('Episode', color='#94a3b8')
        ax1.set_ylabel('Reward', color='#94a3b8')
        ax1.set_title('Episode Rewards', color='#f1f5f9')
        ax1.tick_params(colors='#64748b')

        lengths = np.array(cb.episode_lengths)
        ax2.set_facecolor('#1e293b')
        ax2.plot(lengths, color='#f59e0b', alpha=0.5, linewidth=0.8)
        if len(lengths) > window:
            smooth_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(len(smooth_len)), smooth_len, color='#10b981', linewidth=2)
        ax2.set_xlabel('Episode', color='#94a3b8')
        ax2.set_ylabel('Steps', color='#94a3b8')
        ax2.set_title('Episode Length', color='#f1f5f9')
        ax2.tick_params(colors='#64748b')

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Policy info ────────────────────────────────────────────
    if st.session_state.rl_model is not None:
        st.markdown("---")
        st.markdown(
            '<div class="section-header"><h3>Policy Network</h3></div>',
            unsafe_allow_html=True,
        )
        model = st.session_state.rl_model
        st.code(str(model.policy), language="text")


# ══════════════════════════════════════════════════════════════
#  Page 6: Answer Generation
# ══════════════════════════════════════════════════════════════

def page_answer_generation():
    st.markdown("# 💬 Answer Generation")

    if st.session_state.retriever is None:
        st.warning("⚠️ Build the knowledge graph first.")
        return

    G = st.session_state.graph
    retriever = st.session_state.retriever

    # Check Ollama
    ollama_ok = check_ollama_status()
    if ollama_ok:
        st.success(f"🟢 Ollama is available (model: {config.OLLAMA_MODEL})")
    else:
        st.info("🟡 Ollama not detected — extractive fallback will be used.")

    st.markdown("---")

    query = st.text_input("Enter your question:", key="ans_query",
                          placeholder="e.g. What techniques prevent overfitting?")

    if query and st.button("Generate Answer", key="gen_ans", type="primary"):
        # Step 1: Retrieve
        model = st.session_state.rl_model
        with st.spinner("Retrieving evidence…"):
            if model:
                result = rl_retrieve(query, G, retriever, model)
            else:
                result = fixed_depth_retrieve(query, G, retriever, depth=2)
            st.session_state.retrieval_result = result

        # Step 2: Build context
        with st.spinner("Constructing context…"):
            ctx = build_context(result, G, retriever)
            st.session_state.context = ctx

        # Step 3: Generate answer
        with st.spinner("Generating answer…"):
            ans = generate_answer(query, ctx)
            st.session_state.answer = ans

        # ── Display ────────────────────────────────────────────
        st.markdown(
            '<div class="section-header"><h3>Retrieved Evidence</h3></div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Entities", ctx.num_entities)
        c2.metric("Relationships", ctx.num_relationships)
        c3.metric("Est. Tokens", ctx.estimated_tokens)

        with st.expander("📋 Compact Context", expanded=False):
            st.code(ctx.context_text, language="text")

        if ctx.entities:
            with st.expander("🔗 Entity Details"):
                for ent in ctx.entities:
                    st.markdown(
                        f"**{ent.entity_name}** "
                        f"(relevance: {ent.relevance_score:.3f}) — "
                        f"*{ent.source_document}*"
                    )

        if ctx.relationships:
            with st.expander("🔗 Relationship Details"):
                for rel in ctx.relationships:
                    st.markdown(
                        f"- {rel['source']} —[{rel['relation']}]→ "
                        f"{rel['target']}"
                    )

        st.markdown("---")
        st.markdown(
            '<div class="section-header"><h3>Generated Answer</h3></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"*Method: `{ans.method}`*")
        st.markdown(ans.answer)

        if ans.source_references:
            st.markdown("**Source References:**")
            for ref in ans.source_references:
                st.markdown(f"- 📄 {ref}")


# ══════════════════════════════════════════════════════════════
#  Page 7: Evaluation
# ══════════════════════════════════════════════════════════════

def page_evaluation():
    st.markdown("# 📊 Evaluation — Fixed-Depth vs RL")

    if st.session_state.retriever is None:
        st.warning("⚠️ Build the knowledge graph first.")
        return

    model = st.session_state.rl_model
    if model is None and is_trained():
        model = load_checkpoint()
        st.session_state.rl_model = model
        st.session_state.rl_trained = True

    demo_qs = st.session_state.demo_questions
    if not demo_qs:
        st.warning("No demo questions available for evaluation.")
        return

    st.markdown(f"Evaluating on **{len(demo_qs)}** labelled demo questions.")

    if st.button("🏁 Run Evaluation", key="run_eval", type="primary"):
        with st.spinner("Running evaluation (this may take a minute)…"):
            report = evaluate(
                demo_questions=demo_qs,
                graph=st.session_state.graph,
                retriever=st.session_state.retriever,
                model=model,
            )
            st.session_state.eval_report = report
        st.success("✅ Evaluation complete.")

    report = st.session_state.eval_report
    if report is None:
        st.info("Click **Run Evaluation** to compare methods.")
        return

    # ── Summary table ──────────────────────────────────────────
    st.markdown(
        '<div class="section-header"><h3>Summary Metrics</h3></div>',
        unsafe_allow_html=True,
    )

    rows = []
    for method, data in report.summary.items():
        rows.append({
            'Method': method.replace('_', ' ').title(),
            'Avg Expanded': f"{data['avg_expanded']:.1f}",
            'Avg Depth': f"{data['avg_depth']:.1f}",
            'Avg Latency (s)': f"{data['avg_latency']:.4f}",
            'Avg Tokens': f"{data['avg_tokens']:.0f}",
            'Avg Recall': f"{data['avg_recall']:.3f}",
            'Avg Cost': f"{data['avg_cost']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Charts ─────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header"><h3>Comparison Charts</h3></div>',
        unsafe_allow_html=True,
    )

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = list(report.summary.keys())
    metric_names = ['avg_expanded', 'avg_recall', 'avg_tokens', 'avg_latency']
    display_names = ['Nodes Expanded', 'Recall', 'Token Count', 'Latency (s)']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor('#0f172a')

    colors = ['#6366f1', '#22d3ee', '#f59e0b', '#10b981', '#ec4899']

    for ax, metric, title in zip(axes, metric_names, display_names):
        ax.set_facecolor('#1e293b')
        vals = [report.summary[m][metric] for m in methods]
        labels = [m.replace('_', ' ').title() for m in methods]
        bars = ax.bar(labels, vals, color=colors[:len(methods)], alpha=0.85)
        ax.set_title(title, color='#f1f5f9', fontsize=10)
        ax.tick_params(colors='#64748b', labelsize=7)
        ax.tick_params(axis='x', rotation=25)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom',
                    color='#e2e8f0', fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Per-question details ───────────────────────────────────
    st.markdown(
        '<div class="section-header"><h3>Per-Question Results</h3></div>',
        unsafe_allow_html=True,
    )

    all_details = []
    for m in report.rl_metrics:
        all_details.append({
            'Question': m.question[:60] + '…' if len(m.question) > 60 else m.question,
            'Method': 'RL',
            'Expanded': m.num_expanded,
            'Depth': m.depth,
            'Tokens': m.token_count,
            'Recall': f"{m.recall:.3f}",
            'Latency': f"{m.latency_s:.4f}s",
        })
    for depth, metrics in report.fixed_metrics.items():
        for m in metrics:
            all_details.append({
                'Question': m.question[:60] + '…' if len(m.question) > 60 else m.question,
                'Method': f'Fixed-{depth}',
                'Expanded': m.num_expanded,
                'Depth': m.depth,
                'Tokens': m.token_count,
                'Recall': f"{m.recall:.3f}",
                'Latency': f"{m.latency_s:.4f}s",
            })

    if all_details:
        st.dataframe(pd.DataFrame(all_details), use_container_width=True,
                      hide_index=True)


# ══════════════════════════════════════════════════════════════
#  Router
# ══════════════════════════════════════════════════════════════

_PAGES = {
    "🏠 Overview": page_overview,
    "📄 Document Processing": page_documents,
    "🕸️ Knowledge Graph": page_knowledge_graph,
    "🔍 Query & Retrieval": page_query_retrieval,
    "🤖 RL Agent": page_rl_agent,
    "💬 Answer Generation": page_answer_generation,
    "📊 Evaluation": page_evaluation,
}

_PAGES[page]()
