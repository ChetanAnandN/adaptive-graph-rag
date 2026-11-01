"""
modules/answer_generator.py — Answer Generation (Module 7).

Attempts to use Ollama as a local LLM.  If Ollama is not reachable,
falls back to an extractive answer assembled from the retrieved evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

import config
from modules.context_builder import BuiltContext


@dataclass
class GeneratedAnswer:
    """Container for the final answer and metadata."""
    answer: str
    method: str                  # "ollama" or "extractive_fallback"
    source_references: List[str] = field(default_factory=list)
    model_name: str = ""
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# Ollama LLM
# ──────────────────────────────────────────────────────────────

def _try_ollama(query: str, context: BuiltContext) -> Optional[GeneratedAnswer]:
    """
    Call Ollama's REST API for answer generation.

    Returns None if Ollama is unavailable or the request fails.
    """
    try:
        import requests
    except ImportError:
        return None

    prompt = (
        "You are a knowledgeable assistant. Answer the question using ONLY "
        "the evidence provided below. If the evidence is insufficient, "
        "state what information is missing. Cite source documents when "
        "possible.\n\n"
        f"### Evidence\n{context.context_text}\n\n"
        f"### Question\n{query}\n\n"
        "### Answer"
    )

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            answer_text = data.get("response", "").strip()
            if answer_text:
                return GeneratedAnswer(
                    answer=answer_text,
                    method="ollama",
                    source_references=context.source_documents,
                    model_name=config.OLLAMA_MODEL,
                )
    except Exception:
        pass

    return None


# ──────────────────────────────────────────────────────────────
# Extractive Fallback
# ──────────────────────────────────────────────────────────────

def _extractive_fallback(query: str, context: BuiltContext) -> GeneratedAnswer:
    """
    Build an answer by selecting and combining the most relevant
    evidence snippets.  Clearly labelled as a fallback.
    """
    if not context.entities:
        return GeneratedAnswer(
            answer="No relevant evidence was found to answer this question.",
            method="extractive_fallback",
            source_references=[],
        )

    parts: List[str] = []
    parts.append(f"**[Extractive Answer — Ollama unavailable]**\n")
    parts.append(f"Based on the retrieved evidence for: *\"{query}\"*\n")

    # Top entities
    parts.append("**Key Findings:**\n")
    for i, ent in enumerate(context.entities[:6]):
        parts.append(f"{i + 1}. **{ent.entity_name}** — {ent.description}")
        parts.append(f"   *(Source: {ent.source_document}, "
                     f"Chunks: {', '.join(ent.chunk_ids[:3])})*\n")

    # Relationships
    if context.relationships:
        parts.append("**Relevant Relationships:**\n")
        for rel in context.relationships[:8]:
            parts.append(
                f"- {rel['source']} —[{rel['relation']}]→ {rel['target']}"
            )

    # Disclaimer
    if len(context.entities) < 2:
        parts.append(
            "\n⚠️ Limited evidence was retrieved. The answer may be "
            "incomplete. Consider refining your query."
        )

    return GeneratedAnswer(
        answer='\n'.join(parts),
        method="extractive_fallback",
        source_references=context.source_documents,
    )


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def generate_answer(query: str, context: BuiltContext) -> GeneratedAnswer:
    """
    Generate an answer, trying Ollama first, then extractive fallback.
    """
    # Try Ollama
    ollama_ans = _try_ollama(query, context)
    if ollama_ans is not None:
        return ollama_ans

    # Fallback
    return _extractive_fallback(query, context)


def check_ollama_status() -> bool:
    """Return True if Ollama API is reachable."""
    try:
        import requests
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags",
                            timeout=3)
        return resp.status_code == 200
    except Exception:
        return False
