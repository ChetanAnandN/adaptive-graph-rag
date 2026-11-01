"""
modules/entity_extractor.py — Entity and Relationship Extraction (Module 2a).

Uses rule-based and pattern-based NLP to extract entities and relationships
from text chunks.  No external NER model is required — the extractor relies
on capitalisation patterns, known technical terms, and syntactic templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional

from utils.helpers import normalize_entity_name
import config


# ──────────────────────────────────────────────────────────────
# Data-classes
# ──────────────────────────────────────────────────────────────
@dataclass
class Entity:
    """A named concept extracted from text."""
    name: str                    # canonical display name
    description: str = ""        # first sentence mentioning the entity
    source_document: str = ""
    chunk_ids: List[str] = field(default_factory=list)
    entity_type: str = "concept" # concept | method | metric | model

    @property
    def normalized(self) -> str:
        return normalize_entity_name(self.name)


@dataclass
class Relationship:
    """A directed relationship between two entities."""
    source: str           # entity name
    target: str           # entity name
    relation: str         # e.g. "is_a", "uses", "contains"
    evidence: str = ""    # sentence where the relationship was found
    chunk_id: str = ""
    source_document: str = ""


# ──────────────────────────────────────────────────────────────
# Common English stop words (kept short — used only for filtering)
# ──────────────────────────────────────────────────────────────
_STOP_WORDS: Set[str] = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
    'should', 'may', 'might', 'must', 'can', 'could', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'out',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'but', 'and', 'or', 'if', 'while', 'because', 'until', 'about',
    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
    'their', 'what', 'which', 'who', 'whom', 'we', 'us', 'our', 'he',
    'she', 'him', 'her', 'his', 'my', 'your', 'also', 'however',
    'although', 'since', 'whether', 'i', 'me', 'any', 'many', 'much',
    'well', 'even', 'still', 'yet', 'one', 'two', 'three', 'new',
    'first', 'last', 'long', 'great', 'little', 'right', 'old', 'big',
    'high', 'different', 'small', 'large', 'next', 'early', 'young',
    'important', 'public', 'bad', 'good', 'example', 'common',
    'examples', 'including', 'given', 'rather', 'particularly',
    'instead', 'without', 'using', 'according', 'typically', 'based',
    'called', 'known', 'used', 'certain', 'either', 'among',
}

# Additional single-word entity blacklist
_ENTITY_BLACKLIST: Set[str] = {
    'however', 'therefore', 'furthermore', 'moreover', 'additionally',
    'consequently', 'alternatively', 'meanwhile', 'nevertheless',
    'otherwise', 'specifically', 'generally', 'particularly',
    'essentially', 'approximately', 'during', 'despite', 'whereas',
    'solutions', 'examples', 'result', 'results', 'output', 'input',
    'inputs', 'outputs', 'figure', 'table', 'section', 'chapter',
}


# ──────────────────────────────────────────────────────────────
# Relationship patterns  (regex-based templates)
# ──────────────────────────────────────────────────────────────
# Each entry: (compiled regex, relation label, source_group, target_group)
_RELATIONSHIP_PATTERNS: List[Tuple[re.Pattern, str, int, int]] = [
    (re.compile(
        r'([A-Z][a-zA-Z\s\-]+?)\s+is\s+(?:a|an)\s+([A-Z][a-zA-Z\s\-]+?)(?:\s+that|\s+which|\s+where|\.|\,)',
        re.DOTALL), 'is_a', 1, 2),
    (re.compile(
        r'([A-Z][a-zA-Z\s\-]+?)\s+(?:is\s+a\s+type\s+of|is\s+a\s+subfield\s+of|is\s+a\s+branch\s+of|is\s+a\s+variant\s+of)\s+([A-Z][a-zA-Z\s\-]+?)(?:\.|\,)',
        re.DOTALL), 'is_type_of', 1, 2),
    (re.compile(
        r'([A-Z][a-zA-Z\s\-]+?)\s+(?:uses|utilizes|employs|leverages|applies)\s+(?:the\s+)?([A-Z][a-zA-Z\s\-]+?)(?:\s+to|\s+for|\s+in|\.|\,)',
        re.DOTALL), 'uses', 1, 2),
    (re.compile(
        r'([A-Z][a-zA-Z\s\-]+?)\s+(?:consists?\s+of|is\s+composed\s+of|contains?|includes?)\s+([A-Z][a-zA-Z\s\-]+?)(?:\.|\,|\s+and)',
        re.DOTALL), 'contains', 1, 2),
    (re.compile(
        r'([A-Z][a-zA-Z\s\-]+?)\s+(?:produces?|generates?|outputs?|creates?)\s+(?:the\s+)?([A-Z][a-zA-Z\s\-]+?)(?:\.|\,)',
        re.DOTALL), 'produces', 1, 2),
    (re.compile(
        r'([A-Z][a-zA-Z\s\-]+?)\s+(?:prevents?|reduces?|mitigates?|eliminates?)\s+(?:the\s+)?([A-Z][a-zA-Z\s\-]+?)(?:\.|\,)',
        re.DOTALL), 'mitigates', 1, 2),
    (re.compile(
        r'([A-Z][a-zA-Z\s\-]+?)\s+(?:extends?|improves?|enhances?)\s+([A-Z][a-zA-Z\s\-]+?)(?:\s+by|\.|\,)',
        re.DOTALL), 'extends', 1, 2),
    (re.compile(
        r'([A-Z][a-zA-Z\s\-]+?)\s+(?:is\s+based\s+on|relies\s+on|builds\s+on)\s+(?:the\s+)?([A-Z][a-zA-Z\s\-]+?)(?:\.|\,)',
        re.DOTALL), 'based_on', 1, 2),
    (re.compile(
        r'([A-Z][a-zA-Z\s\-]+?)\s+(?:is\s+used\s+(?:for|in))\s+([A-Z]?[a-zA-Z\s\-]+?)(?:\.|\,)',
        re.DOTALL), 'used_for', 1, 2),
    (re.compile(
        r'([A-Z][a-zA-Z\s\-]+?)\s+(?:combines?)\s+(?:the\s+)?([A-Z][a-zA-Z\s\-]+?)(?:\s+and|\s+with|\.|\,)',
        re.DOTALL), 'combines', 1, 2),
]


# ──────────────────────────────────────────────────────────────
# Entity extraction
# ──────────────────────────────────────────────────────────────

def _is_valid_entity(name: str) -> bool:
    """Heuristic filter for candidate entity names."""
    name = name.strip()
    if len(name) < config.MIN_ENTITY_NAME_LENGTH:
        return False
    if len(name) > config.MAX_ENTITY_NAME_LENGTH:
        return False
    # Must start with a letter
    if not name[0].isalpha():
        return False
    # Reject pure stop-words or blacklisted terms
    if normalize_entity_name(name) in _STOP_WORDS:
        return False
    if normalize_entity_name(name) in _ENTITY_BLACKLIST:
        return False
    # Reject if all words are stop words
    words = name.lower().split()
    if all(w in _STOP_WORDS for w in words):
        return False
    return True


def extract_entities_from_text(text: str,
                                source_document: str = "",
                                chunk_id: str = "") -> List[Entity]:
    """
    Extract candidate entities from a text chunk using capitalisation
    patterns and acronym detection.

    Strategy
    --------
    1.  Multi-word capitalised phrases  (e.g. "Neural Network").
    2.  Parenthetical acronyms / abbreviations  (e.g. "(NER)").
    3.  Single capitalised words that are not common English words.
    """
    entities_dict: Dict[str, Entity] = {}
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        # ── Multi-word capitalised phrases ──────────────────────
        # Match sequences of 2+ capitalised words (may include hyphens)
        for m in re.finditer(r'(?:[A-Z][a-z]+(?:[\-\s](?:[A-Z][a-z]+|[a-z]+)){0,5})', sentence):
            name = m.group().strip()
            # Require at least one capitalised word that isn't a stop word
            words = name.split()
            significant = [w for w in words if w.lower() not in _STOP_WORDS and w[0].isupper()]
            if not significant:
                continue
            # Trim trailing lowercase words that are likely connectors
            while words and words[-1].lower() in _STOP_WORDS:
                words.pop()
            name = ' '.join(words)
            if _is_valid_entity(name):
                key = normalize_entity_name(name)
                if key not in entities_dict:
                    entities_dict[key] = Entity(
                        name=name,
                        description=sentence.strip(),
                        source_document=source_document,
                        chunk_ids=[chunk_id],
                    )
                else:
                    if chunk_id not in entities_dict[key].chunk_ids:
                        entities_dict[key].chunk_ids.append(chunk_id)

        # ── Acronyms in parentheses ────────────────────────────
        for m in re.finditer(r'\(([A-Z]{2,})\)', sentence):
            name = m.group(1)
            if _is_valid_entity(name):
                key = normalize_entity_name(name)
                if key not in entities_dict:
                    entities_dict[key] = Entity(
                        name=name,
                        description=sentence.strip(),
                        source_document=source_document,
                        chunk_ids=[chunk_id],
                        entity_type="acronym",
                    )
                else:
                    if chunk_id not in entities_dict[key].chunk_ids:
                        entities_dict[key].chunk_ids.append(chunk_id)

    return list(entities_dict.values())


# ──────────────────────────────────────────────────────────────
# Relationship extraction
# ──────────────────────────────────────────────────────────────

def extract_relationships_from_text(text: str,
                                     known_entities: Set[str],
                                     source_document: str = "",
                                     chunk_id: str = "") -> List[Relationship]:
    """
    Extract relationships using syntactic patterns and co-occurrence.

    Parameters
    ----------
    known_entities : Set of *normalised* entity names already extracted.
    """
    relationships: List[Relationship] = []
    seen: Set[Tuple[str, str, str]] = set()
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        # ── Pattern-based extraction ───────────────────────────
        for pattern, relation, src_grp, tgt_grp in _RELATIONSHIP_PATTERNS:
            for m in pattern.finditer(sentence):
                src = m.group(src_grp).strip()
                tgt = m.group(tgt_grp).strip()
                if not _is_valid_entity(src) or not _is_valid_entity(tgt):
                    continue
                # Skip self-loops
                if normalize_entity_name(src) == normalize_entity_name(tgt):
                    continue
                key = (normalize_entity_name(src), relation, normalize_entity_name(tgt))
                if key not in seen:
                    seen.add(key)
                    relationships.append(Relationship(
                        source=src, target=tgt, relation=relation,
                        evidence=sentence.strip(), chunk_id=chunk_id,
                        source_document=source_document,
                    ))

        # ── Co-occurrence (entities in the same sentence) ──────
        present = []
        for ent in known_entities:
            if ent in normalize_entity_name(sentence):
                present.append(ent)
        # Create co-occurrence edges for pairs that don't already
        # have a pattern-based edge
        for i, e1 in enumerate(present):
            for e2 in present[i + 1:]:
                key_fwd = (e1, 'co_occurs', e2)
                key_rev = (e2, 'co_occurs', e1)
                if key_fwd not in seen and key_rev not in seen:
                    seen.add(key_fwd)
                    relationships.append(Relationship(
                        source=e1, target=e2, relation='co_occurs',
                        evidence=sentence.strip(), chunk_id=chunk_id,
                        source_document=source_document,
                    ))

    return relationships


# ──────────────────────────────────────────────────────────────
# High-level extraction from chunks
# ──────────────────────────────────────────────────────────────

def extract_from_chunks(chunks) -> Tuple[List[Entity], List[Relationship]]:
    """
    Run entity + relationship extraction over a list of ``TextChunk`` objects.

    Returns
    -------
    (entities, relationships) — deduplicated across all chunks.
    """
    all_entities: Dict[str, Entity] = {}
    all_relationships: List[Relationship] = []

    # Pass 1: extract entities
    for chunk in chunks:
        ents = extract_entities_from_text(
            chunk.text,
            source_document=chunk.source_document,
            chunk_id=chunk.chunk_id,
        )
        for e in ents:
            key = e.normalized
            if key in all_entities:
                # Merge chunk IDs
                for cid in e.chunk_ids:
                    if cid not in all_entities[key].chunk_ids:
                        all_entities[key].chunk_ids.append(cid)
            else:
                all_entities[key] = e

    known_names = set(all_entities.keys())

    # Pass 2: extract relationships
    seen_rels: Set[Tuple[str, str, str]] = set()
    for chunk in chunks:
        rels = extract_relationships_from_text(
            chunk.text,
            known_entities=known_names,
            source_document=chunk.source_document,
            chunk_id=chunk.chunk_id,
        )
        for r in rels:
            key = (normalize_entity_name(r.source), r.relation,
                   normalize_entity_name(r.target))
            if key not in seen_rels:
                seen_rels.add(key)
                all_relationships.append(r)

    return list(all_entities.values()), all_relationships
