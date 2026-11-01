"""
modules/document_processor.py — Document Upload and Preprocessing (Module 1).

Handles text extraction from PDF, TXT, and DOCX files, along with cleaning,
normalisation, and chunking.  Each chunk carries metadata (source document
name and a sequential chunk ID).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from utils.helpers import clean_text, chunk_text
import config


# ──────────────────────────────────────────────────────────────
# Data-classes
# ──────────────────────────────────────────────────────────────
@dataclass
class TextChunk:
    """A single chunk of text with provenance metadata."""
    chunk_id: str          # e.g. "neural_networks_chunk_0"
    text: str
    source_document: str   # filename of the originating document
    chunk_index: int       # positional index within the document
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class ProcessedDocument:
    """Result of processing one document."""
    filename: str
    raw_text: str
    cleaned_text: str
    chunks: List[TextChunk] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Extractors
# ──────────────────────────────────────────────────────────────
def extract_text_from_txt(filepath: str) -> str:
    """Read plain-text files."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(filepath)
        pages = [page.get_text() for page in doc]
        doc.close()
        return '\n'.join(pages)
    except ImportError:
        # Fallback to pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            return '\n'.join(page.extract_text() or '' for page in reader.pages)
        except ImportError:
            raise ImportError(
                "Install PyMuPDF (`pip install PyMuPDF`) or pypdf (`pip install pypdf`) "
                "for PDF support."
            )


def extract_text_from_docx(filepath: str) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
        doc = Document(filepath)
        return '\n'.join(para.text for para in doc.paragraphs)
    except ImportError:
        raise ImportError(
            "Install python-docx (`pip install python-docx`) for DOCX support."
        )


# Dispatcher
_EXTRACTORS = {
    '.txt': extract_text_from_txt,
    '.pdf': extract_text_from_pdf,
    '.docx': extract_text_from_docx,
}


def extract_text(filepath: str) -> str:
    """Route to the correct extractor based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(f"Unsupported file extension: {ext}. Supported: {list(_EXTRACTORS)}")
    return extractor(filepath)


# ──────────────────────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────────────────────
def process_document(filepath: str,
                     chunk_size: int = config.CHUNK_SIZE,
                     chunk_overlap: int = config.CHUNK_OVERLAP,
                     min_chunk_length: int = config.MIN_CHUNK_LENGTH
                     ) -> ProcessedDocument:
    """
    End-to-end processing of a single document file.

    1. Extract raw text.
    2. Clean and normalise.
    3. Chunk into overlapping segments.
    4. Tag each chunk with source metadata.
    """
    filename = os.path.basename(filepath)
    raw_text = extract_text(filepath)

    if not raw_text.strip():
        raise ValueError(f"No text could be extracted from '{filename}'.")

    cleaned = clean_text(raw_text)
    raw_chunks = chunk_text(cleaned, chunk_size, chunk_overlap, min_chunk_length)

    if not raw_chunks:
        raise ValueError(
            f"Document '{filename}' produced no usable chunks "
            f"(min {min_chunk_length} words required)."
        )

    base_name = os.path.splitext(filename)[0]
    chunks = [
        TextChunk(
            chunk_id=f"{base_name}_chunk_{i}",
            text=txt,
            source_document=filename,
            chunk_index=i,
        )
        for i, txt in enumerate(raw_chunks)
    ]

    return ProcessedDocument(
        filename=filename,
        raw_text=raw_text,
        cleaned_text=cleaned,
        chunks=chunks,
    )


def process_uploaded_file(uploaded_file, tmp_dir: str = "temp_uploads") -> ProcessedDocument:
    """
    Process a Streamlit ``UploadedFile`` object.

    Saves to a temp directory, processes, then cleans up.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)
    try:
        with open(tmp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return process_document(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_sample_documents() -> List[ProcessedDocument]:
    """Load and process all files in the built-in sample-documents folder."""
    docs: List[ProcessedDocument] = []
    if not os.path.isdir(config.SAMPLE_DOCS_DIR):
        return docs
    for fname in sorted(os.listdir(config.SAMPLE_DOCS_DIR)):
        fpath = os.path.join(config.SAMPLE_DOCS_DIR, fname)
        if os.path.isfile(fpath):
            try:
                docs.append(process_document(fpath))
            except (ValueError, ImportError) as exc:
                print(f"[WARN] Skipping {fname}: {exc}")
    return docs
