"""
Chunking strategy (documented for Phase 2).
- Chunk size: 2048 chars with 256 char overlap.
- Section-aware: we split on paragraph boundaries when possible to avoid mid-sentence cuts.
- Each chunk gets chunk_id = f"{source_id}_chunk_{i}" for citation.
"""
import json
import re
from pathlib import Path


# Approximate chars per token for English (~4 chars/token). We target ~512 tokens.
CHUNK_CHARS = 2048
OVERLAP_CHARS = 256


def _split_into_paragraphs(text: str) -> list[str]:
    """Split on double newlines first to respect paragraph boundaries."""
    blocks = re.split(r"\n\s*\n", text.strip())
    return [b.strip() for b in blocks if b.strip()]


def chunk_text(text: str, source_id: str) -> list[dict]:
    """
    Chunk text with overlap. Each chunk: { "chunk_id", "source_id", "text" }.
    Uses paragraph boundaries when possible; otherwise fixed-size with overlap.
    """
    if not text or not text.strip():
        return []

    paragraphs = _split_into_paragraphs(text)
    chunks = []
    buffer = ""
    chunk_index = 0

    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= CHUNK_CHARS and buffer:
            buffer += "\n\n" + para
        else:
            if buffer:
                chunk_id = f"{source_id}_chunk_{chunk_index}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "text": buffer.strip(),
                })
                chunk_index += 1
                # Overlap: keep last OVERLAP_CHARS of buffer
                buffer = buffer[-OVERLAP_CHARS:] if len(buffer) > OVERLAP_CHARS else buffer
            buffer = para

    if buffer.strip():
        chunk_id = f"{source_id}_chunk_{chunk_index}"
        chunks.append({
            "chunk_id": chunk_id,
            "source_id": source_id,
            "text": buffer.strip(),
        })

    return chunks


def load_processed_doc(path: Path) -> dict:
    """Load one JSON from data/processed/."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def chunk_corpus(base_dir: Path, manifest_raw_paths: list[tuple[str, str]]) -> list[dict]:
    """
    Build all chunks from processed JSONs.
    manifest_raw_paths: list of (source_id, processed_path) from manifest.
    processed_path is relative to base_dir (e.g. data/processed/Aksoy2023.json).
    Returns list of chunk dicts with chunk_id, source_id, text.
    """
    all_chunks = []
    for source_id, processed_path in manifest_raw_paths:
        path = base_dir / processed_path
        if not path.exists():
            continue
        doc = load_processed_doc(path)
        text = doc.get("text", "")
        if not text:
            continue
        chunks = chunk_text(text, source_id)
        all_chunks.extend(chunks)
    return all_chunks
