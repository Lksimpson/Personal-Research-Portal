"""
Retrieve top-k chunks from FAISS index (or keyword fallback). Local embeddings only: FastEmbed or sentence-transformers.
On query-embed failure, falls back to keyword retrieval.
"""
from pathlib import Path
import json
import re
import numpy as np
import faiss

MODEL_NAME = "all-MiniLM-L6-v2"


def _keyword_retrieve(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Retrieve top-k chunks by keyword overlap (no embeddings)."""
    words = set(re.findall(r"\w+", query.lower()))
    if not words:
        return chunks[:top_k]
    scored = []
    for c in chunks:
        text = (c.get("text") or "").lower()
        count = sum(1 for w in words if w in text)
        scored.append((count, c))
    scored.sort(key=lambda x: -x[0])
    out = []
    for count, c in scored[:top_k]:
        out.append(dict(c, score=float(count) / max(len(words), 1)))
    return out


def _embed_query_fastembed(query: str) -> np.ndarray:
    """Embed single query via FastEmbed. Returns (1, dim) float32, L2-normalized."""
    from src.rag.embed_index import FASTEMBED_MODEL
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise ImportError("fastembed not available. pip install fastembed.") from e
    text = f"query: {query}" if not query.strip().startswith("query:") else query
    embedding_model = TextEmbedding(FASTEMBED_MODEL)
    q = np.array(list(embedding_model.embed([text]))[0], dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    return q


def load_index(index_path: Path, chunks_path: Path):
    """Load FAISS index and chunks. Returns (index, chunks, use_fastembed_emb)."""
    index = faiss.read_index(str(index_path))
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    meta_path = index_path.parent / "embed_meta.json"
    use_fastembed = False
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        use_fastembed = meta.get("embedding", "") == "fastembed"
    return index, chunks, use_fastembed


def load_chunks_only(chunks_path: Path) -> list[dict]:
    """Load chunk list only (when index is missing)."""
    with open(chunks_path, encoding="utf-8") as f:
        return json.load(f)


def retrieve(
    query: str,
    index,
    chunks: list[dict],
    model=None,
    top_k: int = 5,
    use_openai_embeddings: bool = False,
    use_gemini_embeddings: bool = False,
    use_fastembed_embeddings: bool = False,
    skip_query_embed: bool = False,
) -> list[dict]:
    """
    Return top_k chunks with scores. Uses semantic retrieval when index exists; falls back to keyword on exception.
    use_openai_embeddings / use_gemini_embeddings are ignored (local only).
    """
    del use_openai_embeddings, use_gemini_embeddings  # local only
    if skip_query_embed or index is None:
        return _keyword_retrieve(query, chunks, top_k)
    try:
        if use_fastembed_embeddings:
            q = _embed_query_fastembed(query)
        else:
            if model is None:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(MODEL_NAME)
            q = model.encode([query])
            faiss.normalize_L2(q)
            q = q.astype(np.float32)
        scores, indices = index.search(q, min(top_k, len(chunks)))
        out = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            c = dict(chunks[idx])
            c["score"] = float(scores[0][i])
            out.append(c)
        return out
    except Exception:
        return _keyword_retrieve(query, chunks, top_k)
