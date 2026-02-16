"""
Retrieve top-k chunks from FAISS index and return chunks with (source_id, chunk_id) for citations.
Uses Gemini or OpenAI for query embedding when index was built with the same backend.
Set SKIP_QUERY_EMBED=1 to use keyword retrieval only (1 API call per query = generate only).
"""
from pathlib import Path
import json
import os
import re
import numpy as np
import faiss

MODEL_NAME = "all-MiniLM-L6-v2"


def _keyword_retrieve(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Retrieve top-k chunks by simple keyword overlap (no API). For use when SKIP_QUERY_EMBED=1."""
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


def _embed_query_gemini(query: str) -> np.ndarray:
    """Embed single query via Gemini. Returns (1, dim) float32, L2-normalized."""
    from google import genai
    from src.rag.embed_index import GEMINI_EMBED_MODEL
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=key)
    resp = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=query,
        task_type="retrieval_query",
    )
    if hasattr(resp, "embedding") and resp.embedding is not None:
        emb = resp.embedding.values if hasattr(resp.embedding, "values") else resp.embedding
    elif isinstance(resp, dict) and "embedding" in resp:
        emb = resp["embedding"]
    else:
        raise ValueError("Unexpected Gemini embedding response format")
    q = np.array([emb], dtype=np.float32)
    faiss.normalize_L2(q)
    return q


def _embed_query_fastembed(query: str) -> np.ndarray:
    """Embed single query via FastEmbed (ONNX). Returns (1, dim) float32, L2-normalized. Use 'query:' prefix for BGE."""
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise ImportError(
            "fastembed not available. Install with: pip install fastembed. "
            "If already installed, use the same Python that runs this script (e.g. python -m pip install fastembed)."
        ) from e
    from src.rag.embed_index import FASTEMBED_MODEL
    text = f"query: {query}" if not query.strip().startswith("query:") else query
    embedding_model = TextEmbedding(FASTEMBED_MODEL)
    q = np.array(list(embedding_model.embed([text]))[0], dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    return q


def _embed_query_openai(query: str) -> np.ndarray:
    """Embed single query via OpenAI with retry. Returns (1, dim) float32, L2-normalized."""
    from openai import OpenAI
    import httpx
    import time
    client = OpenAI()
    from openai import APIConnectionError
    for attempt in range(4):
        try:
            resp = client.embeddings.create(
                model="text-embedding-3-small", input=[query]
            )
            q = np.array([resp.data[0].embedding], dtype=np.float32)
            faiss.normalize_L2(q)
            return q
        except Exception as e:
            retryable = isinstance(e, (
                httpx.ReadError, httpx.ConnectError, OSError, APIConnectionError
            ))
            if not retryable or attempt == 3:
                raise
            time.sleep(2 * (2 ** attempt))


def load_index(index_path: Path, chunks_path: Path):
    """Load FAISS index and chunk list. Returns (index, chunks, use_openai_emb, use_gemini_emb, use_fastembed_emb)."""
    index = faiss.read_index(str(index_path))
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    meta_path = index_path.parent / "embed_meta.json"
    use_openai = use_gemini = use_fastembed = False
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        emb = meta.get("embedding", "")
        use_openai = emb == "openai"
        use_gemini = emb == "gemini"
        use_fastembed = emb == "fastembed"
    return index, chunks, use_openai, use_gemini, use_fastembed


def load_chunks_only(chunks_path: Path) -> list[dict]:
    """Load chunk list only (for SKIP_QUERY_EMBED=1 when index is missing)."""
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
    Return top_k chunks with scores. Each item: chunk dict + "score" key.
    If skip_query_embed=True, use keyword retrieval only (no embed API call).
    """
    if skip_query_embed:
        return _keyword_retrieve(query, chunks, top_k)
    if use_fastembed_embeddings:
        q = _embed_query_fastembed(query)
    elif use_gemini_embeddings:
        q = _embed_query_gemini(query)
    elif use_openai_embeddings:
        q = _embed_query_openai(query)
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
