"""
Embed chunks with sentence-transformers, OpenAI, or Gemini; then index with FAISS.
Local options: USE_LOCAL_EMBEDDINGS=1 (sentence-transformers) or USE_LOCAL_EMBEDDINGS=fastembed (FastEmbed, ONNX, avoids segfault on Mac).
Priority: local (1 or fastembed) -> USE_OPENAI -> Gemini -> sentence-transformers.
"""
from pathlib import Path
import json
import os
import time
import numpy as np
import faiss

MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"  # default; ONNX-based, no PyTorch
OPENAI_EMBED_MODEL = "text-embedding-3-small"
# Gemini embedding: text-embedding-004 (legacy) or gemini-embedding-001 (current). Override with GEMINI_EMBED_MODEL env.
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL") or "text-embedding-004"
META_FILENAME = "embed_meta.json"
OPENAI_MAX_INPUTS_PER_REQUEST = 2048
OPENAI_MAX_RETRIES = 4
OPENAI_RETRY_BASE_SEC = 2
GEMINI_BATCH_SIZE = 100  # Gemini batch for embedding


def _extract_embed_values(resp) -> list[list[float]]:
    """Extract embedding vectors from google-genai response."""
    if hasattr(resp, "embeddings") and resp.embeddings:
        out = []
        for emb in resp.embeddings:
            if hasattr(emb, "values"):
                out.append(emb.values)
            elif isinstance(emb, dict) and "values" in emb:
                out.append(emb["values"])
            else:
                out.append(emb)
        return out
    if hasattr(resp, "embedding") and resp.embedding is not None:
        emb = resp.embedding
        if hasattr(emb, "values"):
            return [emb.values]
        if isinstance(emb, dict) and "values" in emb:
            return [emb["values"]]
        return [emb]
    if isinstance(resp, dict):
        if "embeddings" in resp:
            return [e.get("values", e) for e in resp["embeddings"]]
        if "embedding" in resp:
            return [resp["embedding"]]
    raise ValueError("Unexpected Gemini embedding response format")


def _embed_gemini(texts: list[str]) -> np.ndarray:
    """Embed texts via Gemini API in batch (1 call per up to 100 texts). Returns (n, dim) float32, L2-normalized."""
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=key)
    # Pass list of texts: client batches internally for embed_content
    if len(texts) == 1:
        resp = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=texts[0],
        )
    else:
        resp = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=texts,
        )
    out = _extract_embed_values(resp)
    arr = np.array(out, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


def _embed_fastembed(texts: list[str], model_name: str = FASTEMBED_MODEL) -> np.ndarray:
    """Embed via FastEmbed (ONNX-based, no PyTorch). Avoids segfault on Apple Silicon. Returns (n, dim) float32, L2-normalized."""
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise ImportError(
            "fastembed not available. Install with: pip install fastembed. "
            "If already installed, use the same Python that runs this script (e.g. python -m pip install fastembed)."
        ) from e
    # Default bge model expects "passage:" prefix for documents
    prefixed = [f"passage: {t}" if not t.strip().startswith("passage:") and not t.strip().startswith("query:") else t for t in texts]
    embedding_model = TextEmbedding(model_name)
    out = np.array(list(embedding_model.embed(prefixed)), dtype=np.float32)
    faiss.normalize_L2(out)
    return out


def _embed_openai(texts: list[str]) -> np.ndarray:
    """Embed texts via OpenAI API. One request for all texts when <= 2048 (minimizes rate-limit hits)."""
    from openai import OpenAI
    import httpx
    client = OpenAI()
    out = []
    batch_size = min(len(texts), OPENAI_MAX_INPUTS_PER_REQUEST)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(OPENAI_MAX_RETRIES):
            try:
                resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
                vecs = [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]
                out.extend(vecs)
                break
            except Exception as e:
                # Retry on connection/SSL errors (OpenAI wraps httpx errors as APIConnectionError)
                from openai import APIConnectionError
                retryable = isinstance(e, (
                    httpx.ReadError, httpx.ConnectError, OSError, APIConnectionError
                ))
                if not retryable or attempt == OPENAI_MAX_RETRIES - 1:
                    raise
                time.sleep(OPENAI_RETRY_BASE_SEC * (2 ** attempt))
        # Throttle between batches if corpus is huge (stay under ~2 requests/min)
        if i + batch_size < len(texts):
            time.sleep(35)
    arr = np.array(out, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


def _use_fastembed() -> bool:
    return os.environ.get("USE_LOCAL_EMBEDDINGS", "").lower() == "fastembed"


def embed_chunks(chunks: list[dict], model=None) -> np.ndarray:
    """Return (n_chunks, dim) array. USE_LOCAL_EMBEDDINGS=fastembed -> FastEmbed; 1 -> sentence-transformers; else API/local."""
    texts = [c["text"] for c in chunks]
    if _use_fastembed():
        return _embed_fastembed(texts)
    if os.environ.get("USE_LOCAL_EMBEDDINGS") == "1":
        from sentence_transformers import SentenceTransformer
        if model is None:
            model = SentenceTransformer(MODEL_NAME)
        return model.encode(texts, show_progress_bar=True)
    if (os.environ.get("USE_OPENAI") == "1" or os.environ.get("USE_OPENAI_EMBEDDINGS") == "1") and os.environ.get("OPENAI_API_KEY"):
        return _embed_openai(texts)
    if os.environ.get("GEMINI_API_KEY"):
        return _embed_gemini(texts)
    from sentence_transformers import SentenceTransformer
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    return model.encode(texts, show_progress_bar=True)


def build_index(chunks: list[dict], index_path: Path, chunks_path: Path, model=None):
    """
    Embed all chunks, build FAISS index, save index and chunk list for retrieval.
    USE_LOCAL_EMBEDDINGS=1 -> local only (API used only for answer); else Gemini / OpenAI / sentence-transformers.
    """
    use_fastembed = _use_fastembed()
    use_local_st = os.environ.get("USE_LOCAL_EMBEDDINGS") == "1"
    use_local = use_fastembed or use_local_st
    use_openai = not use_local and (os.environ.get("USE_OPENAI") == "1" or os.environ.get("USE_OPENAI_EMBEDDINGS") == "1") and os.environ.get("OPENAI_API_KEY")
    use_gemini = not use_local and not use_openai and bool(os.environ.get("GEMINI_API_KEY"))
    if use_fastembed:
        embeddings = _embed_fastembed([c["text"] for c in chunks])
    elif use_local_st:
        from sentence_transformers import SentenceTransformer
        if model is None:
            model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
    elif use_gemini:
        embeddings = _embed_gemini([c["text"] for c in chunks])
    elif use_openai:
        embeddings = _embed_openai([c["text"] for c in chunks])
    else:
        from sentence_transformers import SentenceTransformer
        if model is None:
            model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    meta_path = index_path.parent / META_FILENAME
    if use_fastembed:
        embedding_backend = "fastembed"
    elif use_local_st:
        embedding_backend = "sentence_transformers"
    elif use_gemini:
        embedding_backend = "gemini"
    elif use_openai:
        embedding_backend = "openai"
    else:
        embedding_backend = "sentence_transformers"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"embedding": embedding_backend}, f)
    return index, chunks
