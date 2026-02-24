"""
Embed chunks with FastEmbed or sentence-transformers (local only); then index with FAISS.
Priority: FastEmbed (ONNX, no PyTorch; avoids segfault on Apple Silicon) then sentence-transformers.
"""
from pathlib import Path
import json
import numpy as np
import faiss

MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"  # ONNX-based, no PyTorch
META_FILENAME = "embed_meta.json"


def _embed_fastembed(texts: list[str], model_name: str = FASTEMBED_MODEL) -> np.ndarray:
    """Embed via FastEmbed (ONNX-based, no PyTorch). Returns (n, dim) float32, L2-normalized."""
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise ImportError(
            "fastembed not available. Install with: pip install fastembed."
        ) from e
    prefixed = [
        f"passage: {t}" if not t.strip().startswith("passage:") and not t.strip().startswith("query:") else t
        for t in texts
    ]
    embedding_model = TextEmbedding(model_name)
    out = np.array(list(embedding_model.embed(prefixed)), dtype=np.float32)
    faiss.normalize_L2(out)
    return out


def embed_chunks(chunks: list[dict], model=None) -> np.ndarray:
    """Return (n_chunks, dim) array. Tries FastEmbed first, then sentence-transformers."""
    texts = [c["text"] for c in chunks]
    try:
        return _embed_fastembed(texts)
    except Exception:
        pass
    from sentence_transformers import SentenceTransformer
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    faiss.normalize_L2(emb)
    return emb


def build_index(chunks: list[dict], index_path: Path, chunks_path: Path, model=None):
    """
    Embed all chunks with FastEmbed then sentence-transformers on failure; build FAISS index.
    Writes only "fastembed" or "sentence_transformers" to embed_meta.json.
    """
    use_fastembed = False
    try:
        embeddings = _embed_fastembed([c["text"] for c in chunks])
        use_fastembed = True
    except Exception:
        from sentence_transformers import SentenceTransformer
        if model is None:
            model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    meta_path = index_path.parent / META_FILENAME
    embedding_backend = "fastembed" if use_fastembed else "sentence_transformers"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"embedding": embedding_backend}, f)
    return index, chunks
