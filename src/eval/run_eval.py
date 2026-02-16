#!/usr/bin/env python3
"""
Run Phase 2 evaluation queries in batch.
- Optionally rebuild index (ingest -> chunk -> embed/index)
- Run each query, retrieve top-k, generate answer, and log results to outputs/eval_runs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_queries(path: Path) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def _infer_model_id() -> str:
    """Infer which model is being used. Priority: Ollama (default) → explicit API requests."""
    force_openai = os.environ.get("USE_OPENAI") == "1" or os.environ.get("USE_OPENAI_GENERATE") == "1"
    if force_openai and os.environ.get("OPENAI_API_KEY"):
        return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    if os.environ.get("OPENAI_API_KEY"):
        # OpenAI key exists but not forced; Ollama still preferred
        pass
    # Default: local Ollama model (no quota issues)
    ollama_model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    return f"ollama-{ollama_model}"


def _build_index_if_needed(manifest_path: Path, index_path: Path, chunks_path: Path, no_build: bool) -> None:
    if no_build:
        return
    from src.ingest.parse_pdf import run_ingest
    from src.rag.chunk import chunk_corpus
    from src.rag.embed_index import build_index
    import pandas as pd

    results = run_ingest(manifest_path, ROOT)
    print(f"Ingest: {len(results)} sources processed")

    df = pd.read_csv(manifest_path)
    manifest_raw_paths = list(zip(df["source_id"], df["processed_path"]))
    chunks = chunk_corpus(ROOT, manifest_raw_paths)
    print(f"Chunks: {len(chunks)} total")

    use_fastembed = os.environ.get("USE_LOCAL_EMBEDDINGS", "").lower() == "fastembed"
    model = None
    use_local_st = os.environ.get("USE_LOCAL_EMBEDDINGS") == "1"
    use_api_embed = (
        os.environ.get("USE_OPENAI") == "1"
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("USE_OPENAI_EMBEDDINGS") == "1"
    )
    if use_fastembed:
        model = None
    elif use_local_st or not use_api_embed:
        from sentence_transformers import SentenceTransformer
        from src.rag.embed_index import MODEL_NAME
        model = SentenceTransformer(MODEL_NAME)
    build_index(chunks, index_path, chunks_path, model=model)
    print(f"Index: {index_path}")


def _load_index_or_chunks(index_path: Path, chunks_path: Path, skip_query_embed: bool):
    from src.rag.retrieve import load_index, load_chunks_only

    if skip_query_embed and not index_path.exists() and chunks_path.exists():
        chunks_list = load_chunks_only(chunks_path)
        index = None
        use_openai_emb = use_gemini_emb = use_fastembed_emb = False
        model = None
        return index, chunks_list, use_openai_emb, use_gemini_emb, use_fastembed_emb, model

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. Run without --no-build first to build the index and chunks."
        )

    index, chunks_list, use_openai_emb, use_gemini_emb, use_fastembed_emb = load_index(index_path, chunks_path)
    model = None
    if not skip_query_embed and not use_openai_emb and not use_gemini_emb and not use_fastembed_emb:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks_list, use_openai_emb, use_gemini_emb, use_fastembed_emb, model


def _write_result(out_path: Path, entry: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_next_eval_run_path():
    outputs_dir = ROOT / "outputs"
    base = "eval_run"
    ext = ".jsonl"
    n = 1
    while True:
        candidate = outputs_dir / f"{base}{n}{ext}"
        if not candidate.exists():
            return candidate
        n += 1

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 eval queries in batch")
    parser.add_argument("--queries", type=str, default=str(ROOT / "src" / "eval" / "queries.jsonl"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-build", action="store_true", help="Skip ingest and index build")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieve top-k chunks")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between queries to avoid rate limits")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of queries to run (0 = all)")
    args = parser.parse_args()

    manifest_path = ROOT / "data" / "data_manifest.csv"
    index_path = ROOT / "data" / "processed" / "faiss.index"
    chunks_path = ROOT / "data" / "processed" / "chunks.json"

    _build_index_if_needed(manifest_path, index_path, chunks_path, args.no_build)

    skip_query_embed = os.environ.get("SKIP_QUERY_EMBED") == "1"
    index, chunks_list, use_openai_emb, use_gemini_emb, use_fastembed_emb, model = _load_index_or_chunks(
        index_path, chunks_path, skip_query_embed
    )

    if use_openai_emb and os.environ.get("OPENAI_API_KEY") and args.sleep_seconds == 0:
        print("Warning: OpenAI embeddings are enabled; consider --sleep-seconds 35 to avoid rate limits.")

    queries = _load_queries(Path(args.queries))
    if args.limit > 0:
        queries = queries[: args.limit]

    from src.rag.retrieve import retrieve
    from src.rag.generate import generate
    from src.rag.citations import append_structured_citations
    from src.eval.metrics import compute_groundedness, compute_answer_relevance

    model_id = _infer_model_id()

    output_path = Path(args.output) if args.output else get_next_eval_run_path()

    for i, q in enumerate(queries, start=1):
        query_text = q.get("query", "")
        query_id = q.get("query_id", f"Q{i:02d}")
        query_type = q.get("type", "unknown")

        retrieved = retrieve(
            query_text,
            index,
            chunks_list,
            model=model,
            top_k=args.top_k,
            use_openai_embeddings=use_openai_emb,
            use_gemini_embeddings=use_gemini_emb,
            use_fastembed_embeddings=use_fastembed_emb,
            skip_query_embed=skip_query_embed,
        )
        answer = generate(query_text, retrieved, use_openai=False)
        answer = append_structured_citations(answer, retrieved, manifest_path)
        # Compute metrics
        # For groundedness, pass the retrieved chunks with their text
        groundedness = compute_groundedness(answer, retrieved)
        answer_relevance = compute_answer_relevance(answer, query_text)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_id": query_id,
            "query_type": query_type,
            "query": query_text,
            "top_k": args.top_k,
            "retrieved": [
                {
                    "source_id": c.get("source_id"),
                    "chunk_id": c.get("chunk_id"),
                    "score": c.get("score"),
                }
                for c in retrieved
            ],
            "answer": answer,
            "model_id": model_id,
            "prompt_id": "phase2_baseline",
            "groundedness": groundedness,
            "answer_relevance": answer_relevance,
        }
        _write_result(output_path, entry)
        print(f"[{i}/{len(queries)}] {query_id} ({query_type}) done")

        if args.sleep_seconds > 0 and i < len(queries):
            time.sleep(args.sleep_seconds)
            # Also log to logs/runs.jsonl
            log_path = ROOT / "logs" / "runs.jsonl"
            _write_result(log_path, {
                "timestamp": entry["timestamp"],
                "query": entry["query"],
                "retrieved_chunk_ids": [c["chunk_id"] for c in entry["retrieved"]],
                "answer": entry["answer"],
                "prompt_id": entry["prompt_id"],
                "model_id": entry["model_id"],
                "top_k": entry["top_k"]
            })


if __name__ == "__main__":
    main()
