#!/usr/bin/env python3
"""
RAG pipeline: ingest → chunk → embed/index → retrieve + generate → log.
Uses Ollama only for generation. Semantic retrieval by default (FastEmbed then sentence-transformers); keyword fallback when index is missing or query embed fails.

Usage (from project root):
  python run_rag.py [--query "Your question"] [--no-build]
  SKIP_QUERY_EMBED=1 python run_rag.py ...  # Force keyword-only retrieval.

  --query     Run a single query and print answer + log. If omitted, only build (ingest + index).
  --no-build  Skip ingest/index; use existing data/processed and index.
  --diagnose  Print each step to isolate failures.
"""
from pathlib import Path
import argparse
import sys
import os
import json

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="RAG: ingest, index, query, log (Ollama only)")
    parser.add_argument("--query", type=str, default="", help="Run one query and log")
    parser.add_argument("--no-build", action="store_true", help="Skip ingest and index build")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieve top-k chunks")
    parser.add_argument("--diagnose", action="store_true", help="Print each step")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    manifest_path = ROOT / "data" / "data_manifest.csv"
    index_path = ROOT / "data" / "processed" / "faiss.index"
    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    log_dir = ROOT / "logs"

    def step(msg: str) -> None:
        if args.diagnose:
            print(f"[diagnose] {msg}", flush=True)

    if not args.no_build:
        step("1) Starting ingest...")
        from src.ingest.parse_pdf import run_ingest
        results = run_ingest(manifest_path, ROOT)
        print(f"Ingest: {len(results)} sources processed")
        for r in results:
            if "error" in r:
                print(f"  Error {r['source_id']}: {r['error']}")
            else:
                print(f"  {r['source_id']}: {r.get('chars', 0)} chars -> {r.get('path', '')}")

        step("2) Starting chunking...")
        import pandas as pd
        from src.rag.chunk import chunk_corpus
        df = pd.read_csv(manifest_path)
        manifest_raw_paths = list(zip(df["source_id"], df["processed_path"]))
        chunks = chunk_corpus(ROOT, manifest_raw_paths)
        print(f"Chunks: {len(chunks)} total")

        step("3) Building index (FastEmbed, then sentence-transformers on failure)...")
        from src.rag.embed_index import build_index
        try:
            build_index(chunks, index_path, chunks_path, model=None)
            print(f"✓ Semantic index built: {index_path}")
            print("  Retrieval mode: FAISS (semantic)")
        except Exception as e:
            print(f"⚠️  Index build failed: {e}")
            print("Falling back to keyword-only retrieval...")
            chunks_path.parent.mkdir(parents=True, exist_ok=True)
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"✓ Chunks saved: {chunks_path}")
            print("  Retrieval mode: Keyword-only")

    if not args.query:
        print("No --query given; run with --query 'Your question' to get an answer and log.")
        return

    step("4) Loading index and retrieving...")
    from src.rag.retrieve import load_index, load_chunks_only, retrieve
    from src.rag.generate import generate
    from src.rag.log_run import log_run
    from src.rag.citations import append_structured_citations
    from src.eval.metrics import compute_groundedness, compute_answer_relevance

    # Semantic default; keyword only when no index or SKIP_QUERY_EMBED=1
    skip_query_embed = os.environ.get("SKIP_QUERY_EMBED", "").strip() == "1"

    if index_path.exists():
        index, chunks_list, use_fastembed_emb = load_index(index_path, chunks_path)
        model = None
        if skip_query_embed:
            step("4a) Using keyword-only retrieval (SKIP_QUERY_EMBED=1)")
    elif chunks_path.exists():
        chunks_list = load_chunks_only(chunks_path)
        index, use_fastembed_emb, model = None, False, None
        skip_query_embed = True
        if not args.no_build:
            step("4) Loading chunks for keyword-only retrieval (index build had failed)")
    else:
        raise FileNotFoundError(
            "Neither index nor chunks found. Run without --no-build first to ingest and process the PDFs."
        )

    retrieved = retrieve(
        args.query, index, chunks_list, model=model,
        top_k=args.top_k,
        use_fastembed_embeddings=use_fastembed_emb,
        skip_query_embed=skip_query_embed,
    )
    chunk_ids = [c.get("chunk_id", "") for c in retrieved]
    answer = generate(args.query, retrieved)
    answer = append_structured_citations(answer, retrieved, manifest_path)

    groundedness = compute_groundedness(answer, retrieved)
    answer_relevance = compute_answer_relevance(answer, args.query)
    ollama_model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    model_id = f"ollama-{ollama_model}"

    log_file = log_run(
        args.query, chunk_ids, answer, log_dir, top_k=args.top_k,
        retrieved=retrieved, groundedness=groundedness, answer_relevance=answer_relevance,
        model_id=model_id,
    )

    print("\n" + "=" * 70)
    print("RAG PIPELINE: COMPLETE")
    print("=" * 70)
    print("\n[1] RETRIEVAL RESULTS (top-{} chunks):".format(args.top_k))
    print("-" * 70)
    for i, c in enumerate(retrieved, 1):
        source = c.get("source_id", "?")
        chunk = c.get("chunk_id", "?")
        score = c.get("score", 0)
        text_preview = (c.get("text", "")[:80] + "...") if len(c.get("text", "")) > 80 else c.get("text", "")
        print(f"  [{i}] {source} / {chunk} (similarity: {score:.4f})")
        print(f"       Preview: {text_preview}")
    print("\n[2] ANSWER WITH CITATIONS:")
    print("-" * 70)
    print(answer)
    print("\n[3] LOG ENTRY SAVED:")
    print("-" * 70)
    print(f"  File: {log_file}")
    print(f"  Metrics: groundedness={groundedness:.4f}, answer_relevance={answer_relevance:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
