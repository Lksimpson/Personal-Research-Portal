#!/usr/bin/env python3
"""
Phase 2 baseline RAG: one-command run.
  ingest → chunk → embed/index → retrieve + generate → log

Usage (from project root):
  python run_rag.py [--query "Your question"] [--no-build]
  USE_LOCAL_EMBEDDINGS=1 python run_rag.py ...   # Local embeddings only; API only for answer (if local runs)
  GEMINI_API_KEY=... python run_rag.py ...      # Gemini: 1 batch embed at build, 1 embed + 1 generate per query
  SKIP_QUERY_EMBED=1 python run_rag.py ...      # After index built: keyword retrieval only → 1 API call per query (generate)
  USE_OPENAI=1 python run_rag.py ...           # Use OpenAI for both embed and generate (avoids Gemini 429 quota).
  USE_OPENAI_GENERATE=1 python run_rag.py ...  # Use OpenAI only for the answer step; embed still uses Gemini if set.
  USE_LOCAL_EMBEDDINGS=fastembed python ...    # Local embeddings via FastEmbed (ONNX; avoids segfault on Apple Silicon).

  --query     Run a single query and print answer + log. If omitted, only build (ingest + index).
  --no-build  Skip ingest/index; use existing data/processed and index (faster for repeated queries).
  --diagnose  Run each step in isolation and print progress (to find where Segmentation fault: 11 occurs).
"""
from pathlib import Path
import argparse
import sys
import os

# Load .env first so OPENAI_API_KEY and USE_OPENAI_EMBEDDINGS are set
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# Project root
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Phase 2 RAG: ingest, index, query, log")
    parser.add_argument("--query", type=str, default="", help="Run one query and log")
    parser.add_argument("--no-build", action="store_true", help="Skip ingest and index build")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieve top-k chunks")
    parser.add_argument("--diagnose", action="store_true", help="Print each step to isolate segfault")
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

        # Determine upfront whether we'll use embeddings or keyword-only retrieval
        use_fastembed = os.environ.get("USE_LOCAL_EMBEDDINGS", "").lower() == "fastembed"
        use_local_st = os.environ.get("USE_LOCAL_EMBEDDINGS") == "1"
        use_openai_emb = os.environ.get("USE_OPENAI_EMBEDDINGS") == "1" or os.environ.get("USE_OPENAI") == "1"
        use_gemini_emb = bool(os.environ.get("GEMINI_API_KEY"))
        any_embedding_requested = use_fastembed or use_local_st or use_openai_emb or use_gemini_emb

        if any_embedding_requested:
            # Build semantic index for embeddings-based retrieval
            step("3) Building index (embed + FAISS)...")
            if use_fastembed:
                step("3a) Using FastEmbed (ONNX, local; avoids segfault on Apple Silicon)")
            elif use_local_st:
                step("3a) Using local embeddings (sentence-transformers); API only for answer")
            elif use_openai_emb:
                step("3a) Using OpenAI embeddings")
            elif use_gemini_emb:
                step("3a) Using Gemini embeddings")
            
            from src.rag.embed_index import build_index
            model = None
            if use_fastembed:
                model = None  # FastEmbed used inside build_index
            elif use_local_st:
                from sentence_transformers import SentenceTransformer
                from src.rag.embed_index import MODEL_NAME
                model = SentenceTransformer(MODEL_NAME)
            
            try:
                build_index(chunks, index_path, chunks_path, model=model)
                print(f"✓ Semantic index built: {index_path}")
                print("  Retrieval mode: FAISS (semantic embeddings)")
            except Exception as e:
                print(f"⚠️  Index build failed: {e}")
                print("Falling back to keyword-only retrieval...")
                # Save chunks for keyword-only fallback
                chunks_path.parent.mkdir(parents=True, exist_ok=True)
                import json
                with open(chunks_path, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                print(f"✓ Chunks saved for keyword-only retrieval: {chunks_path}")
                print("  Retrieval mode: Keyword-only (fast, no embeddings needed)")
        else:
            # Keyword-only retrieval: no index needed
            print("3) Skipping index build (keyword-only retrieval mode)")
            chunks_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"✓ Chunks saved: {chunks_path}")
            print("  Retrieval mode: Keyword-only (fast, no embeddings needed)")
    if not args.query:
        print("No --query given; run with --query 'Your question' to get an answer and log.")
        return

    step("4) Loading index and retrieving...")
    from src.rag.retrieve import load_index, load_chunks_only, retrieve
    from src.rag.generate import generate
    from src.rag.log_run import log_run
    from src.rag.citations import append_structured_citations
    from src.eval.metrics import compute_groundedness, compute_answer_relevance

    # Default to keyword-only retrieval (SKIP_QUERY_EMBED=1) unless user explicitly sets embeddings.
    # This avoids segfaults on Apple Silicon with sentence-transformers.
    skip_query_embed = os.environ.get("SKIP_QUERY_EMBED", "1") == "1"  # Default: True
    use_fastembed_embeddings_env = os.environ.get("USE_LOCAL_EMBEDDINGS") == "fastembed"
    use_local_st_env = os.environ.get("USE_LOCAL_EMBEDDINGS") == "1"
    use_openai_emb_env = os.environ.get("USE_OPENAI_EMBEDDINGS") == "1" or os.environ.get("USE_OPENAI") == "1"
    use_gemini_emb_env = bool(os.environ.get("GEMINI_API_KEY"))
    
    # Only use semantic embeddings if user explicitly requested one
    if use_fastembed_embeddings_env or use_local_st_env or use_openai_emb_env or use_gemini_emb_env:
        skip_query_embed = False
    
    if index_path.exists():
        # Index successfully built: use semantic embeddings
        index, chunks_list, use_openai_emb, use_gemini_emb, use_fastembed_emb = load_index(index_path, chunks_path)
        model = None
        if not skip_query_embed and not use_openai_emb and not use_gemini_emb and not use_fastembed_emb:
            # If other embeddings failed, fall back to keyword-only
            step("4a) Could not load semantic embeddings. Falling back to keyword-only retrieval.")
            skip_query_embed = True
    elif chunks_path.exists():
        # Index build failed or not attempted: fall back to keyword-only
        chunks_list = load_chunks_only(chunks_path)
        index, use_openai_emb, use_gemini_emb, use_fastembed_emb = None, False, False, False
        model = None
        skip_query_embed = True  # Force keyword-only retrieval
        if not args.no_build:
            # First run failed to build index: inform user
            step("4) Loading chunks for keyword-only retrieval (index build had failed)")
    else:
        raise FileNotFoundError(
            f"Neither index nor chunks found. Run without --no-build first to ingest and process the PDFs."
        )
    
    if skip_query_embed:
        step("4b) Using keyword-only retrieval (no semantic query embedding)")
    
    retrieved = retrieve(
        args.query, index, chunks_list, model=model,
        top_k=args.top_k, use_openai_embeddings=use_openai_emb, use_gemini_embeddings=use_gemini_emb,
        use_fastembed_embeddings=use_fastembed_emb, skip_query_embed=skip_query_embed,
    )
    chunk_ids = [c.get("chunk_id", "") for c in retrieved]
    # Space out OpenAI calls when using OpenAI for both embed + generate (~2 requests/min)
    if use_openai_emb and os.environ.get("OPENAI_API_KEY"):
        import time
        print("Waiting 35s before generate to stay under rate limit...", flush=True)
        time.sleep(35)
    answer = generate(args.query, retrieved, use_openai=False)
    answer = append_structured_citations(answer, retrieved, manifest_path)
    
    # Compute evaluation metrics
    groundedness = compute_groundedness(answer, retrieved)
    answer_relevance = compute_answer_relevance(answer, args.query)
    
    # Determine which model was used for generation (Ollama is default, APIs only if forced/Ollama failed)
    force_openai = os.environ.get("USE_OPENAI") == "1" or os.environ.get("USE_OPENAI_GENERATE") == "1"
    if force_openai and os.environ.get("OPENAI_API_KEY"):
        model_id = "openai-gpt-4o-mini"
    elif os.environ.get("GEMINI_API_KEY") and force_openai:
        model_id = "gemini-2.0-flash"
    else:
        # Default to Ollama (local LLM, no quota issues)
        ollama_model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
        model_id = f"ollama-{ollama_model}"
    
    # Log with metrics
    log_file = log_run(
        args.query, chunk_ids, answer, log_dir, top_k=args.top_k,
        retrieved=retrieved, groundedness=groundedness, answer_relevance=answer_relevance,
        model_id=model_id
    )

    print("\n" + "="*70)
    print("PHASE 2 RAG PIPELINE: COMPLETE")
    print("="*70)
    
    print("\n[1] RETRIEVAL RESULTS (top-{} chunks):".format(args.top_k))
    print("-" * 70)
    for i, c in enumerate(retrieved, 1):
        source = c.get('source_id', '?')
        chunk = c.get('chunk_id', '?')
        score = c.get('score', 0)
        text_preview = (c.get('text', '')[:80] + '...') if len(c.get('text', '')) > 80 else c.get('text', '')
        print(f"  [{i}] {source} / {chunk} (similarity: {score:.4f})")
        print(f"       Preview: {text_preview}")
    
    print("\n[2] ANSWER WITH CITATIONS:")
    print("-" * 70)
    print(answer)
    
    print("\n[3] LOG ENTRY SAVED:")
    print("-" * 70)
    print(f"  File: {log_file}")
    print(f"  Query: {args.query}")
    print(f"  Metrics: groundedness={groundedness:.4f}, answer_relevance={answer_relevance:.4f}")
    print(f"  Retrieved chunks: {len(retrieved)} sources logged")
    print("\n" + "="*70)
    print("✓ Workflow complete: retrieval + answer + log saved")
    print("="*70)


if __name__ == "__main__":
    main()
