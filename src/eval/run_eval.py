#!/usr/bin/env python3
"""
Run evaluation queries in batch. Ollama only; local embeddings (FastEmbed then sentence-transformers).
Exposes run_evaluation() for programmatic use (e.g. from the UI).
"""
from __future__ import annotations

import argparse
import json
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
    """Ollama only."""
    ollama_model = __import__("os").environ.get("OLLAMA_MODEL", "gemma3:4b")
    return f"ollama-{ollama_model}"


def _build_index_if_needed(manifest_path: Path, index_path: Path, chunks_path: Path, no_build: bool) -> None:
    if no_build:
        return
    import pandas as pd
    from src.ingest.parse_pdf import run_ingest
    from src.rag.chunk import chunk_corpus
    from src.rag.embed_index import build_index

    results = run_ingest(manifest_path, ROOT)
    df = pd.read_csv(manifest_path)
    manifest_raw_paths = list(zip(df["source_id"], df["processed_path"]))
    chunks = chunk_corpus(ROOT, manifest_raw_paths)
    try:
        build_index(chunks, index_path, chunks_path, model=None)
    except Exception:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)


def _load_index_or_chunks(index_path: Path, chunks_path: Path, skip_query_embed: bool):
    from src.rag.retrieve import load_index, load_chunks_only

    if not index_path.exists() and chunks_path.exists():
        chunks_list = load_chunks_only(chunks_path)
        return None, chunks_list, False, None
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. Run without --no-build first or build from the UI."
        )
    index, chunks_list, use_fastembed_emb = load_index(index_path, chunks_path)
    model = None if use_fastembed_emb else None  # sentence_transformers loaded in retrieve() when needed
    return index, chunks_list, use_fastembed_emb, model


def _write_result(out_path: Path, entry: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_next_eval_run_path(output_dir: Path | None = None) -> Path:
    outputs_dir = output_dir or (ROOT / "outputs" / "eval_runs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    base, ext = "eval_run", ".jsonl"
    n = 1
    while True:
        candidate = outputs_dir / f"{base}{n}{ext}"
        if not candidate.exists():
            return candidate
        n += 1


def run_evaluation(
    queries_path: Path | None = None,
    no_build: bool = True,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    limit: int = 0,
    top_k: int = 5,
    sleep_seconds: float = 0.0,
    progress_callback: callable | None = None,
) -> Path:
    """
    Run the evaluation query set and write results to outputs/eval_runs/eval_runN.jsonl.
    Returns the path to the written file. Uses Ollama only; local embeddings.
    progress_callback(i, total, query_id) is called after each query if provided.
    """
    manifest_path = ROOT / "data" / "data_manifest.csv"
    index_path = ROOT / "data" / "processed" / "faiss.index"
    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    queries_path = queries_path or ROOT / "src" / "eval" / "queries.jsonl"

    _build_index_if_needed(manifest_path, index_path, chunks_path, no_build)
    skip_query_embed = __import__("os").environ.get("SKIP_QUERY_EMBED", "").strip() == "1"
    index, chunks_list, use_fastembed_emb, model = _load_index_or_chunks(index_path, chunks_path, skip_query_embed)

    from src.rag.retrieve import retrieve
    from src.rag.generate import generate
    from src.rag.citations import append_structured_citations
    from src.eval.metrics import (
        compute_groundedness,
        compute_answer_relevance,
        compute_citation_correctness,
        scale_0_1_to_1_4,
        infer_failure_tag,
        infer_notes,
    )

    queries = _load_queries(queries_path)
    if limit > 0:
        queries = queries[:limit]
    model_id = _infer_model_id()
    prompt_id = "phase2_baseline"
    task_id = "eval_batch"
    out_dir = output_dir or (ROOT / "outputs" / "eval_runs")
    result_path = output_path or get_next_eval_run_path(out_dir)

    for i, q in enumerate(queries, start=1):
        query_text = q.get("query", "")
        query_id = q.get("query_id", f"Q{i:02d}")
        test_case_id = q.get("test_case_id", query_id)
        query_type = q.get("type", "unknown")
        retrieved = retrieve(
            query_text,
            index,
            chunks_list,
            model=model,
            top_k=top_k,
            use_fastembed_embeddings=use_fastembed_emb,
            skip_query_embed=skip_query_embed,
        )
        answer = generate(query_text, retrieved)
        answer = append_structured_citations(answer, retrieved, manifest_path)
        groundedness = compute_groundedness(answer, retrieved)
        answer_relevance = compute_answer_relevance(answer, query_text)
        citation_correctness = compute_citation_correctness(answer, retrieved)

        score_groundedness_1_4 = scale_0_1_to_1_4(groundedness)
        score_usefulness_1_4 = scale_0_1_to_1_4(answer_relevance)
        score_citation_correctness_1_4 = citation_correctness

        retrieved_evidence_ids = [c.get("chunk_id", "") for c in retrieved if c.get("chunk_id")]

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "test_case_id": test_case_id,
            "query_id": query_id,
            "query_type": query_type,
            "query": query_text,
            "prompt_id": prompt_id,
            "model_id": model_id,
            "retrieved_evidence_ids": retrieved_evidence_ids,
            "top_k": top_k,
            "retrieved": [{"source_id": c.get("source_id"), "chunk_id": c.get("chunk_id"), "score": c.get("score")} for c in retrieved],
            "answer": answer,
            "groundedness": groundedness,
            "answer_relevance": answer_relevance,
            "score_groundedness_1_4": score_groundedness_1_4,
            "score_citation_correctness_1_4": score_citation_correctness_1_4,
            "score_usefulness_1_4": score_usefulness_1_4,
            "failure_tag": "",
            "notes": "",
        }
        entry["failure_tag"] = infer_failure_tag(entry)
        entry["notes"] = infer_notes(entry)
        _write_result(result_path, entry)
        if progress_callback:
            progress_callback(i, len(queries), query_id)
        if sleep_seconds > 0 and i < len(queries):
            time.sleep(sleep_seconds)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval queries in batch (Ollama only)")
    parser.add_argument("--queries", type=str, default=str(ROOT / "src" / "eval" / "queries.jsonl"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-build", action="store_true", help="Skip ingest and index build")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    def progress(i: int, total: int, query_id: str) -> None:
        print(f"[{i}/{total}] {query_id} done", flush=True)

    path = run_evaluation(
        queries_path=Path(args.queries),
        no_build=args.no_build,
        output_path=output_path,
        limit=args.limit,
        top_k=args.top_k,
        sleep_seconds=args.sleep_seconds,
        progress_callback=progress,
    )
    queries = _load_queries(Path(args.queries))
    n = args.limit if args.limit > 0 else len(queries)
    print(f"Evaluation complete. {n} queries written to {path}")


if __name__ == "__main__":
    main()
