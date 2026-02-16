"""
Log each run: query, retrieved chunk IDs, model output, prompt/version ID.
Appends to logs/runs.jsonl (one JSON object per line).
"""
from pathlib import Path
import json
from datetime import datetime, timezone


def log_run(
    query: str,
    chunk_ids: list[str],
    answer: str,
    log_dir: Path,
    prompt_id: str = "phase2_baseline",
    model_id: str = "ollama-gemma3:4b",
    top_k: int = 5,
    retrieved: list[dict] = None,
    groundedness: float = None,
    answer_relevance: float = None,
) -> Path:
    """Append one run to logs/runs.jsonl with optional metrics."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "runs.jsonl"
    
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "retrieved_chunk_ids": chunk_ids,
        "answer": answer,
        "prompt_id": prompt_id,
        "model_id": model_id,
        "top_k": top_k,
    }
    
    # Add retrieved chunks if provided (for evidence strength and source tracking)
    if retrieved:
        entry["retrieved"] = [
            {
                "source_id": c.get("source_id"),
                "chunk_id": c.get("chunk_id"),
                "score": c.get("score"),
            }
            for c in retrieved
        ]
    
    # Add evaluation metrics if provided
    if groundedness is not None:
        entry["groundedness"] = groundedness
    if answer_relevance is not None:
        entry["answer_relevance"] = answer_relevance
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return log_file
