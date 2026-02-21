"""
File-based research threads: save and load query + retrieved evidence + answer.
Each thread: thread_id, query, retrieved, answer, timestamp, groundedness, answer_relevance.
"""
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone


def get_threads_dir(root: Path) -> Path:
    """Return threads directory; create if needed."""
    d = root / "threads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _threads_file(root: Path) -> Path:
    return get_threads_dir(root) / "threads.jsonl"


def save_thread(
    root: Path,
    query: str,
    retrieved: list[dict],
    answer: str,
    groundedness: float = None,
    answer_relevance: float = None,
    thread_id: str = None,
) -> dict:
    """
    Append one thread to threads/threads.jsonl.
    retrieved: list of chunk dicts (source_id, chunk_id, score, text optional).
    Returns the saved thread dict (with thread_id, timestamp).
    """
    thread_id = thread_id or str(uuid.uuid4())[:8]
    # Store minimal retrieved for display (include text snippet for artifact generation)
    retrieved_serializable = [
        {
            "source_id": c.get("source_id"),
            "chunk_id": c.get("chunk_id"),
            "score": c.get("score"),
            "text": (c.get("text") or "")[:2000],  # truncate for file size
        }
        for c in retrieved
    ]
    entry = {
        "thread_id": thread_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "retrieved": retrieved_serializable,
        "answer": answer,
        "groundedness": groundedness,
        "answer_relevance": answer_relevance,
    }
    path = _threads_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load_threads(root: Path, limit: int = 100) -> list[dict]:
    """
    Load most recent threads from threads/threads.jsonl (newest last in file order).
    Returns list of thread dicts, most recent last (so reverse for "newest first").
    """
    path = _threads_file(root)
    if not path.exists():
        return []
    threads = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                threads.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # File is append-only so last entries are newest; take last `limit` and reverse for newest-first
    return list(reversed(threads[-limit:]))
