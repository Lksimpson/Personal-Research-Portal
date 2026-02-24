import re


def _words(s: str) -> set:
    """Lowercased word set for overlap checks (skip very short tokens)."""
    return set(w.lower() for w in re.findall(r"\w+", s) if len(w) > 1)


def compute_groundedness(answer: str, retrieved_chunks: list) -> float:
    """
    Groundedness: fraction of answer sentences that are supported by the retrieved context.
    A sentence is counted as grounded if at least OVERLAP_THRESHOLD of its words appear in
    the context (allows paraphrasing; a verbatim-sentence match would be too strict).
    """
    OVERLAP_THRESHOLD = 0.4  # fraction of sentence words that must appear in context
    answer_sents = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    context = " ".join([c.get("text", "") or "" for c in retrieved_chunks])
    context_words = _words(context.lower())
    grounded = 0
    total = 0
    for sent in answer_sents:
        if not sent or len(sent) < 3:
            continue
        sent_words = _words(sent)
        if not sent_words:
            continue
        total += 1
        in_context = sum(1 for w in sent_words if w in context_words)
        if in_context / len(sent_words) >= OVERLAP_THRESHOLD:
            grounded += 1
    return grounded / total if total else 0.0


def compute_answer_relevance(answer: str, query: str) -> float:
    """
    Simple answer relevance: returns the fraction of query keywords found in the answer.
    """
    query_words = set(re.findall(r'\w+', query.lower()))
    answer_words = set(re.findall(r'\w+', answer.lower()))
    if not query_words:
        return 0.0
    overlap = query_words & answer_words
    return len(overlap) / len(query_words)


# Citation pattern used for correctness check (must match retrieved chunk IDs)
_CITATION_PATTERN = re.compile(r"\(SRC\d+,\s*SRC\d+_chunk_\d+\)", re.IGNORECASE)


def compute_citation_correctness(answer: str, retrieved: list) -> int:
    """
    Score 1–4: citation correctness. 4 = citations present and match retrieved; 1 = no valid citations.
    """
    valid_pairs = {(c.get("source_id"), c.get("chunk_id")) for c in retrieved if c.get("source_id") and c.get("chunk_id")}
    matches = list(_CITATION_PATTERN.findall(answer))
    if not matches:
        return 1  # No citations
    # Normalize match to (source_id, chunk_id) and check if in valid_pairs
    correct = 0
    for m in matches:
        # m is like "(SRC001, SRC001_chunk_0)" -> extract SRC001 and SRC001_chunk_0
        inner = m[1:-1].replace(" ", "")
        parts = inner.split(",")
        if len(parts) == 2 and (parts[0].strip(), parts[1].strip()) in valid_pairs:
            correct += 1
    if correct == 0:
        return 2  # Citations present but none match retrieved
    if correct < len(matches):
        return 3  # Some match, some don't
    return 4  # All match (or only one citation and it matches)


def scale_0_1_to_1_4(x: float | None) -> int:
    """Map a 0–1 score to 1–4 (1=worst, 4=best). None -> 1."""
    if x is None:
        return 1
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 1
    x = max(0.0, min(1.0, x))
    if x >= 0.75:
        return 4
    if x >= 0.5:
        return 3
    if x >= 0.25:
        return 2
    return 1


def infer_failure_tag(entry: dict) -> str:
    """
    Infer a failure tag from scores and answer: missing_evidence, wrong_citation, overconfident, or empty.
    """
    g_14 = entry.get("score_groundedness_1_4")
    if g_14 is None and entry.get("groundedness") is not None:
        g_14 = scale_0_1_to_1_4(entry["groundedness"])
    g_14 = g_14 if g_14 is not None else 1
    cit = entry.get("score_citation_correctness_1_4", 1)
    answer = (entry.get("answer") or "").lower()
    if g_14 <= 2 and ("not found" in answer or "not in corpus" in answer or "evidence not found" in answer):
        return "missing_evidence"
    if cit <= 2:
        return "wrong_citation"
    if g_14 <= 2 and ("suggested next" not in answer and "suggest a" not in answer):
        return "overconfident"
    return ""


def infer_notes(entry: dict) -> str:
    """Short notes for the evaluation row (e.g. low groundedness, no citation)."""
    parts = []
    if (entry.get("score_groundedness_1_4") or 1) <= 2:
        parts.append("Low groundedness")
    if (entry.get("score_citation_correctness_1_4") or 1) <= 2:
        parts.append("Citation issues")
    if (entry.get("score_usefulness_1_4") or 1) <= 2:
        parts.append("Low relevance to query")
    tag = entry.get("failure_tag", "")
    if tag and tag not in str(parts):
        parts.append(tag)
    return "; ".join(parts) if parts else ""
