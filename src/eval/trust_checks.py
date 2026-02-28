"""
Trust behavior checks: citations present and suggested next retrieval step when evidence is missing.
Exposes run_trust_behavior_checks() for UI and tests.
"""
from __future__ import annotations

import re
import shutil


def _ollama_available() -> bool:
    """Return True if ollama binary is on PATH."""
    return shutil.which("ollama") is not None


# Citation pattern: (SRC001, SRC001_chunk_0) or (SRC001, SRC001_chunk_1)
CITATION_PATTERN = re.compile(r"\(SRC\d+,\s*SRC\d+_chunk_\d+\)", re.IGNORECASE)

# Phrases indicating a suggested next retrieval step
SUGGESTED_NEXT_PHRASES = [
    "suggested",
    "try searching",
    "consider",
    "next step",
    "query such as",
    "next retrieval",
    "suggest a",
    "would resolve",
    "search for",
    "look for",
    "check for",
    "another query",
    "different search",
    "additional sources",
    "further research",
    "more specific",
]

# Phrases indicating missing/absent evidence
MISSING_EVIDENCE_PHRASES = [
    "not found",
    "not in corpus",
    "does not support",
    "evidence not found",
    "not specified",
    "no chunk supports",
    "corpus does not",
    "does not mention",
    "does not contain",
    "no evidence",
    "not present",
    "not available",
    "cannot answer",
    "cannot find",
    "no information",
    "lacks evidence",
    "not addressed",
    "not discussed",
    "no data",
    "not in the provided",
    "doesn't mention",
    "doesn't contain",
]


def _has_citation(answer: str) -> bool:
    """Return True if answer contains at least one (source_id, chunk_id) citation."""
    return bool(CITATION_PATTERN.search(answer))


def _has_suggested_next_step(answer: str) -> bool:
    """Return True if answer contains a phrase suggesting a next retrieval step."""
    lower = answer.lower()
    return any(phrase in lower for phrase in SUGGESTED_NEXT_PHRASES)


def _has_missing_evidence_phrase(answer: str) -> bool:
    """Return True if answer indicates missing/absent evidence."""
    lower = answer.lower()
    return any(phrase in lower for phrase in MISSING_EVIDENCE_PHRASES)


def run_trust_behavior_checks(
    generate_fn=None,
) -> dict:
    """
    Run trust behavior checks: (1) answer contains citation, (2) when evidence is missing, answer suggests next step.
    Uses generate() with mock chunks. Returns dict with citations_ok, suggested_next_step_ok, skipped, message.
    If Ollama is not available, returns skipped=True.
    """
    result = {
        "citations_ok": False,
        "suggested_next_step_ok": False,
        "skipped": False,
        "message": "",
    }
    if not _ollama_available():
        result["skipped"] = True
        result["message"] = "Ollama not available (ollama not on PATH). Start Ollama and retry."
        return result

    if generate_fn is None:
        from src.rag.generate import generate
        generate_fn = generate

    # Check 1: citation presence — use chunks that have content so the model can cite
    mock_chunks_with_evidence = [
        {
            "source_id": "SRC001",
            "chunk_id": "SRC001_chunk_0",
            "text": "Remote work saves commute time. Average daily savings in the survey was 48 minutes.",
            "score": 0.8,
        },
    ]
    query_with_evidence = "What is the average commute-time savings when working from home?"
    try:
        answer1 = generate_fn(query_with_evidence, mock_chunks_with_evidence)
    except Exception as e:
        result["message"] = f"Generate failed: {e}"
        return result

    result["citations_ok"] = _has_citation(answer1)
    if not result["citations_ok"]:
        result["message"] = "Citation check failed: answer did not contain (source_id, chunk_id) citation."
        return result

    # Check 2: when evidence is missing, answer should suggest a next step (and state missing evidence)
    mock_chunks_no_evidence = [
        {
            "source_id": "SRC001",
            "chunk_id": "SRC001_chunk_0",
            "text": "This passage is about productivity in offices. It does not mention promotion rates.",
            "score": 0.3,
        },
    ]
    query_no_evidence = "Does this corpus contain evidence that fully remote work increases promotion rates?"
    try:
        answer2 = generate_fn(query_no_evidence, mock_chunks_no_evidence)
    except Exception as e:
        result["message"] = f"Citation check passed. Suggested-next-step generate failed: {e}"
        return result

    has_missing = _has_missing_evidence_phrase(answer2)
    has_suggested = _has_suggested_next_step(answer2)
    result["suggested_next_step_ok"] = has_missing and has_suggested
    if not result["suggested_next_step_ok"]:
        result["message"] = (
            "Suggested next step check failed: when evidence is missing, answer should state that and suggest a next retrieval step. "
            f"Missing-evidence phrase: {has_missing}; Suggested-next phrase: {has_suggested}."
        )
        return result

    result["message"] = "All checks passed: citations present and suggested next step when evidence is missing."
    return result
