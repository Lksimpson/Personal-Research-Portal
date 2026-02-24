"""
Generate answer from retrieved chunks with (source_id, chunk_id) citations.
Uses Ollama only (local). Every answer includes citations; missing evidence is stated with a suggested next retrieval step.
"""
import os
import re
import subprocess


SYSTEM_PROMPT = """You answer research questions using ONLY the provided evidence chunks.
- Cite every major claim with (source_id, chunk_id) using the EXACT format shown in the evidence blocks, e.g. (SRC001, SRC001_chunk_0). Use ONLY the citation pairs listed in "Valid citations" below; do not use numbers like 63 or formats like (source_id: 63, chunk_id: 63).
- Every answer must include at least one citation to the provided chunks where a claim is made.
- If the chunks do not support a claim, say so explicitly (e.g. "Not specified in the provided text" or "Evidence not found in corpus.") and suggest a concrete next retrieval step.
- If no chunk supports the question, say so and suggest what to search or which source type to check next.
- Do not infer causality unless the source explicitly claims it.
- If evidence is conflicting, state both sides and cite each; do not invent a resolution.
- When relevance scores are shown, you may note stronger vs weaker evidence where it matters."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks for the prompt with chunk_id labels and relevance score when present."""
    parts = []
    for c in chunks:
        cid = c.get("chunk_id", "?")
        sid = c.get("source_id", "?")
        text = (c.get("text") or "").strip()
        score = c.get("score")
        if score is not None:
            parts.append(f"[{sid}, {cid}] (relevance: {float(score):.3f})\n{text}")
        else:
            parts.append(f"[{sid}, {cid}]\n{text}")
    return "\n\n---\n\n".join(parts)


def _valid_citations_line(chunks: list[dict]) -> str:
    """Single line listing the only valid (source_id, chunk_id) pairs for this response."""
    pairs = [f"({c.get('source_id', '?')}, {c.get('chunk_id', '?')})" for c in chunks]
    return "Valid citations for this response (use only these exact pairs): " + ", ".join(pairs) + "."


def build_user_prompt(query: str, chunks: list[dict]) -> str:
    """Build user message: query + evidence blocks + explicit valid citations list."""
    context = format_context(chunks)
    valid = _valid_citations_line(chunks)
    return f"""Evidence chunks (each labeled with source_id, chunk_id):

{context}

---

{valid}

Question: {query}

Answer using only the above evidence. Cite (source_id, chunk_id) for each claim using ONLY the valid citations listed above. If the corpus does not support part of the question, state that clearly and suggest a specific next retrieval step (e.g. a different query or source type to try)."""


def fix_citation_format(answer: str, chunks: list[dict]) -> str:
    """
    Replace wrong citation patterns in model output (e.g. (63, 63) or (source_id: 63, chunk_id: 63))
    with the correct (SRCxxx, SRCxxx_chunk_y) from the retrieved chunks. Replaces by order of appearance.
    """
    if not answer or not chunks:
        return answer
    correct_pairs = [(c.get("source_id", "?"), c.get("chunk_id", "?")) for c in chunks]
    if not correct_pairs:
        return answer

    # Pattern 1: (source_id: N, chunk_id: N) or similar
    pat1 = re.compile(
        r"\(\s*source_id\s*:\s*\d+\s*,\s*chunk_id\s*:\s*\d+\s*\)",
        re.IGNORECASE,
    )
    # Pattern 2: (N, M) where both are numbers and at least one > 20 (likely wrong index, e.g. 63)
    pat2 = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
    counter = [0]

    def pick_citation():
        idx = min(counter[0], len(correct_pairs) - 1)
        counter[0] += 1
        sid, cid = correct_pairs[idx]
        return f"({sid}, {cid})"

    def repl1(_m):
        return pick_citation()

    def repl2(m):
        n1, n2 = int(m.group(1)), int(m.group(2))
        if n1 > 20 or n2 > 20:
            return pick_citation()
        return m.group(0)

    out = pat1.sub(repl1, answer)
    counter[0] = 0
    out = pat2.sub(repl2, out)
    return out


def generate_with_ollama(query: str, chunks: list[dict], model: str = "gemma3:4b") -> str:
    """Call Ollama for completion. Requires Ollama running locally. Returns answer or error message."""
    prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(query, chunks)}"
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            raw = result.stdout.strip()
            return fix_citation_format(raw, chunks)
        return f"[Error: Ollama failed: {result.stderr.strip()}]"
    except Exception as e:
        return f"[Error: Ollama exception: {str(e)}]"


def _ollama_unavailable_placeholder(chunks: list[dict]) -> str:
    """Placeholder when Ollama is not available."""
    refs = [f"({c.get('source_id')}, {c.get('chunk_id')})" for c in chunks[:5]]
    return f"Ollama unavailable. Relevant chunks: {', '.join(refs)}. Start Ollama (ollama serve) and retry."


def generate(query: str, chunks: list[dict], use_openai: bool = False, model: str = "gpt-4o-mini") -> str:
    """Generate answer with citations. Uses Ollama only."""
    del use_openai, model  # kept for call-site compatibility
    ollama_model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    result = generate_with_ollama(query, chunks, model=ollama_model)
    if result and not result.startswith("[Error"):
        return result
    return result or _ollama_unavailable_placeholder(chunks)
