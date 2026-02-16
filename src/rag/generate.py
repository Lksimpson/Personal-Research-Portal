import subprocess

def generate_with_ollama(query: str, chunks: list[dict], model: str = "llama2") -> str:
    """Call Ollama for completion. Requires Ollama running locally. Returns answer or error message."""
    prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(query, chunks)}"
    try:
        result = subprocess.run([
            "ollama", "run", model, prompt
        ], capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return result.stdout.strip()
        return f"[Error: Ollama failed: {result.stderr.strip()}]"
    except Exception as e:
        return f"[Error: Ollama exception: {str(e)}]"

"""
Generate answer from retrieved chunks with (source_id, chunk_id) citations.
Uses Ollama as default (local, avoids API quotas). APIs available but require explicit opt-in.
"""
import os
import json
from pathlib import Path


SYSTEM_PROMPT = """You answer research questions using ONLY the provided evidence chunks.
- Cite every major claim with (source_id, chunk_id). Example: (SRC001, SRC001_chunk_0).
- If the chunks do not support a claim, say "Not specified in the provided text" or "Evidence not found in corpus."
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


def build_user_prompt(query: str, chunks: list[dict]) -> str:
    """Build user message: query + evidence blocks."""
    context = format_context(chunks)
    return f"""Evidence chunks (each labeled with source_id, chunk_id):

{context}

---

Question: {query}

Answer using only the above evidence. Cite (source_id, chunk_id) for each claim. If the corpus does not support something, say so explicitly."""


def _quota_placeholder(provider: str, chunks: list[dict], detail: str = "") -> str:
    """Placeholder when API quota is exceeded so the pipeline still completes (run logs, eval set)."""
    refs = [f"({c.get('source_id')}, {c.get('chunk_id')})" for c in chunks[:5]]
    msg = f"[Answer skipped: {provider} quota exceeded (429). Retrieval completed; cited chunks: {', '.join(refs)}.]"
    if detail:
        msg += f" Detail: {detail[:200]}"
    return msg


def _extract_gemini_text(resp) -> str:
    """Best-effort extraction of text from a google-genai response."""
    text = getattr(resp, "text", None)
    if text:
        return text.strip()
    candidates = getattr(resp, "candidates", None)
    if candidates:
        parts = getattr(candidates[0].content, "parts", [])
        if parts and getattr(parts[0], "text", None):
            return parts[0].text.strip()
    return ""


def generate_with_gemini(query: str, chunks: list[dict], model: str = "models/gemini-2.0-flash") -> str:
    """Call Gemini API for completion. Set GEMINI_API_KEY in env. On 429 returns placeholder so pipeline can complete."""
    try:
        from google import genai
    except ImportError:
        return "[Error: google-genai not installed. pip install google-genai]"
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return "[Error: GEMINI_API_KEY not set]"
    client = genai.Client(api_key=key)
    config = genai.types.GenerateContentConfig(temperature=0.2)
    full_prompt = f"""{SYSTEM_PROMPT}

{build_user_prompt(query, chunks)}"""
    try:
        resp = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=config,
        )
        text = _extract_gemini_text(resp)
        if text:
            return text
        return "[Error: Gemini returned no text]"
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "resourceexhausted" in err_str:
            return _quota_placeholder("Gemini", chunks, str(e))
        raise


def generate_with_openai(query: str, chunks: list[dict], model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API for completion. Set OPENAI_API_KEY in env. Retries on connection errors; on 429 returns placeholder so pipeline can complete."""
    try:
        from openai import OpenAI, APIConnectionError, RateLimitError
    except ImportError:
        return "[Error: openai not installed. pip install openai]"
    import time
    client = OpenAI()
    user = build_user_prompt(query, chunks)
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()
        except RateLimitError as e:
            # 429 / insufficient_quota: return placeholder so Phase 2 pipeline can complete (run logs, eval set)
            return _quota_placeholder("OpenAI", chunks, str(e))
        except APIConnectionError:
            if attempt == 3:
                raise
            time.sleep(2 * (2 ** attempt))


def generate_with_placeholder(query: str, chunks: list[dict]) -> str:
    """Return a placeholder answer when no API key is configured."""
    refs = [f"({c.get('source_id')}, {c.get('chunk_id')})" for c in chunks[:5]]
    return f"[Placeholder: set GEMINI_API_KEY or OPENAI_API_KEY for real generation.]\n\nRelevant chunks: {', '.join(refs)}"


def generate(query: str, chunks: list[dict], use_openai: bool = False, model: str = "gpt-4o-mini") -> str:
    """Generate answer with citations. Priority: Ollama → explicit API requests → fallback Ollama."""
    force_openai = os.environ.get("USE_OPENAI") == "1" or os.environ.get("USE_OPENAI_GENERATE") == "1"
    
    # If Ollama is running, use it by default (local, no quota issues)
    ollama_model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    ollama_result = generate_with_ollama(query, chunks, model=ollama_model)
    if ollama_result and not ollama_result.startswith("[Error"):
        return ollama_result
    
    # Only try APIs if explicitly requested or if Ollama failed
    if force_openai and os.environ.get("OPENAI_API_KEY"):
        return generate_with_openai(query, chunks, model=model)
    
    if os.environ.get("GEMINI_API_KEY"):
        gemini_model = os.environ.get("GEMINI_MODEL", "models/gemini-2.0-flash")
        try:
            return generate_with_gemini(query, chunks, model=gemini_model)
        except Exception:
            pass
    
    if use_openai and os.environ.get("OPENAI_API_KEY"):
        try:
            return generate_with_openai(query, chunks, model=model)
        except Exception:
            pass
    
    # Final fallback: return Ollama error or placeholder
    return ollama_result or generate_with_placeholder(query, chunks)
