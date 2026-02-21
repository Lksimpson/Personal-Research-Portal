"""
Phase 3 Personal Research Portal — Streamlit UI.
Run from project root: streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root (parent of src)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import streamlit as st
import pandas as pd

# RAG and app imports (after path is set)
from src.rag.retrieve import load_index, load_chunks_only, retrieve
from src.rag.generate import generate
from src.rag.log_run import log_run
from src.rag.citations import append_structured_citations
from src.eval.metrics import compute_groundedness, compute_answer_relevance
from src.app.threads import save_thread, load_threads, get_threads_dir


def get_or_build_rag_state():
    """
    Load or build index/chunks; return (index, chunks_list, model, skip_query_embed,
    use_openai_emb, use_gemini_emb, use_fastembed_emb).
    Cached in st.session_state.rag_state.
    """
    if getattr(st.session_state, "rag_state", None) is not None:
        return st.session_state.rag_state

    manifest_path = ROOT / "data" / "data_manifest.csv"
    index_path = ROOT / "data" / "processed" / "faiss.index"
    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    log_dir = ROOT / "logs"

    # If chunks don't exist, run ingest + chunk (and optionally build index)
    if not chunks_path.exists():
        with st.spinner("First run: ingesting PDFs and building index..."):
            from src.ingest.parse_pdf import run_ingest
            from src.rag.chunk import chunk_corpus
            from src.rag.embed_index import build_index
            import json

            results = run_ingest(manifest_path, ROOT)
            df = pd.read_csv(manifest_path)
            manifest_raw_paths = list(zip(df["source_id"], df["processed_path"]))
            chunks = chunk_corpus(ROOT, manifest_raw_paths)

            use_fastembed = os.environ.get("USE_LOCAL_EMBEDDINGS", "").lower() == "fastembed"
            use_local_st = os.environ.get("USE_LOCAL_EMBEDDINGS") == "1"
            use_openai_emb = os.environ.get("USE_OPENAI_EMBEDDINGS") == "1" or os.environ.get("USE_OPENAI") == "1"
            use_gemini_emb = bool(os.environ.get("GEMINI_API_KEY"))
            any_embedding = use_fastembed or use_local_st or use_openai_emb or use_gemini_emb

            model = None
            if any_embedding:
                try:
                    if use_local_st:
                        from sentence_transformers import SentenceTransformer
                        from src.rag.embed_index import MODEL_NAME
                        model = SentenceTransformer(MODEL_NAME)
                    build_index(chunks, index_path, chunks_path, model=model)
                except Exception:
                    # Fallback: save chunks only (keyword retrieval)
                    chunks_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(chunks_path, "w", encoding="utf-8") as f:
                        json.dump(chunks, f, ensure_ascii=False, indent=2)

            else:
                chunks_path.parent.mkdir(parents=True, exist_ok=True)
                with open(chunks_path, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Load index or chunks only
    skip_query_embed = os.environ.get("SKIP_QUERY_EMBED", "1") == "1"
    use_fastembed_env = os.environ.get("USE_LOCAL_EMBEDDINGS") == "fastembed"
    use_local_st_env = os.environ.get("USE_LOCAL_EMBEDDINGS") == "1"
    use_openai_env = os.environ.get("USE_OPENAI_EMBEDDINGS") == "1" or os.environ.get("USE_OPENAI") == "1"
    use_gemini_env = bool(os.environ.get("GEMINI_API_KEY"))
    if use_fastembed_env or use_local_st_env or use_openai_env or use_gemini_env:
        skip_query_embed = False

    if index_path.exists():
        index, chunks_list, use_openai_emb, use_gemini_emb, use_fastembed_emb = load_index(index_path, chunks_path)
        model = None
        if not skip_query_embed and not use_openai_emb and not use_gemini_emb and not use_fastembed_emb:
            skip_query_embed = True
    else:
        chunks_list = load_chunks_only(chunks_path)
        index = None
        use_openai_emb, use_gemini_emb, use_fastembed_emb = False, False, False
        model = None
        skip_query_embed = True

    state = (index, chunks_list, model, skip_query_embed, use_openai_emb, use_gemini_emb, use_fastembed_emb)
    st.session_state.rag_state = state
    return state


def run_ask(query: str, top_k: int = 5) -> dict | None:
    """Run retrieve → generate → citations → metrics → log_run → save_thread. Returns result dict or None on error."""
    try:
        manifest_path = ROOT / "data" / "data_manifest.csv"
        log_dir = ROOT / "logs"
        index, chunks_list, model, skip_query_embed, use_openai_emb, use_gemini_emb, use_fastembed_emb = get_or_build_rag_state()

        retrieved = retrieve(
            query, index, chunks_list, model=model,
            top_k=top_k,
            use_openai_embeddings=use_openai_emb,
            use_gemini_embeddings=use_gemini_emb,
            use_fastembed_embeddings=use_fastembed_emb,
            skip_query_embed=skip_query_embed,
        )
        answer = generate(query, retrieved, use_openai=False)
        answer = append_structured_citations(answer, retrieved, manifest_path)
        groundedness = compute_groundedness(answer, retrieved)
        answer_relevance = compute_answer_relevance(answer, query)

        force_openai = os.environ.get("USE_OPENAI") == "1" or os.environ.get("USE_OPENAI_GENERATE") == "1"
        if force_openai and os.environ.get("OPENAI_API_KEY"):
            model_id = "openai-gpt-4o-mini"
        elif os.environ.get("GEMINI_API_KEY") and force_openai:
            model_id = "gemini-2.0-flash"
        else:
            model_id = f"ollama-{os.environ.get('OLLAMA_MODEL', 'gemma3:4b')}"

        log_run(
            query, [c.get("chunk_id", "") for c in retrieved], answer, log_dir,
            top_k=top_k, retrieved=retrieved, groundedness=groundedness, answer_relevance=answer_relevance,
            model_id=model_id,
        )
        thread_entry = save_thread(
            ROOT, query, retrieved, answer,
            groundedness=groundedness, answer_relevance=answer_relevance,
        )
        return {
            "query": query,
            "retrieved": retrieved,
            "answer": answer,
            "groundedness": groundedness,
            "answer_relevance": answer_relevance,
            "thread_id": thread_entry["thread_id"],
        }
    except Exception as e:
        st.error(f"Error running RAG: {e}")
        return None


def render_sources(retrieved: list[dict]) -> None:
    """Render retrieved chunks in an expander."""
    for i, c in enumerate(retrieved, 1):
        sid = c.get("source_id", "?")
        cid = c.get("chunk_id", "?")
        score = c.get("score")
        text = (c.get("text") or "")[:400] + ("..." if len(c.get("text") or "") > 400 else "")
        score_str = f" (score: {score:.3f})" if score is not None else ""
        st.markdown(f"**[{i}] {sid} / {cid}**{score_str}")
        st.text(text)
        st.divider()


# --- Page config and sidebar ---
st.set_page_config(page_title="Personal Research Portal", layout="wide")
st.title("Personal Research Portal")
st.caption("Ask research questions over your corpus. Every answer includes citations; missing evidence is stated explicitly.")

page = st.sidebar.radio(
    "Navigate",
    ["Ask", "History", "Artifacts", "Evaluation"],
    index=0,
)

# --- Ask ---
if page == "Ask":
    st.header("Ask a research question")
    query = st.text_area("Question", placeholder="e.g. What is the average commute-time savings when working from home?")
    if st.button("Ask"):
        if not (query or "").strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                result = run_ask(query.strip())
            if result:
                st.session_state.last_result = result
                st.subheader("Answer")
                st.markdown(result["answer"])
                st.caption("Missing evidence is stated explicitly when the corpus does not support a claim.")
                with st.expander("Sources (retrieved chunks)"):
                    render_sources(result["retrieved"])
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Groundedness", f"{result['groundedness']:.3f}")
                with col2:
                    st.metric("Answer relevance", f"{result['answer_relevance']:.3f}")
                st.success(f"Saved to thread {result.get('thread_id', '')} and logs.")

# --- History ---
elif page == "History":
    st.header("Research threads")
    threads = load_threads(ROOT)
    if not threads:
        st.info("No threads yet. Use **Ask** to run a query.")
    else:
        selected_id = st.selectbox(
            "Select a thread",
            options=[t["thread_id"] for t in threads],
            format_func=lambda tid: next(
                (f"{t['thread_id']} — {t['query'][:50]}... ({t.get('timestamp', '')[:10]})"
                for t in threads if t["thread_id"] == tid
            ), None),
            index=0,
        )
        if selected_id:
            thread = next(t for t in threads if t["thread_id"] == selected_id)
            st.subheader("Query")
            st.text(thread["query"])
            st.subheader("Answer")
            st.markdown(thread["answer"])
            with st.expander("Retrieved sources"):
                for i, c in enumerate(thread.get("retrieved", []), 1):
                    sid = c.get("source_id", "?")
                    cid = c.get("chunk_id", "?")
                    score = c.get("score")
                    text = (c.get("text") or "")[:400] + ("..." if len(c.get("text") or "") > 400 else "")
                    score_str = f" (score: {score:.3f})" if score is not None else ""
                    st.markdown(f"**[{i}] {sid} / {cid}**{score_str}")
                    st.text(text)
                    st.divider()
            if thread.get("groundedness") is not None:
                st.caption(f"Groundedness: {thread['groundedness']:.3f}  |  Answer relevance: {thread['answer_relevance']:.3f}")
            st.session_state.selected_thread = thread

# --- Artifacts ---
elif page == "Artifacts":
    st.header("Generate research artifact")
    threads = load_threads(ROOT)
    if not threads:
        st.info("No threads yet. Use **Ask** to run a query, then return here to generate an evidence table.")
    else:
        thread_options = {f"{t['thread_id']} — {t['query'][:40]}...": t for t in threads}
        choice = st.selectbox("Select a thread", list(thread_options.keys()), index=0)
        if choice:
            thread = thread_options[choice]
            if st.button("Generate Evidence Table"):
                from src.app.artifacts import build_evidence_table
                table = build_evidence_table(thread)
                if table:
                    st.session_state.artifact_table = table
                    st.subheader("Evidence table (preview)")
                    st.dataframe(table, use_container_width=True, hide_index=True)
                    st.success("Use the Export section below to download as Markdown, CSV, or PDF.")
                else:
                    st.warning("Could not build evidence table from this thread.")
        if getattr(st.session_state, "artifact_table", None) is not None:
            st.divider()
            st.subheader("Export")
            from src.app.artifacts import export_artifact_markdown, export_artifact_csv, export_artifact_pdf
            table = st.session_state.artifact_table
            col1, col2, col3 = st.columns(3)
            with col1:
                md = export_artifact_markdown(table)
                st.download_button("Download Markdown", md, file_name="evidence_table.md", mime="text/markdown")
            with col2:
                csv_bytes = export_artifact_csv(table)
                st.download_button("Download CSV", csv_bytes, file_name="evidence_table.csv", mime="text/csv")
            with col3:
                pdf_bytes = export_artifact_pdf(table)
                if pdf_bytes:
                    st.download_button("Download PDF", pdf_bytes, file_name="evidence_table.pdf", mime="application/pdf")
                else:
                    st.caption("PDF export requires fpdf2. pip install fpdf2")

# --- Evaluation ---
elif page == "Evaluation":
    st.header("Evaluation summary")
    outputs_dir = ROOT / "outputs"
    eval_files = sorted(outputs_dir.glob("eval_run*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not eval_files:
        st.info(
            "No evaluation runs found. Run the full evaluation from the command line: "
            "`python -m src.eval.run_eval --no-build`"
        )
    else:
        import json
        latest = eval_files[0]
        st.caption(f"Latest run: **{latest.name}**")
        entries = []
        with open(latest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if not entries:
            st.warning("No entries in this eval file.")
        else:
            gr = [e.get("groundedness") for e in entries if e.get("groundedness") is not None]
            rel = [e.get("answer_relevance") for e in entries if e.get("answer_relevance") is not None]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Queries", len(entries))
            with c2:
                st.metric("Mean groundedness", f"{sum(gr)/len(gr):.3f}" if gr else "—")
            with c3:
                st.metric("Mean answer relevance", f"{sum(rel)/len(rel):.3f}" if rel else "—")
            st.subheader("Representative examples")
            # One direct, one synthesis, one edge
            by_type = {}
            for e in entries:
                t = e.get("query_type", "Direct")
                if t not in by_type:
                    by_type[t] = e
            for label, e in [("Direct", by_type.get("Direct", entries[0])), ("Synthesis", by_type.get("Synthesis", entries[min(10, len(entries)-1)] if len(entries) > 10 else entries[-1])), ("Edge case", by_type.get("Edge Case", entries[-1] if entries else None))]:
                if e is None:
                    continue
                with st.expander(f"{label}: {e.get('query_id', '')} — {e.get('query', '')[:60]}..."):
                    st.text("Query: " + e.get("query", ""))
                    st.caption(f"Groundedness: {e.get('groundedness', 0):.3f}  |  Relevance: {e.get('answer_relevance', 0):.3f}")
                    st.text_area("Answer excerpt", value=(e.get("answer", "")[:800] + "..." if len(e.get("answer", "")) > 800 else e.get("answer", "")), height=120, disabled=True)

if __name__ == "__main__":
    pass
