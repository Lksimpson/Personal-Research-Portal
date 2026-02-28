"""
Phase 3 Personal Research Portal — Streamlit UI. Ollama only; semantic retrieval by default.
Run from project root: streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Avoid Hugging Face tokenizers parallelism warnings when Streamlit forks/reloads.
# Set this before any code that may import/use `sentence_transformers` or `tokenizers`.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
import pandas as pd

from src.rag.retrieve import load_index, load_chunks_only, retrieve
from src.rag.generate import generate
from src.rag.log_run import log_run
from src.rag.citations import append_structured_citations
from src.eval.metrics import compute_groundedness, compute_answer_relevance
from src.app.threads import save_thread, load_threads


def _do_build_index() -> dict:
    """Run ingest → chunk → build index. Returns {success, backend, message}."""
    manifest_path = ROOT / "data" / "data_manifest.csv"
    index_path = ROOT / "data" / "processed" / "faiss.index"
    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    from src.ingest.parse_pdf import run_ingest
    from src.rag.chunk import chunk_corpus
    from src.rag.embed_index import build_index

    try:
        results = run_ingest(manifest_path, ROOT)
    except Exception as e:
        return {"success": False, "backend": "keyword_only", "message": f"Ingest failed: {e}. Ensure PDFs are in data/raw/."}
    df = pd.read_csv(manifest_path)
    manifest_raw_paths = list(zip(df["source_id"], df["processed_path"]))
    try:
        chunks = chunk_corpus(ROOT, manifest_raw_paths)
    except Exception as e:
        return {"success": False, "backend": "keyword_only", "message": f"Chunking failed: {e}."}
    try:
        build_index(chunks, index_path, chunks_path, model=None)
        meta_path = index_path.parent / "embed_meta.json"
        backend = "keyword_only"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                backend = json.load(f).get("embedding", "sentence_transformers")
        return {"success": True, "backend": backend, "message": f"Index built with {backend}. {len(chunks)} chunks."}
    except Exception as e:
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        return {"success": False, "backend": "keyword_only", "message": f"Index build failed: {e}. Chunks saved; keyword-only retrieval will be used."}


def get_or_build_rag_state():
    """Load or build index/chunks; return (index, chunks_list, model, skip_query_embed, use_fastembed_emb). Cached in st.session_state.rag_state."""
    if getattr(st.session_state, "rag_state", None) is not None:
        return st.session_state.rag_state

    manifest_path = ROOT / "data" / "data_manifest.csv"
    index_path = ROOT / "data" / "processed" / "faiss.index"
    chunks_path = ROOT / "data" / "processed" / "chunks.json"

    if not chunks_path.exists():
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
            chunks_path.parent.mkdir(parents=True, exist_ok=True)
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

    skip_query_embed = os.environ.get("SKIP_QUERY_EMBED", "").strip() == "1"
    if index_path.exists():
        index, chunks_list, use_fastembed_emb = load_index(index_path, chunks_path)
        model = None
    else:
        chunks_list = load_chunks_only(chunks_path)
        index, use_fastembed_emb, model = None, False, None
        skip_query_embed = True

    state = (index, chunks_list, model, skip_query_embed, use_fastembed_emb)
    st.session_state.rag_state = state
    return state


def run_ask(query: str, top_k: int = 5, filter_source_ids: list[str] = None) -> dict | None:
    """Run retrieve → generate → citations → metrics → log_run → save_thread. Returns result dict or None on error."""
    try:
        manifest_path = ROOT / "data" / "data_manifest.csv"
        log_dir = ROOT / "logs"
        index, chunks_list, model, skip_query_embed, use_fastembed_emb = get_or_build_rag_state()

        # Retrieve more candidates if filtering is enabled
        retrieve_k = top_k * 3 if filter_source_ids else top_k
        
        retrieved = retrieve(
            query, index, chunks_list, model=model,
            top_k=retrieve_k,
            use_fastembed_embeddings=use_fastembed_emb,
            skip_query_embed=skip_query_embed,
        )
        
        # Apply source filtering after retrieval if specified
        if filter_source_ids:
            retrieved = [c for c in retrieved if c.get("source_id") in filter_source_ids]
            if not retrieved:
                st.warning(f"No chunks found matching selected filters. Using all retrieved sources.")
                # Re-retrieve without filter
                retrieved = retrieve(
                    query, index, chunks_list, model=model,
                    top_k=top_k,
                    use_fastembed_embeddings=use_fastembed_emb,
                    skip_query_embed=skip_query_embed,
                )
            else:
                # Trim to top_k after filtering
                retrieved = retrieved[:top_k]
        answer = generate(query, retrieved)
        answer = append_structured_citations(answer, retrieved, manifest_path)
        groundedness = compute_groundedness(answer, retrieved)
        answer_relevance = compute_answer_relevance(answer, query)
        model_id = f"ollama-{os.environ.get('OLLAMA_MODEL', 'gemma3:4b')}"

        log_run(
            query, [c.get("chunk_id", "") for c in retrieved], answer, log_dir,
            top_k=top_k, retrieved=retrieved, groundedness=groundedness, answer_relevance=answer_relevance,
            model_id=model_id,
        )
        thread_entry = save_thread(ROOT, query, retrieved, answer, groundedness=groundedness, answer_relevance=answer_relevance)
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
    for i, c in enumerate(retrieved, 1):
        sid = c.get("source_id", "?")
        cid = c.get("chunk_id", "?")
        score = c.get("score")
        text = (c.get("text") or "")[:400] + ("..." if len(c.get("text") or "") > 400 else "")
        score_str = f" (score: {score:.3f})" if score is not None else ""
        st.markdown(f"**[{i}] {sid} / {cid}**{score_str}")
        st.text(text)
        st.divider()


st.set_page_config(page_title="Personal Research Portal", layout="wide")

# ----- Sidebar: navigation + explanation -----
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown(
        "Use the options below to move through the portal. **Recommended workflow:** build the index first (or use **Build or rebuild index** on the Ask page), then ask questions."
    )
    page = st.radio(
        "Go to",
        ["Ask", "History", "Artifacts", "Build & run", "Evaluation"],
        index=0,
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**What each page does:**")
    st.markdown("- **Ask** — Run research questions against the corpus. Build the index first if this is your first time or after adding sources.")
    st.markdown("- **History** — View past queries, answers, and retrieved sources (research threads).")
    st.markdown("- **Artifacts** — Generate an evidence table from a thread and export as Markdown, CSV, or PDF.")
    st.markdown("- **Build & run** — Ingest PDFs, build/rebuild the search index, run the full evaluation suite, or run trust checks.")
    st.markdown("- **Evaluation** — See evaluation summary and per-answer scores (groundedness, citation correctness, usefulness).")

# ----- Main area: title + project description -----
st.title("Personal Research Portal")
st.markdown(
    "This portal lets you **ask research questions** over a curated corpus on **remote work and productivity**. "
    "It uses semantic search over ingested PDFs, then generates answers that cite specific sources and chunks. "
    "Every major claim is backed by an inline citation; when the corpus does not support something, the system says so and suggests a next step."
)
st.divider()

if page == "Ask":
    st.subheader("Workflow")
    st.info("**Step 1:** Build or rebuild the index so your questions run against the current corpus. **Step 2:** Ask your research question below.")
    if st.button("Build or rebuild index", type="secondary"):
        if "build_result" in st.session_state:
            del st.session_state["build_result"]
        if "rag_state" in st.session_state:
            del st.session_state["rag_state"]
        with st.spinner("Ingesting PDFs and building index..."):
            build_result = _do_build_index()
        st.session_state.build_result = build_result
    if getattr(st.session_state, "build_result", None) is not None:
        br = st.session_state.build_result
        if br.get("success"):
            st.success(br.get("message", "Index built."))
        else:
            st.warning(br.get("message", "Build failed."))
    
    st.subheader("Ask a research question")
    query = st.text_area("Question", placeholder="e.g. What is the average commute-time savings when working from home?", height=100)
    
    # Source filtering controls (user-friendly, inline)
    manifest_path = ROOT / "data" / "data_manifest.csv"
    filter_source_ids = None
    
    with st.expander("🔍 Filter sources (optional)", expanded=False):
        st.caption("Narrow your search to specific years, source types, or topics. Leave empty to search all sources.")
        
        if manifest_path.exists():
            try:
                df = pd.read_csv(manifest_path)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Year filter
                    years = sorted(df["year"].dropna().unique().tolist(), reverse=True)
                    selected_years = st.multiselect(
                        "📅 Year", 
                        years, 
                        default=[],
                        help="Select one or more years to include"
                    )
                
                with col2:
                    # Source type filter
                    types = sorted(df["source_type"].dropna().unique().tolist())
                    selected_types = st.multiselect(
                        "📄 Source Type", 
                        types, 
                        default=[],
                        help="Filter by publication type (peer-reviewed, working paper, etc.)"
                    )
                
                with col3:
                    # Tag filter (split and flatten tags column)
                    if "tags" in df.columns:
                        all_tags = set()
                        for tags_str in df["tags"].dropna():
                            all_tags.update([t.strip() for t in str(tags_str).split(",")])
                        all_tags = sorted([t for t in all_tags if t])
                        selected_tags = st.multiselect(
                            "🏷️ Topics", 
                            all_tags, 
                            default=[],
                            help="Filter by research topics or keywords"
                        )
                    else:
                        selected_tags = []
                
                # Apply filters and show feedback
                filtered_df = df
                filters_applied = []
                
                if selected_years:
                    filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]
                    filters_applied.append(f"{len(selected_years)} year(s)")
                
                if selected_types:
                    filtered_df = filtered_df[filtered_df["source_type"].isin(selected_types)]
                    filters_applied.append(f"{len(selected_types)} type(s)")
                
                if selected_tags:
                    filtered_df = filtered_df[filtered_df["tags"].apply(
                        lambda x: any(tag in str(x) for tag in selected_tags) if pd.notna(x) else False
                    )]
                    filters_applied.append(f"{len(selected_tags)} topic(s)")
                
                # Build filter result
                if filters_applied:
                    filter_source_ids = filtered_df["source_id"].tolist() if not filtered_df.empty else None
                    if filter_source_ids:
                        st.success(f"✓ **{len(filter_source_ids)} of {len(df)} sources** will be searched ({', '.join(filters_applied)} selected)")
                    else:
                        st.warning(f"⚠️ No sources match your filters. Will search all {len(df)} sources instead.")
                        filter_source_ids = None
                else:
                    st.info(f"ℹ️ No filters applied. Searching all **{len(df)} sources**.")
                    
            except Exception as e:
                st.error(f"Error loading filters: {e}")
        else:
            st.warning("Data manifest not found. Filters unavailable.")
    
    if st.button("Ask", type="primary"):
        if not (query or "").strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                result = run_ask(query.strip(), filter_source_ids=filter_source_ids)
            if result:
                st.session_state.last_result = result
                st.subheader("Answer")
                st.markdown(result["answer"])
                st.caption("Missing evidence is stated explicitly when the corpus does not support a claim; a suggested next retrieval step is included.")
                with st.expander("Sources (retrieved chunks)"):
                    render_sources(result["retrieved"])
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Groundedness", f"{result['groundedness']:.3f}")
                with col2:
                    st.metric("Answer relevance", f"{result['answer_relevance']:.3f}")
                st.success(f"Saved to thread {result.get('thread_id', '')} and logs.")

elif page == "History":
    st.header("Research threads")
    threads = load_threads(ROOT)
    if not threads:
        st.info("No threads yet. Use **Ask** to run a query.")
    else:
        def _label(tid):
            t = next((x for x in threads if x["thread_id"] == tid), None)
            return f"{tid} — {(t['query'][:50] + '...') if t and len(t.get('query', '')) > 50 else (t.get('query', '') or '')} ({(t.get('timestamp') or '')[:10]})" if t else tid
        selected_id = st.selectbox("Select a thread", options=[t["thread_id"] for t in threads], format_func=_label, index=0)
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
                manifest_path = ROOT / "data" / "data_manifest.csv"
                artifact = build_evidence_table(thread, manifest_path)
                if artifact:
                    st.session_state.artifact_table = artifact
                    st.subheader("Evidence table (preview)")
                    claim = artifact.get("claim", "")
                    if claim:
                        st.caption("Claim / query")
                        st.write(claim)
                    st.dataframe(artifact["rows"], width='stretch', hide_index=True)
                    st.success("Use the Export section below to download as Markdown, CSV, or PDF.")
                else:
                    st.warning("Could not build evidence table from this thread.")
        if getattr(st.session_state, "artifact_table", None) is not None:
            st.divider()
            st.subheader("Export")
            from src.app.artifacts import export_artifact_markdown, export_artifact_csv, export_artifact_pdf, export_bibtex
            table = st.session_state.artifact_table
            col1, col2, col3, col4 = st.columns(4)
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
            with col4:
                # Extract unique source_ids from artifact rows
                source_ids = []
                for row in table.get("rows", []):
                    citation = row.get("Citation", "")
                    # Extract source_id from citation format (SRC001, chunk_id)
                    import re
                    match = re.search(r'\((SRC\d+),', citation)
                    if match:
                        source_ids.append(match.group(1))
                source_ids = list(dict.fromkeys(source_ids))  # dedupe
                if source_ids:
                    manifest_path = ROOT / "data" / "data_manifest.csv"
                    bibtex = export_bibtex(source_ids, manifest_path)
                    if bibtex:
                        st.download_button("Download BibTeX", bibtex, file_name="references.bib", mime="text/plain")
                    else:
                        st.caption("BibTeX generation failed")
                else:
                    st.caption("No sources to export")

elif page == "Build & run":
    st.header("Build & run")
    st.caption("Build the index, run the full evaluation, or run trust behavior checks. All operations can be done from here instead of the CLI.")

    if st.button("Build or rebuild index"):
        if "build_result" in st.session_state:
            del st.session_state["build_result"]
        if "rag_state" in st.session_state:
            del st.session_state["rag_state"]
        with st.spinner("Ingesting PDFs and building index..."):
            build_result = _do_build_index()
        st.session_state.build_result = build_result
    if getattr(st.session_state, "build_result", None) is not None:
        br = st.session_state.build_result
        if br.get("success"):
            st.success(br.get("message", "Index built."))
        else:
            st.warning(br.get("message", "Build failed."))

    st.divider()
    if st.button("Run full evaluation (20 queries)"):
        try:
            from src.eval.run_eval import run_evaluation
            progress = st.progress(0.0)
            status = st.empty()
            def prog(i, total, qid):
                progress.progress(i / total)
                status.caption(f"Query {i}/{total} {qid}...")
            out_path = run_evaluation(no_build=True, progress_callback=prog)
            progress.progress(1.0)
            status.caption("")
            st.success(f"Evaluation complete. Results saved to **{out_path.name}**. Open the **Evaluation** page to see the summary.")
        except FileNotFoundError as e:
            st.warning(f"Index or chunks not found. Run **Build or rebuild index** first, then try again. Details: {e}")
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

    st.divider()
    if st.button("Run trust behavior test"):
        with st.spinner("Running trust behavior checks..."):
            from src.eval.trust_checks import run_trust_behavior_checks
            result = run_trust_behavior_checks()
        if result.get("skipped"):
            st.info(result.get("message", "Checks skipped."))
        elif result.get("citations_ok") and result.get("suggested_next_step_ok"):
            st.success(result.get("message", "All checks passed."))
        else:
            st.error(result.get("message", "Some checks failed."))
            st.caption(f"Citations OK: {result.get('citations_ok')}; Suggested next step OK: {result.get('suggested_next_step_ok')}.")

elif page == "Evaluation":
    st.header("Evaluation summary")
    eval_runs_dir = ROOT / "outputs" / "eval_runs"
    eval_runs_dir.mkdir(parents=True, exist_ok=True)
    eval_files = sorted(eval_runs_dir.glob("eval_run*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not eval_files:
        st.info("No evaluation runs found. Use **Build & run** to run the full evaluation (20 queries), or run `python -m src.eval.run_eval --no-build` from the CLI.")
    else:
        # Build options with metadata
        file_options = {}
        for p in eval_files:
            # Count entries in the file
            entry_count = 0
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry_count += 1
            # Get modification time
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            label = f"{p.name} — {entry_count} queries, {mtime}"
            file_options[label] = p
        
        file_choice = st.selectbox(
            "Select evaluation run",
            options=list(file_options.keys()),
            index=0,
            key="eval_file_choice",
        )
        latest = file_options[file_choice]
        st.caption(f"Run: **{latest.name}** ({latest.relative_to(ROOT)})")
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
            s1 = [e.get("score_groundedness_1_4") for e in entries if e.get("score_groundedness_1_4") is not None]
            s2 = [e.get("score_citation_correctness_1_4") for e in entries if e.get("score_citation_correctness_1_4") is not None]
            s3 = [e.get("score_usefulness_1_4") for e in entries if e.get("score_usefulness_1_4") is not None]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Queries", len(entries))
            with c2:
                st.metric("Mean groundedness", f"{sum(gr)/len(gr):.3f}" if gr else "—")
            with c3:
                st.metric("Mean answer relevance", f"{sum(rel)/len(rel):.3f}" if rel else "—")
            with c4:
                avg_s1 = sum(s1) / len(s1) if s1 else None
                st.metric("Avg score (1–4) groundedness", f"{avg_s1:.2f}" if avg_s1 is not None else "—")

            st.subheader("Per-answer evaluation rows")
            def _score1(e):
                if e.get("score_groundedness_1_4") is not None:
                    return e["score_groundedness_1_4"]
                g = e.get("groundedness")
                if g is not None:
                    try:
                        x = max(0, min(1, float(g)))
                        return 4 if x >= 0.75 else 3 if x >= 0.5 else 2 if x >= 0.25 else 1
                    except (TypeError, ValueError):
                        pass
                return "—"
            def _score2(e):
                if e.get("score_citation_correctness_1_4") is not None:
                    return e["score_citation_correctness_1_4"]
                try:
                    from src.eval.metrics import compute_citation_correctness
                    ret = e.get("retrieved") or []
                    return compute_citation_correctness(e.get("answer") or "", ret)
                except Exception:
                    return "—"
            def _score3(e):
                if e.get("score_usefulness_1_4") is not None:
                    return e["score_usefulness_1_4"]
                r = e.get("answer_relevance")
                if r is not None:
                    try:
                        x = max(0, min(1, float(r)))
                        return 4 if x >= 0.75 else 3 if x >= 0.5 else 2 if x >= 0.25 else 1
                    except (TypeError, ValueError):
                        pass
                return "—"
            rows = []
            for e in entries:
                prompt_model = f"{e.get('prompt_id', '')} + {e.get('model_id', '')}".strip(" +")
                ret_ids = e.get("retrieved_evidence_ids")
                if ret_ids is None and e.get("retrieved"):
                    ret_ids = [c.get("chunk_id", "") for c in e["retrieved"] if c.get("chunk_id")]
                ret_str = ", ".join(ret_ids) if ret_ids else "—"
                notes_tag = (e.get("notes", "") or "").strip()
                ft = (e.get("failure_tag", "") or "").strip()
                notes_display = "; ".join(filter(None, [notes_tag, ft])) or "—"
                rows.append({
                    "Task ID": e.get("task_id", "—"),
                    "Test case ID": e.get("test_case_id", e.get("query_id", "—")),
                    "Query ID": e.get("query_id", "—"),
                    "Prompt ID + model": prompt_model,
                    "Retrieved evidence IDs": ret_str[:80] + "…" if len(ret_str) > 80 else ret_str,
                    "Score 1 — Groundedness (1–4)": _score1(e),
                    "Score 2 — Citation correctness (1–4)": _score2(e),
                    "Score 3 — Usefulness (1–4)": _score3(e),
                    "Notes + failure tag": notes_display,
                })
            st.dataframe(rows, width='stretch', hide_index=True)

            st.subheader("Representative examples")
            by_type = {}
            for e in entries:
                t = e.get("query_type", "Direct")
                if t not in by_type:
                    by_type[t] = e
            for label, e in [
                ("Direct", by_type.get("Direct", entries[0])),
                ("Synthesis", by_type.get("Synthesis", entries[min(10, len(entries)-1)] if len(entries) > 10 else entries[-1])),
                ("Edge case", by_type.get("Edge Case", entries[-1] if entries else None)),
            ]:
                if e is None:
                    continue
                with st.expander(f"{label}: {e.get('query_id', '')} — {e.get('query', '')[:60]}..."):
                    st.text("Query: " + e.get("query", ""))
                    g, r = e.get("groundedness", 0), e.get("answer_relevance", 0)
                    st.caption(f"Groundedness: {g:.3f}  |  Relevance: {r:.3f}  |  Scores 1–4: G={e.get('score_groundedness_1_4', '—')} C={e.get('score_citation_correctness_1_4', '—')} U={e.get('score_usefulness_1_4', '—')}  |  Tag: {e.get('failure_tag', '') or '—'}")
                    ans = e.get("answer", "")
                    st.text_area("Answer excerpt", value=ans[:800] + "..." if len(ans) > 800 else ans, height=120, disabled=True, key=f"ans_{e.get('query_id', '')}")

if __name__ == "__main__":
    pass
