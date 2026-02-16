# Personal Research Portal — Phase 2 RAG System

A Retrieval-Augmented Generation (RAG) system for answering research questions about remote work and productivity using a curated corpus of 15 academic sources.

**Domain:** Remote work and productivity  
**Main Question:** How does remote work impact productivity at individual, organizational, and industry levels?

---

## ⚠️ PREREQUISITE: Ollama Must Be Running

This project uses **Ollama** as the default local language model for answer generation. No API quotas, no rate limits, no credentials needed. Before running any queries, ensure Ollama is running in the background:

```bash
# Start Ollama (runs in background)
ollama serve
```

Then, in a separate terminal, pull the model once (if not already present):

```bash
ollama pull gemma3:4b
```

Ollama provides local LLM inference without requiring API credentials or hitting quota limits.

---

## Quick Start for Graders (2 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify PDFs Are Present
The project expects PDFs in `data/raw/` matching entries in `data/data_manifest.csv`. The PDFs should already be in place for this submission.

### 3. Run a Single Query (build index, then verify)

**Step 3a: Build the chunk index (first run only, ~10 seconds)**
```bash
python run_rag.py --query "What is the average daily commute-time savings when working from home?"
```

This ingests the 15 PDFs and saves chunks to `data/processed/chunks.json`. The system then attempts to build a semantic index:
- If embeddings are available (Gemini API, OpenAI, or `USE_LOCAL_EMBEDDINGS=fastembed`), it builds a FAISS index for semantic retrieval
- If index building fails or APIs are unavailable, it gracefully falls back to fast keyword-only retrieval
- Either way, the workflow completes and Ollama generates an answer

Both retrieval modes work well for this corpus. Keyword-only is more reliable and avoids API quota issues.

**Step 3b: Run with `--no-build` for subsequent queries (instant)**
```bash
python run_rag.py --no-build --query "Does remote work increase employee retention rates?"
```

The `--no-build` flag skips re-ingesting PDFs, making repeated queries fast.

**Output shows three sections:**
- **[1] RETRIEVAL RESULTS**: Top-5 retrieved chunks with source IDs, chunk IDs, and similarity scores
- **[2] ANSWER WITH CITATIONS**: LLM-generated answer with inline `(source_id, chunk_id)` citations, References section, and Evidence Strength summary (High/Medium/Low confidence based on retrieval scores)
- **[3] LOG ENTRY SAVED**: Entry appended to `logs/runs.jsonl` including query, retrieved chunks, answer, groundedness metric, and answer_relevance metric

### 4. Run the Full Evaluation Set (20 queries)

```bash
python -m src.eval.run_eval --no-build
```

This executes all 20 test queries (9 direct, 5 synthesis, 6 edge cases) and saves results to `outputs/eval_runN.jsonl` (auto-incremented filename). Each entry includes query, retrieved chunks, generated answer, and automated metrics.

---

## Project Structure

```
data/
  data_manifest.csv        # Source metadata (title, authors, year, URL/DOI)
  raw/                     # PDF files (one per source)
  processed/               # Generated during run: parsed JSONs, FAISS index, chunks
    faiss.index            # FAISS vector index for retrieval
    chunks.json            # Serialized chunks with metadata
    *.json                 # Ingested PDFs (parsed text, metadata)

src/
  ingest/
    parse_pdf.py           # PDF parsing and text extraction
  rag/
    chunk.py               # Semantic chunking (2048 chars, 256 overlap)
    embed_index.py         # Embedding and FAISS indexing
    retrieve.py            # Semantic/keyword retrieval
    generate.py            # Answer generation via Ollama LLM
    citations.py           # Structured citations and evidence strength scoring
    log_run.py             # JSONL logging of runs
  eval/
    metrics.py             # Automated evaluation: groundedness, answer_relevance
    queries.jsonl          # 20 test queries with expected behaviors
    run_eval.py            # Batch evaluation runner

logs/
  runs.jsonl               # Single-query run logs (from run_rag.py)

outputs/
  eval_runN.jsonl          # Evaluation results (N auto-incremented)
  Phase2_Evaluation_Report.md  # Comprehensive evaluation report

run_rag.py                 # Main pipeline entry point (single query)
requirements.txt           # Python dependencies
```

---

## Key Files Explained

### `data/data_manifest.csv`
Index of all 15 sources with columns:
- `source_id`: Unique identifier (SRC001–SRC015)
- `authors`, `year`, `title`: Bibliographic information
- `url_or_doi`: Link or DOI for the source
- `raw_path`, `processed_path`: File paths

Used by the system to locate PDFs, build formatted References sections, and tag retrieved chunks.

### `src/eval/queries.jsonl`
20 manually curated test queries across three difficulty tiers:
- **Direct Factual (Q01–Q09)**: Extract specific stats/findings from single sources
- **Synthesis/Multi-hop (Q11–Q15)**: Integrate evidence across multiple sources
- **Edge Cases (Q16–Q20)**: Test uncertainty handling and absence of evidence

Each query includes an `expected_behavior` field describing what a high-quality answer should do.

---

## Phase 2 Enhancements

### 1. **Structured Citations**
Inline citations `(source_id, chunk_id)` paired with a formatted References section from `data_manifest.csv` for full bibliographic attribution.

### 2. **Evidence Strength Scoring**
Automatic High/Medium/Low confidence labels (≥0.45/≥0.30/<0.30 retrieval scores) shown in answers, helping users gauge evidence reliability.

### 3. **Automated Evaluation Metrics**
- **Groundedness**: Fraction of answer sentences with direct textual overlap in retrieved chunks
- **Answer Relevance**: Fraction of query keywords present in answer
- Logged for all queries with reproducible computations.

---

## How to Verify the Workflow

### After running a query:
```bash
# View the most recent log entry with all 3 components
tail -1 logs/runs.jsonl | python -m json.tool
```

You'll see:
- `retrieved`: Top-5 chunks with source_id, chunk_id, score
- `answer`: Full answer with citations and evidence strength
- `groundedness`: Computed metric (0-1 scale)
- `answer_relevance`: Computed metric (0-1 scale)

### After running the evaluation set:
```bash
# View results from the latest eval run
tail -1 outputs/eval_runN.jsonl | python -m json.tool
```

Each query includes retrieved chunks, answer, and both metrics.

---

## System Components Summary

| File | Purpose |
|------|---------|
| `parse_pdf.py` | Extract and clean text from PDFs |
| `chunk.py` | Split text into 2048-char chunks with 256-char overlap |
| `embed_index.py` | Build FAISS vector index (optional; keyword retrieval used by default) |
| `retrieve.py` | Retrieve top-k chunks via keyword matching |
| `generate.py` | Call local Ollama LLM to generate answers |
| `citations.py` | Format citations, references, and evidence strength labels |
| `log_run.py` | Log query, answer, and metrics to JSONL |
| `metrics.py` | Compute groundedness and answer_relevance |

---

## Advanced Options (Optional)

Use semantic embeddings instead of keyword-only retrieval:
```bash
USE_LOCAL_EMBEDDINGS=fastembed python run_rag.py --no-build --query "..."
```

Use OpenAI instead of Ollama for answer generation:
```bash
USE_OPENAI=1 python run_rag.py --no-build --query "..."
```

Use Gemini instead of Ollama (set GEMINI_API_KEY first):
```bash
GEMINI_API_KEY=your-key python run_rag.py --no-build --query "..."
```

---

## Troubleshooting

**"⚠️ Index build failed"**: The system tried to build a semantic index with embeddings (Gemini, OpenAI, or local) but failed. It automatically fell back to keyword-only retrieval, which is fast and reliable. The workflow will still complete successfully. To skip embedding attempts and go straight to keyword-only retrieval, unset any embedding-related environment variables.

**"Segmentation fault: 11"**: This was a known issue with sentence-transformers on Apple Silicon. The system now defaults to keyword-only retrieval (safe) and only loads embedding models if explicitly requested.

**"Ollama not found"**: Ensure `ollama serve` is running in a separate terminal.

**"No chunks retrieved"**: Run the first query without `--no-build` to ingest PDFs and create `data/processed/chunks.json`. Subsequent queries can use `--no-build` for speed.

---

## Deliverables

✅ Complete end-to-end RAG pipeline (single command = retrieval + answer + log)  
✅ 20-query evaluation set with baseline results  
✅ Phase 2 evaluation report analyzing system performance and metrics  
✅ Automated quality metrics (groundedness, answer_relevance) for all queries
