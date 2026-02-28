# Personal Research Portal — AI-Powered Research Document Synthesis

A complete Retrieval-Augmented Generation (RAG) system that answers research questions using a curated corpus of 15 academic sources on remote work and productivity. **Local-first architecture** — runs entirely on your machine with Ollama (no API keys, no rate limits). Includes a production web UI for searching, organizing research threads, exporting evidence tables, and running evaluations.

---

## 🚀 Run in 5 Minutes or Less

### Prerequisites (1 minute)

1. **Start Ollama** (the local language model server):
   ```bash
   ollama serve
   ```
   
2. **In a new terminal, pull the model** (if not already present):
   ```bash
   ollama pull gemma3:4b
   ```

### Run the System (3 minutes)

From the project root:

```bash
# 1. Install dependencies (if not already done)
python3 -m pip install -r requirements.txt

# 2. Launch the web UI. Run the command below then open http://localhost:8501
streamlit run src/app/streamlit_app.py

# 2. OR CLI: Run your first query (builds index automatically, ~30 seconds)
python run_rag.py --query "What is the impact of remote work on productivity?"

```

**That's it.** The output shows your answer with citations and evidence strength. Open http://localhost:8501 to use the full interface.

---

## 📚 What Is This System?

### Overview

The Personal Research Portal is an intelligent research assistant that synthesizes answers from domain-specific academic sources. It combines:
- **Semantic retrieval** (embeddings + vector search) to find relevant passages
- **Local language generation** (Ollama) to synthesis answers with citations
- **Structured output** with bibliographic references and confidence scoring
- **Web interface** for exploring research threads, saving findings, and exporting evidence

**Domain:** Remote work and productivity research  
**Corpus:** 15 carefully selected sources (peer-reviewed papers, working papers, government reports, industry surveys)  
**Use case:** Answer exploratory research questions like: "How does remote work affect productivity?" "What evidence exists for employee retention improvements?" "Do remote workers get promoted?"

### User Workflow

```
1. Ask a Question
   ↓
2. System retrieves relevant passages from 15 sources
   ↓
3. Local LLM synthesizes an answer with inline citations
   ↓
4. Result shows: Answer + Sources + Confidence scores
   ↓
5. (Optional) Save as research thread, generate evidence table, export to PDF/CSV/BibTeX
```

---

## ⚠️ Important: Ollama Must Be Running

This system requires **Ollama** to generate answers. It's a local LLM server—no credentials, quotas, or rate limits needed. Before running queries:

```bash
# Terminal 1: Start the Ollama server
ollama serve

# Terminal 2: Pull the model (one-time setup)
ollama pull gemma3:4b
```

---

## 🎯 Core Features

### 1. **Single-Query CLI**
```bash
python run_rag.py --query "Your research question"
```
- Automatically ingests PDFs and builds index on first run
- Returns answer with inline citations and evidence strength
- Logs query, answer, chunks, and metrics to `logs/runs.jsonl`

### 2. **Batch Evaluation (20 test queries)**
```bash
python -m src.eval.run_eval --no-build
```
- Runs 20 curated test queries across 3 difficulty tiers
- Computes automated metrics: groundedness, answer relevance
- Saves results to `outputs/eval_runs/eval_runN.jsonl`

### 3. **Web Interface (Streamlit)**
```bash
streamlit run src/app/streamlit_app.py
```

#### Ask Page
- Search bar with optional **source filtering** (by year, source type, topics)
- Inline citations: `(source_id, chunk_id)` format
- Evidence strength labels: High/Medium/Low confidence
- Results saved automatically as research threads

#### History Page
- View all saved research threads
- Click to expand and see full sources for any query

#### Artifacts Page
- Generate **Evidence Table**: Claim | Evidence snippets | Citations | Confidence | Notes
- **Export** as Markdown, CSV, PDF, or **BibTeX** (for citation managers like Zotero)
- Pre-filled with results from selected thread

#### Build & Run Page
- **Build Index**: Ingest PDFs → Chunk → Embed → FAISS index (no CLI needed)
- **Run Evaluation**: Execute full 20-query evaluation from UI
- **Run Trust Test**: Check citations + suggested next steps
- Progress indicators for each operation

#### Evaluation Page
- View summary of any evaluation run
- Mean metrics (groundedness, answer relevance)
- Representative example queries and results

### 4. **Structured Citations**
Every answer includes:
- **Inline citations**: `(SRC001, chunk_5)` — traceable to exact source and location
- **References section**: Formatted bibliography with Authors, Year, Title, DOI/URL
- Automatically pulled from `data/data_manifest.csv` for consistency

### 5. **Evidence Strength Scoring**
Automatic confidence labels:
- **High**: Retrieval similarity ≥ 0.45
- **Medium**: Retrieval similarity ≥ 0.30
- **Low**: Similarity < 0.30

Users can quickly see which evidence is strong vs. marginal.

### 6. **Automated Evaluation Metrics**
Computed for every query:
- **Groundedness** (0–1): Fraction of answer sentences supported by retrieved chunks
- **Answer Relevance** (0–1): Fraction of query keywords present in answer
- Enable reproducible quality tracking without manual scoring

### 7. **Research Threads & Export**
- **Threads**: Save query + answer + sources as a research thread (`threads/threads.jsonl`)
- **Export formats**: Markdown, CSV, PDF, BibTeX
- Integrate findings into citation managers or documents directly

### 8. **Retrieval Modes**
- **Semantic retrieval** (default): FastEmbed or sentence-transformers embeddings + FAISS
- **Keyword fallback**: Automatically used if index missing or embedding fails
- **Force keyword-only**: `SKIP_QUERY_EMBED=1 python run_rag.py ...`

### 9. **Trust & Transparency**
- Every answer explicitly cites sources
- Missing evidence stated clearly with suggested next retrieval step
- Generated citations can be verified against source PDFs

---

## 📖 Getting Started: Complete 5-Minute Walkthrough

### Step 1: Install & Setup (1 minute)
```bash
cd /path/to/Research\ Portal
pip install -r requirements.txt
```

### Step 2: Start Ollama (in separate terminal)
```bash
ollama serve
# In another terminal:
ollama pull gemma3:4b
```

### Step 3: CLI: Run Your First Query (30 seconds)
```bash
python run_rag.py --query "What is the average productivity impact of remote work?"
```

### Step 4: Try the Web UI (2 minutes)
```bash
streamlit run src/app/streamlit_app.py
```

---

## 📊 Project Structure

```
data/
  ├─ data_manifest.csv               # 15 academic sources
  ├─ raw/                            # Original PDFs
  └─ processed/
      ├─ faiss.index                 # Semantic search index
      ├─ chunks.json                 # 191 chunks with metadata
      └─ *.json                      # Parsed documents

src/
  ├─ ingest/parse_pdf.py             # PDF extraction
  ├─ rag/                  
  │  ├─ chunk.py                     # Chunking strategy
  │  ├─ embed_index.py               # FAISS indexing
  │  ├─ retrieve.py                  # Semantic + keyword retrieval
  │  ├─ generate.py                  # Ollama answer generation
  │  ├─ citations.py                 # Citation formatting
  │  └─ log_run.py                   # JSONL logging
  ├─ eval/
  │  ├─ metrics.py                   # Groundedness, relevance
  │  ├─ queries.jsonl                # 20 test queries
  │  ├─ run_eval.py                  # Batch evaluation
  │  └─ trust_checks.py              # Citation verification
  └─ app/
     ├─ streamlit_app.py             # Web UI
     ├─ threads.py                   # Save research threads
     └─ artifacts.py                 # Export to PDF/CSV/BibTeX

logs/runs.jsonl                       # Query logs with metrics
threads/threads.jsonl                 # Saved research sessions
outputs/eval_runs/                   # Evaluation batch results
```

---

## 🔧 Common Commands

| Task | Command |
|------|---------|
| Single query | `python run_rag.py --query "..."` |
| Repeated queries | `python run_rag.py --no-build --query "..."` |
| Latest log | `tail -1 logs/runs.jsonl | python -m json.tool` |
| Run 20 tests | `python -m src.eval.run_eval --no-build` |
| Web UI | `streamlit run src/app/streamlit_app.py` |
| Keyword-only | `SKIP_QUERY_EMBED=1 python run_rag.py --query "..."` |
| Custom LLM | `OLLAMA_MODEL=llama3.2:3b python run_rag.py ...` |

---

## 💡 How It Works

**Retrieval**: Convert question to embedding → search FAISS index → get top-5 similar chunks  
**Generation**: Pass chunks + question to Ollama LLM → generate answer with citations  
**Scoring**: Compute groundedness (text overlap) and answer relevance (keyword overlap)  
**Logging**: Save all queries, chunks, answers, and metrics to JSONL for analysis

---

## 📈 Evaluation Results

**Test Set**: 20 queries (direct factual, synthesis, edge cases)

- **Retrieval Precision**: 90% (18/20 queries have relevant info in top-5)
- **Groundedness**: 0.42 average (42% of answer sentences directly supported)
- **Answer Relevance**: 0.87 average (87% of query keywords in answer)
- **Citation Accuracy**: 19/20 answers use correct format

See [outputs/Phase2_Evaluation_Report.md](outputs/Phase2_Evaluation_Report.md) for full analysis.

---

## 🎛️ Configuration

### Environment Variables (`.env`)
```bash
OLLAMA_MODEL=gemma3:4b         # Default LLM model
SKIP_QUERY_EMBED=1             # Force keyword-only retrieval
TOP_K=5                         # Chunks to retrieve
CHUNK_SIZE=2048                 # Chunk size (chars)
CHUNK_OVERLAP=256              # Overlap between chunks
```

### Retrieval Backends (auto-selected)
1. **FastEmbed** (ONNX, lightweight) ← preferred
2. **sentence-transformers** (PyTorch)
3. **Keyword fallback** (no embeddings)

### Ollama Models
- `gemma3:4b` (default, 3GB)
- `llama3.2:3b` (smaller, 2GB)
- `mistral:latest` (alternative)

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Ollama not found" | Run `ollama serve` in separate terminal |
| "No chunks retrieved" | First run without `--no-build` to build index |
| "Index build failed" | Falls back to keyword-only automatically |
| "Segmentation fault (Apple Silicon)" | FastEmbed installed; sentence-transformers optional |
| "Query takes >30s" | Check `data/processed/faiss.index` exists |
| "Can't access http://localhost:8501" | Run `streamlit run src/app/streamlit_app.py` first |

---

## 🚀 Advanced Usage

### Add Your Own Sources
1. Place PDFs in `data/raw/`
2. Add entries to `data/data_manifest.csv`
3. Run: `python run_rag.py --query "..."`

### Export for Academic Writing
- Generate evidence tables on **Artifacts** page
- Export as BibTeX for Zotero/Mendeley
- Copy citations into your paper

### Evaluate Custom Queries
1. Add queries to `src/eval/queries.jsonl`
2. Run: `python -m src.eval.run_eval --no-build`
3. Compare metrics in `outputs/eval_runs/`

---

## 📊 System Overview

This project implements a **complete RAG pipeline** with:

- **Phase 2**: Evaluation metrics, structured citations, evidence scoring
- **Phase 3**: Web UI with search, threads, artifacts, and exports

**Key Technologies**:
- Ollama (local LLM, no API keys)
- FAISS (semantic search)
- Streamlit (web interface)
- pandas + pypdf (data processing)

**Evaluation**: 20 test queries across 3 difficulty levels with automated metrics for system quality assessment
