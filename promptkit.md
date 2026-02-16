# Prompt Kit

---

## Paper Triage Task

### Prompt A (Baseline)
Summarize the following paper about remote work. Tell me the contribution, method, data used, findings, and limitations.

---

### Prompt B (Structured with Guardrails)
You are performing **PAPER TRIAGE** for a research corpus on remote work and productivity.

**INPUT:**
- Paper text or excerpt (with `chunk_ids`).

**TASK:**
Produce a structured summary with **EXACTLY** the following five fields:

1. **Contribution** – What new knowledge this paper adds.  
2. **Method** – Study design and analytical approach.  
3. **Data** – Data source(s), sample size, and time period.  
4. **Findings** – Empirical results supported by the data.  
5. **Limitations** – Explicit weaknesses stated by the authors OR implied by the method/data.

**CONSTRAINTS:**
- Use only information present in the provided text.
- Cite at least one `chunk_id` per field.
- If a field cannot be answered from the text, write:  
  **"Not specified in the provided text."**
- Do **NOT** infer causality unless the paper explicitly claims causal identification.
- Keep each field to **2–4 sentences**.

**OUTPUT FORMAT:**
- Five bullet points labeled **exactly** as the field names above.

**Why these constraints exist:**
- **Exact fields:** enables cross-paper comparison  
- **chunk_id citations:** forces grounding, prevents hallucination  
- **“Not specified” rule:** exposes corpus gaps instead of guessing  
- **Causality guardrail:** critical in remote-work research  
- **Length limits:** discourages vague or padded summaries  

---

## Cross-Source Synthesis Task

### Prompt A (Baseline)
Compare the following sources on remote work and productivity.  
Summarize where they agree and disagree.

---

### Prompt B (Structured with Guardrails)
You are performing **CROSS-SOURCE SYNTHESIS** for a research corpus on remote work and productivity.

**INPUT:**
- Multiple sources, each with `source_id` and `chunk_ids`.

**TASK:**
Produce a comparison table with the following columns:

1. **Agreement**  
2. **Disagreement**  
3. **What evidence supports each side**

**CONSTRAINTS:**
- Each row must reference **at least TWO different sources**.
- Every claim must be supported with citations in the form:  
  `(source_id, chunk_id)`
- Clearly distinguish differences caused by:
  - a) Level of analysis (individual, firm, macro)
  - b) Measurement type (self-reported vs objective)
  - c) Study design (causal vs correlational)
- Do **NOT** resolve disagreements unless the sources explicitly reconcile them.
- If evidence is insufficient to support a conclusion, state that explicitly.

**OUTPUT FORMAT:**
- **Markdown table only** (no prose before or after).

**Why these constraints exist:**
- **Two-source minimum:** prevents single-paper “synthesis”  
- **Evidence column:** makes disagreement explainable, not rhetorical  
- **Measurement/level distinctions:** core to this domain  
- **No forced resolution:** avoids artificial consensus  
- **Table-only output:** machine-checkable structure for Phase 2  