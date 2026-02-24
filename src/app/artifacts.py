"""
Research artifact generator: Evidence table (claim shown once; rows: Evidence snippet | Citation | Confidence | Notes).
Built from a thread (query + retrieved + answer).
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# Evidence strength from retrieval score (align with citations.py)
STRENGTH_HIGH = 0.45
STRENGTH_MEDIUM = 0.30


def _confidence_label(score: float | None) -> str:
    if score is None:
        return "Low"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "Low"
    if s >= STRENGTH_HIGH:
        return "High"
    if s >= STRENGTH_MEDIUM:
        return "Medium"
    return "Low"


def _confidence_display(score: float | None) -> str:
    """Label plus numeric score when available, e.g. 'High (0.792)'."""
    label = _confidence_label(score)
    if score is not None:
        try:
            return f"{label} ({float(score):.3f})"
        except (TypeError, ValueError):
            pass
    return label


def _extract_limitations(answer: str) -> str:
    """Extract limitations/caveats/doubts from model answer for the Notes column."""
    if not answer or not answer.strip():
        return ""
    # Look for common limitation/caveat phrases (case-insensitive)
    lower = answer.lower()
    cues = [
        "suggested next step",
        "suggest a specific next",
        "evidence not found",
        "not specified in the provided",
        "do not support",
        "does not support",
        "corpus does not support",
        "limitation",
        "caveat",
        "uncertain",
        "conflicting evidence",
        "conflict between",
        "cannot infer",
        "do not infer",
    ]
    sentences = re.split(r"[.!?]\s+", answer)
    found = []
    for sent in sentences:
        s = sent.strip()
        if not s or len(s) < 20:
            continue
        s_lower = s.lower()
        if any(c in s_lower for c in cues):
            found.append(s[:400] + ("..." if len(s) > 400 else ""))
    if not found:
        return ""
    return " ".join(found)[:500].strip()


def build_evidence_table(thread: dict, manifest_path: Path | str | None = None) -> dict | None:
    """
    Build evidence table from a thread.
    thread must have: query, retrieved (list of {source_id, chunk_id, score, text}), answer.
    manifest_path: optional path to data_manifest.csv for relevance notes.
    Returns {"claim": str, "rows": list[dict]} with one claim for the whole table and rows
    containing only Evidence snippet, Citation, Confidence, Notes (no repeated claim per row).
    """
    retrieved = thread.get("retrieved") or []
    query = (thread.get("query") or "").strip()
    answer = (thread.get("answer") or "").strip()
    if not retrieved:
        return None

    # Single claim for the whole table: use the user's query (the question being answered)
    claim = query[:500] + ("..." if len(query) > 500 else "")
    
    # Load relevance notes from manifest if available
    relevance_map = {}
    if manifest_path:
        manifest_path = Path(manifest_path)
        if manifest_path.exists():
            try:
                df = pd.read_csv(manifest_path)
                if "source_id" in df.columns and "relevance_note" in df.columns:
                    relevance_map = dict(zip(df["source_id"], df["relevance_note"]))
            except Exception:
                pass

    rows = []
    for c in retrieved:
        sid = c.get("source_id", "?")
        cid = c.get("chunk_id", "?")
        text = (c.get("text") or "").strip()
        score = c.get("score")
        citation = f"({sid}, {cid})"
        confidence = _confidence_display(score)
        evidence_snippet = text[:500] + ("..." if len(text) > 500 else "")
        
        # Use relevance note from manifest if available, otherwise empty
        notes = relevance_map.get(sid, "")
        
        rows.append({
            "Evidence snippet": evidence_snippet,
            "Citation": citation,
            "Confidence": confidence,
            "Notes": notes,
        })
    return {"claim": claim, "rows": rows}


def _artifact_claim_and_rows(artifact: dict | list) -> tuple[str, list[dict]]:
    """Normalize artifact to (claim, rows). Supports {"claim", "rows"} or legacy list of rows with Claim column."""
    if isinstance(artifact, dict) and "rows" in artifact:
        return (artifact.get("claim") or "", artifact["rows"])
    if isinstance(artifact, list) and artifact and isinstance(artifact[0], dict):
        # Legacy: list of rows with Claim column
        claim = (artifact[0].get("Claim") or "") if artifact else ""
        rows = [{"Evidence snippet": r.get("Evidence snippet", ""), "Citation": r.get("Citation", ""), "Confidence": r.get("Confidence", ""), "Notes": r.get("Notes", "")} for r in artifact]
        return (claim, rows)
    return ("", [])


def export_artifact_markdown(artifact: dict | list) -> str:
    """Return Markdown string for the evidence table. Claim is shown once above the table."""
    claim, rows = _artifact_claim_and_rows(artifact)
    if not rows:
        return ""
    def esc(s):
        return (s or "").replace("|", "\\|").replace("\n", " ")
    lines = ["# Evidence table\n"]
    if claim:
        lines.append("## Claim / query\n\n" + esc(claim) + "\n")
    lines.append("| Evidence snippet | Citation | Confidence | Notes |")
    lines.append("| --- | --- | --- | --- |")
    for row in rows:
        lines.append(f"| {esc(row.get('Evidence snippet', ''))} | {row.get('Citation', '')} | {row.get('Confidence', '')} | {esc(row.get('Notes', ''))} |")
    return "\n".join(lines)


def export_artifact_csv(artifact: dict | list) -> bytes:
    """Return CSV file as bytes. First line can be a comment with the claim; table has no Claim column."""
    claim, rows = _artifact_claim_and_rows(artifact)
    if not rows:
        return b""
    out = io.StringIO()
    if claim:
        out.write("# Claim / query: " + claim.replace("\n", " ").replace("\r", "") + "\n")
    writer = csv.DictWriter(out, fieldnames=["Evidence snippet", "Citation", "Confidence", "Notes"])
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def _pdf_safe(s: str, max_len: int = 50) -> str:
    """Strip/replace characters that Helvetica cannot render to avoid FPDFUnicodeEncodingException."""
    if not s:
        return ""
    s = (s[:max_len] + ("..." if len(s) > max_len else "")).replace("\n", " ")
    # Keep only printable ASCII; replace others with space
    return "".join(c if 32 <= ord(c) <= 126 else " " for c in s)


def export_artifact_pdf(artifact: dict | list) -> Optional[bytes]:
    """Return PDF as bytes if fpdf2 is available; else None. Claim shown once; table has no Claim column."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    claim, rows = _artifact_claim_and_rows(artifact)
    if not rows:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    if claim:
        pdf.set_font("Helvetica", size=8)
        pdf.cell(0, 5, _pdf_safe(claim, max_len=120), ln=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", size=10)
    col_w = [55, 25, 22, 28]  # Evidence snippet, Citation, Confidence, Notes
    headers = ["Evidence snippet", "Citation", "Confidence", "Notes"]
    for j, h in enumerate(headers):
        pdf.cell(col_w[j], 7, h[:20], border=1)
    pdf.ln()
    for row in rows:
        for j, key in enumerate(headers):
            val = _pdf_safe(row.get(key) or "", max_len=50)
            pdf.cell(col_w[j], 6, val, border=1)
        pdf.ln()
    out = pdf.output()
    return bytes(out) if out else None


def export_bibtex(source_ids: list[str], manifest_path: Path | str) -> str:
    """
    Generate BibTeX entries for the given source_ids from the data manifest.
    Returns a string with BibTeX entries ready for citation managers.
    """
    if not source_ids:
        return ""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return ""
    try:
        df = pd.read_csv(manifest_path)
    except Exception:
        return ""
    if "source_id" not in df.columns:
        return ""
    
    unique_ids = list(dict.fromkeys(source_ids))  # preserve order, dedupe
    rows = df[df["source_id"].isin(unique_ids)]
    
    entries = []
    for row in rows.itertuples(index=False):
        sid = getattr(row, "source_id", "UNKNOWN")
        authors = getattr(row, "authors", "") or "Unknown"
        title = getattr(row, "title", "") or "Untitled"
        year = getattr(row, "year", "") or "n.d."
        source_type = getattr(row, "source_type", "") or "misc"
        venue = getattr(row, "venue", "") or ""
        url = getattr(row, "url_or_doi", "") or ""
        
        # Map source_type to BibTeX entry type
        entry_type = "article"
        if "working paper" in source_type.lower():
            entry_type = "techreport"
        elif "report" in source_type.lower():
            entry_type = "techreport"
        elif "book" in source_type.lower():
            entry_type = "book"
        elif "conference" in source_type.lower() or "workshop" in source_type.lower():
            entry_type = "inproceedings"
        
        # Build BibTeX entry
        bibtex = f"@{entry_type}{{{sid},\n"
        bibtex += f"  author = {{{authors}}},\n"
        bibtex += f"  title = {{{title}}},\n"
        bibtex += f"  year = {{{year}}}"
        if venue:
            if entry_type == "article":
                bibtex += f",\n  journal = {{{venue}}}"
            elif entry_type == "inproceedings":
                bibtex += f",\n  booktitle = {{{venue}}}"
            elif entry_type == "techreport":
                bibtex += f",\n  institution = {{{venue}}}"
        if url:
            bibtex += f",\n  url = {{{url}}}"
        bibtex += "\n}"
        entries.append(bibtex)
    
    return "\n\n".join(entries)
