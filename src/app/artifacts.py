"""
Research artifact generator: Evidence table (Claim | Evidence snippet | Citation | Confidence | Notes).
Built from a thread (query + retrieved + answer).
"""
from __future__ import annotations

import re
import csv
import io
from typing import Optional

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


def build_evidence_table(thread: dict) -> list[dict] | None:
    """
    Build evidence table rows from a thread.
    thread must have: query, retrieved (list of {source_id, chunk_id, score, text}), answer.
    Returns list of dicts: claim, evidence_snippet, citation, confidence, notes.
    """
    retrieved = thread.get("retrieved") or []
    answer = thread.get("answer") or ""
    if not retrieved:
        return None

    rows = []
    # Pair each retrieved chunk with a snippet and confidence
    for c in retrieved:
        sid = c.get("source_id", "?")
        cid = c.get("chunk_id", "?")
        text = (c.get("text") or "").strip()
        score = c.get("score")
        citation = f"({sid}, {cid})"
        confidence = _confidence_label(score)
        # Extract a short "claim" from the answer that might reference this chunk (simple heuristic: first sentence mentioning source/chunk or first sentence)
        claim = ""
        if citation in answer or sid in answer or cid in answer:
            # Find first sentence that contains this citation or source
            sents = re.split(r"[.!?]\s+", answer)
            for sent in sents:
                if citation in sent or sid in sent or cid in sent:
                    claim = sent.strip() + "."
                    break
        if not claim:
            claim = thread.get("query", "")[:200] + ("..." if len(thread.get("query", "")) > 200 else "")
        evidence_snippet = text[:500] + ("..." if len(text) > 500 else "")
        rows.append({
            "Claim": claim,
            "Evidence snippet": evidence_snippet,
            "Citation": citation,
            "Confidence": confidence,
            "Notes": "",
        })
    return rows


def export_artifact_markdown(table: list[dict]) -> str:
    """Return Markdown string for the evidence table."""
    if not table:
        return ""
    lines = ["# Evidence table\n", "| Claim | Evidence snippet | Citation | Confidence | Notes |", "| --- | --- | --- | --- | --- |"]
    for row in table:
        # Escape pipe in cells
        def esc(s):
            return (s or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {esc(row.get('Claim', ''))} | {esc(row.get('Evidence snippet', ''))} | {row.get('Citation', '')} | {row.get('Confidence', '')} | {esc(row.get('Notes', ''))} |")
    return "\n".join(lines)


def export_artifact_csv(table: list[dict]) -> bytes:
    """Return CSV file as bytes."""
    if not table:
        return b""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["Claim", "Evidence snippet", "Citation", "Confidence", "Notes"])
    writer.writeheader()
    writer.writerows(table)
    return out.getvalue().encode("utf-8")


def export_artifact_pdf(table: list[dict]) -> Optional[bytes]:
    """Return PDF as bytes if fpdf2 is available; else None."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    if not table:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    col_w = [40, 55, 25, 22, 28]  # Claim, Evidence, Citation, Confidence, Notes
    headers = ["Claim", "Evidence snippet", "Citation", "Confidence", "Notes"]
    for j, h in enumerate(headers):
        pdf.cell(col_w[j], 7, h[:20], border=1)
    pdf.ln()
    for row in table:
        for j, key in enumerate(headers):
            val = (row.get(key) or "")[:50].replace("\n", " ")
            pdf.cell(col_w[j], 6, val, border=1)
        pdf.ln()
    # fpdf2: output() with no args returns bytearray
    out = pdf.output()
    return bytes(out) if out else None
