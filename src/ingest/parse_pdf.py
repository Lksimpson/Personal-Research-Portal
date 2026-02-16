"""
Parse PDFs listed in data_manifest.csv and write extracted text + metadata to data/processed/.
Strategy: one JSON per source with keys: source_id, title, authors, year, text, raw_path.
"""
from pathlib import Path
import json
import re
import pandas as pd
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Normalize whitespace and remove common PDF artifacts."""
    if not text or not text.strip():
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return text.strip()


def extract_text_from_pdf(path: Path) -> str:
    """Extract raw text from a PDF file."""
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join(parts)


def ingest_one(row: pd.Series, base_dir: Path) -> dict | None:
    """
    Ingest one source: read PDF at raw_path, write JSON to processed_path.
    Returns record dict for logging, or None if file missing/failed.
    """
    raw_path = base_dir / row["raw_path"]
    processed_path = base_dir / row["processed_path"]

    if not raw_path.exists():
        return None

    processed_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        raw_text = extract_text_from_pdf(raw_path)
        text = clean_text(raw_text)
    except Exception as e:
        return {"source_id": row["source_id"], "error": str(e), "path": str(raw_path)}

    doc = {
        "source_id": row["source_id"],
        "title": row["title"],
        "authors": row["authors"],
        "year": int(row["year"]) if pd.notna(row["year"]) else None,
        "source_type": row["source_type"],
        "venue": row["venue"],
        "url_or_doi": row["url_or_doi"],
        "raw_path": row["raw_path"],
        "text": text,
    }

    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    return {"source_id": row["source_id"], "path": str(processed_path), "chars": len(text)}


def run_ingest(manifest_path: str | Path, base_dir: str | Path) -> list:
    """
    Run ingestion for all rows in the manifest.
    base_dir: project root (paths in manifest are relative to this).
    Returns list of per-source results for logging.
    """
    base_dir = Path(base_dir)
    df = pd.read_csv(manifest_path)
    results = []
    for _, row in df.iterrows():
        out = ingest_one(row, base_dir)
        if out:
            results.append(out)
    return results
