"""
Structured citations: reference list from manifest and evidence-strength scoring.
Used after generation to append a References section and Evidence strength summary.
"""
from pathlib import Path

import pandas as pd

# Evidence strength buckets (retrieval relevance score).
# Semantic similarity typically in [0, 1]; keyword overlap can be < 1.
STRENGTH_HIGH = 0.45
STRENGTH_MEDIUM = 0.30


def format_reference_list(source_ids: list[str], manifest_path: Path) -> str:
    """
    Build a reference list from the data manifest for the given source_ids.
    Returns a string like "## References\\n\\n1. Authors (Year). Title. URL."
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
    # Manifest may use 'source_id' or 'source_id' column
    if "source_id" not in df.columns:
        return ""
    unique_ids = list(dict.fromkeys(source_ids))  # preserve order, dedupe
    rows = df[df["source_id"].isin(unique_ids)].copy()
    # Reorder to match unique_ids
    order = {sid: i for i, sid in enumerate(unique_ids)}
    rows["_order"] = rows["source_id"].map(order)
    rows = rows.sort_values("_order")
    lines = []
    for i, row in enumerate(rows.itertuples(index=False), start=1):
        authors = getattr(row, "authors", "") or ""
        year = getattr(row, "year", "") or ""
        title = getattr(row, "title", "") or ""
        url = getattr(row, "url_or_doi", "") or ""
        part = f"{authors} ({year}). {title}. {url}".strip()
        lines.append(f"{i}. {part}")
    if not lines:
        return ""
    return "## References\n\n" + "\n".join(lines)


def _strength_label(score: float) -> str:
    """Map retrieval score to High / Medium / Low."""
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


def format_evidence_strength(retrieved: list[dict]) -> str:
    """
    Format evidence-strength summary from retrieved chunks (using retrieval score as proxy).
    Returns a string like "## Evidence strength\\n\\n- High: SRC001_chunk_0 (0.52)\\n..."
    """
    if not retrieved:
        return ""
    by_strength = {"High": [], "Medium": [], "Low": []}
    for c in retrieved:
        cid = c.get("chunk_id") or "?"
        score = c.get("score")
        label = _strength_label(score)
        score_str = f"{score:.3f}" if score is not None else "—"
        by_strength[label].append((cid, score_str))
    parts = []
    for level in ("High", "Medium", "Low"):
        items = by_strength[level]
        if not items:
            continue
        entries = ", ".join(f"{cid} ({s})" for cid, s in items)
        parts.append(f"- **{level}**: {entries}")
    if not parts:
        return ""
    return "## Evidence strength\n\n" + "\n\n".join(parts)


def append_structured_citations(
    answer: str,
    retrieved: list[dict],
    manifest_path: Path,
) -> str:
    """
    Append reference list (from manifest) and evidence-strength section to the answer.
    """
    source_ids = [c.get("source_id") for c in retrieved if c.get("source_id")]
    ref_block = format_reference_list(source_ids, manifest_path)
    strength_block = format_evidence_strength(retrieved)
    out = answer.rstrip()
    if ref_block:
        out += "\n\n" + ref_block
    if strength_block:
        out += "\n\n" + strength_block
    return out
