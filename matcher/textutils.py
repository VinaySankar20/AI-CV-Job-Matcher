"""Keyword matching, CV chunking and cosine similarity (no AI needed)."""
from __future__ import annotations

import math
import re

# common short forms -> canonical form (both directions are checked)
ALIASES = {
    "js": "javascript", "ts": "typescript", "py": "python", "ml": "machine learning",
    "dl": "deep learning", "ai": "artificial intelligence", "nlp": "natural language processing",
    "cv": "computer vision", "k8s": "kubernetes", "postgres": "postgresql", "gcp": "google cloud",
    "aws": "amazon web services", "llm": "large language model", "llms": "large language models",
    "ci/cd": "continuous integration", "sql server": "mssql", "ms excel": "excel",
    "power bi": "powerbi", "scikit-learn": "sklearn", "react.js": "react", "reactjs": "react",
    "node.js": "node", "nodejs": "node", "vue.js": "vue", "c sharp": "c#",
    "maschinelles lernen": "machine learning", "künstliche intelligenz": "artificial intelligence",
}


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w+#./' -]", " ", text)
    text = re.sub(r"(?<!\w)[./'-]|[./'-](?!\w)", " ", text)  # strip punctuation at word edges
    return " " + re.sub(r"\s+", " ", text).strip() + " "


def _variants(keyword: str) -> set[str]:
    k = normalize(keyword).strip()
    out = {k}
    if k in ALIASES:
        out.add(ALIASES[k])
    out.update(short for short, full in ALIASES.items() if full == k)
    for v in list(out):
        if len(v) > 3 and v.endswith("s"):
            out.add(v[:-1])
        out.add(v.replace("-", " "))
        out.add(v.replace(" ", "-"))
        out.add(v.replace(" ", ""))
    return {v for v in out if v}


def keyword_in_text(keyword: str, normalized_text: str) -> bool:
    """Whole-word match of a keyword (or a known variant) in already-normalized text.

    "C or C++" / "Power BI oder Tableau" match if ANY alternative is found, and words
    joined in the CV ("Agile-Scrum", "C/C++") are also checked as separate words.
    """
    alternatives = [a for a in re.split(r"\s+(?:or|oder|and/or|bzw\.?)\s+|\s*/\s*(?=[a-z])", keyword.strip(), flags=re.I) if a]
    if len(alternatives) > 1:
        return any(keyword_in_text(a, normalized_text) for a in alternatives)
    split_text = re.sub(r"(?<=\w)[-/](?=\w)", " ", normalized_text)
    return any(f" {v} " in t or f" {v}s " in t for v in _variants(keyword) for t in (normalized_text, split_text))


def keyword_coverage(keywords: list[str], cv_text: str) -> tuple[float, list[str], list[str]]:
    norm = normalize(cv_text)
    seen: dict[str, None] = {}
    for k in keywords:
        if k and k.strip() and normalize(k).strip() not in {normalize(s).strip() for s in seen}:
            seen[k.strip()] = None
    unique = list(seen)
    if not unique:
        return 0.0, [], []
    found = [k for k in unique if keyword_in_text(k, norm)]
    missing = [k for k in unique if k not in found]
    return len(found) / len(unique), found, missing


def evidence_in_text(evidence: str, cv_text: str) -> bool:
    """True if the AI's quoted evidence really appears in the CV (allowing small differences)."""
    ev_words = [w for w in normalize(evidence).split() if len(w) > 2]
    if not ev_words:
        return False
    norm = normalize(cv_text)
    if normalize(evidence) in norm:
        return True
    hits = sum(1 for w in ev_words if f" {w} " in norm)
    return hits / len(ev_words) >= 0.7


def chunk_cv(cv_text: str, max_chars: int = 350) -> list[str]:
    """Group CV lines into small chunks so each requirement can be matched to its best chunk."""
    lines = [ln.strip(" •-–*\t") for ln in cv_text.splitlines()]
    lines = [ln for ln in lines if len(ln) > 2]
    chunks: list[str] = []
    current = ""
    for ln in lines:
        if current and len(current) + len(ln) + 1 > max_chars:
            chunks.append(current)
            current = ln
        else:
            current = f"{current} {ln}".strip()
    if current:
        chunks.append(current)
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
