"""Suggest job-specific rewrites of existing CV lines (never invents experience)."""
from __future__ import annotations

from typing import Any

from .chains import tailor_chain
from .germany import detect_language
from .llm import Models
from .textutils import evidence_in_text


def tailor_cv(models: Models, cv_text: str, analysis: dict[str, Any], n: int = 6) -> list[dict[str, Any]]:
    must = [r["name"] for r in analysis["requirements"] if r["type"] == "must"]
    result = tailor_chain(models).invoke({
        "n": n,
        "cv_lang": "German" if detect_language(cv_text) == "de" else "English",
        "keywords": ", ".join(analysis["job"]["keywords"]),
        "must": "; ".join(must),
        "cv": cv_text[:12000],
    })
    out = []
    for s in result.suggestions[:n]:
        original, suggested = s.original.strip(), s.suggested.strip()
        if not suggested or suggested == original:
            continue
        out.append({
            "original": original,
            "suggested": suggested,
            "why": s.why.strip(),
            "needs_confirmation": s.needs_confirmation,
            "original_found_in_cv": evidence_in_text(original, cv_text),
        })
    return out
