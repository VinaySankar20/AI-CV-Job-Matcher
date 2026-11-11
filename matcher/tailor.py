"""Suggest job-specific rewrites of existing CV lines (never invents experience)."""
from __future__ import annotations

from typing import Any

from . import prompts
from .germany import detect_language
from .llm import LLM
from .textutils import evidence_in_text


def tailor_cv(llm: LLM, cv_text: str, analysis: dict[str, Any], n: int = 6) -> list[dict[str, Any]]:
    must = [r["name"] for r in analysis["requirements"] if r["type"] == "must"]
    cv_lang = "German" if detect_language(cv_text) == "de" else "English"
    data = llm.chat_json(
        prompts.TAILOR_SYSTEM,
        prompts.TAILOR_USER.format(n=n, cv_lang=cv_lang, keywords=", ".join(analysis["job"]["keywords"]),
                                   must="; ".join(must), cv=cv_text[:12000]),
        schema=prompts.TAILOR_SCHEMA,
        temperature=0.3,
    )
    out = []
    for s in (data.get("suggestions") or [])[:n]:
        original = str(s.get("original") or "").strip()
        suggested = str(s.get("suggested") or "").strip()
        if not suggested or suggested == original:
            continue
        out.append({
            "original": original,
            "suggested": suggested,
            "why": str(s.get("why") or "").strip(),
            "needs_confirmation": bool(s.get("needs_confirmation")),
            "original_found_in_cv": evidence_in_text(original, cv_text),
        })
    return out
