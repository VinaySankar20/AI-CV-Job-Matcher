"""Hybrid CV ↔ job matching.

Final score (0-5) = weighted categories − penalty for missing must-haves, where
  skills     = 50% AI judgement + 30% requirement checklist + 20% semantic similarity
  experience = AI judgement
  education  = AI judgement
  languages  = AI judgement, capped by the rule-based German level check
  keywords   = share of ATS keywords from the ad that literally appear in the CV
AI "met" claims are cross-checked: if the quoted evidence isn't in the CV and neither
keywords nor embeddings support it, the requirement is downgraded to "partial".

This module is pure Python (no LLM calls): the LangGraph nodes in graph.py call the model
and hand the answers to the functions here.
"""
from __future__ import annotations

import re
from typing import Any

from . import config
from .llm import Models
from .textutils import chunk_cv, cosine, evidence_in_text, keyword_coverage, keyword_in_text, normalize

STATUS_VALUE = {"met": 1.0, "partial": 0.5, "missing": 0.0}


def _clamp(x: float, lo: float = 0.0, hi: float = 5.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return lo


_STOPWORDS = {"and", "or", "with", "of", "in", "the", "for", "a", "an", "to", "on", "und", "oder", "mit",
              "in", "der", "die", "das", "von", "im", "experience", "erfahrung", "knowledge", "kenntnisse",
              "skills", "skill"}


def _stem(word: str) -> str:
    for suffix in ("ments", "ment", "ings", "ing", "ers", "er", "es", "s", "ed"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2]:  # debugg -> debug
        word = word[:-1]
    return word


def _name_tokens(name: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[\w+#]+", name.lower()) if w not in _STOPWORDS}


def _similar_names(a: str, b: str) -> bool:
    """True for near-duplicates like 'Code Versioning with Git' / 'Source Code Versioning with Git'."""
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.6


_DEGREE_RX = {
    "bachelor": re.compile(r"\b(?:b\.?\s?tech|b\.?\s?e\.|b\.?\s?eng|b\.?\s?sc|bsc|bca|b\.?\s?a\.|bachelor\w*)\b", re.I),
    "master": re.compile(r"\b(?:m\.?\s?tech|m\.?\s?sc|msc|m\.?\s?eng|m\.?\s?a\.|mba|master\w*)\b", re.I),
}


def _line_with(rx: re.Pattern, cv_text: str) -> str:
    for line in cv_text.splitlines():
        if rx.search(line):
            return line.strip()[:200]
    return ""


_GENERIC_DEGREE_WORDS = {"bachelor", "bachelors", "bachelor's", "master", "masters", "master's", "degree", "abschluss",
                         "university", "universitat", "hochschulabschluss", "hochschul", "completed", "abgeschlossen",
                         "abgeschlossene", "relevant", "related", "field", "equivalent", "similar", "vergleichbar",
                         "or", "in", "a", "s", "'s", "of"}


def _degree_evidence(req_name: str, cv_text: str) -> tuple[str, bool]:
    """Rule-based degree check. Returns (CV line, generic) — generic means no specific field is required.

    'Bachelor's degree' is met by a B.Tech/B.E./B.Sc. — or by any Master's (which needs a Bachelor's first).
    """
    name = req_name.lower()
    words = set(re.findall(r"[\w']+", name))
    generic = words <= _GENERIC_DEGREE_WORDS
    if re.search(r"master|m\.?\s?sc", name):
        line = _line_with(_DEGREE_RX["master"], cv_text)
        in_progress = bool(re.search(r"in progress|present|heute|laufend|current|ongoing|since|seit", line, re.I))
        return line, generic and not in_progress
    if re.search(r"bachelor|b\.?\s?sc|undergraduate|degree|abschluss|studium", name) and not re.search(r"phd|doktor", name):
        return _line_with(_DEGREE_RX["bachelor"], cv_text) or _line_with(_DEGREE_RX["master"], cv_text), generic
    return "", False


def _keyword_line(req_name: str, cv_text: str) -> str:
    for line in cv_text.splitlines():
        if keyword_in_text(req_name, normalize(line)):
            return line.strip()[:200]
    return ""


def normalize_job(data: dict[str, Any]) -> dict[str, Any]:
    """Clean the model's job extraction: give ids, drop near-duplicates, cap list sizes."""
    reqs: list[dict[str, Any]] = []
    for kind, prefix, limit in (("must_have", "M", 12), ("nice_to_have", "N", 10)):
        for i, item in enumerate((data.get(kind) or [])[:limit], start=1):
            if isinstance(item, str):
                item = {"name": item, "category": "skill"}
            name = str(item.get("name", "")).strip()
            if name and not any(_similar_names(name, r["name"]) for r in reqs):
                reqs.append({"id": f"{prefix}{i}", "type": "must" if prefix == "M" else "nice",
                             "name": name, "category": item.get("category") or "skill"})
    return {
        "job_title": str(data.get("job_title") or "").strip(),
        "company": str(data.get("company") or "").strip(),
        "location": str(data.get("location") or "").strip(),
        "contact_person": str(data.get("contact_person") or "").strip(),
        "min_years_experience": _clamp(data.get("min_years_experience") or 0, 0, 30),
        "keywords": [str(k).strip() for k in (data.get("keywords") or []) if str(k).strip()][:25],
        "requirements": reqs,
    }


def requirement_lines(reqs: list[dict[str, Any]]) -> str:
    return "\n".join(f"{r['id']} [{r['type']}, {r['category']}]: {r['name']}" for r in reqs) or "(none found)"


def collect_verdicts(items: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """id -> verdict, keeping only valid statuses."""
    return {str(v.get("id")): v for v in (items or [])
            if isinstance(v, dict) and v.get("status") in STATUS_VALUE}


def skipped_requirements(reqs: list[dict[str, Any]], verdicts: dict[str, Any]) -> list[dict[str, Any]]:
    """Requirements the model gave no verdict for (small models often skip the nice-to-haves)."""
    return [r for r in reqs if r["id"] not in verdicts]


def semantic_similarities(models: Models, cv_text: str, reqs: list[dict]) -> list[float] | None:
    """Best cosine similarity between each requirement and any chunk of the CV (None if no embeddings)."""
    chunks = chunk_cv(cv_text)
    if not chunks or not reqs:
        return None
    req_vecs = models.embed([f"search_query: {r['name']}" for r in reqs])
    cv_vecs = models.embed([f"search_document: {c}" for c in chunks])
    if not req_vecs or not cv_vecs:
        return None
    return [max(cosine(rv, cv) for cv in cv_vecs) for rv in req_vecs]


def _sim_to_unit(sim: float) -> float:
    return max(0.0, min(1.0, (sim - config.SIM_LOW) / (config.SIM_HIGH - config.SIM_LOW)))


def build_analysis(job: dict[str, Any], verdicts: dict[str, dict[str, Any]], evaluation: dict[str, Any],
                   sims: list[float] | None, de: dict[str, Any], cv_text: str,
                   weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Combine the model's answers with the rule-based checks into the final scored analysis."""
    weights = weights or config.DEFAULT_WEIGHTS
    reqs = [dict(r) for r in job["requirements"]]
    ev = evaluation
    cv_norm = normalize(cv_text)

    # ---- merge AI verdicts with degree / keyword / embedding / evidence checks
    for i, r in enumerate(reqs):
        v = verdicts.get(r["id"], {})
        ai_status = v.get("status") if v.get("status") in STATUS_VALUE else None
        status = ai_status or "missing"
        evidence = str(v.get("evidence") or "").strip()
        r["keyword_found"] = keyword_in_text(r["name"], cv_norm)
        r["similarity"] = round(sims[i], 3) if sims else None
        r["evidence_verified"] = bool(evidence) and evidence_in_text(evidence, cv_text)
        r["note"] = ""
        sim = r["similarity"]

        degree_line, generic = _degree_evidence(r["name"], cv_text) if r["category"] == "education" else ("", False)
        if degree_line and status != "met" and (generic or status == "missing"):
            status = "met" if generic else "partial"
            evidence, r["evidence_verified"] = degree_line, True
            r["note"] = ("Degree found in your CV (rule-based check)." if generic
                         else "You have a degree — check that the field fits.")
        elif r["keyword_found"] and status != "met":
            short = len(_name_tokens(r["name"])) <= 2
            new = "met" if short else "partial" if status == "missing" else status
            if new != status:
                status = new
                evidence = evidence if r["evidence_verified"] else _keyword_line(r["name"], cv_text)
                r["evidence_verified"] = bool(evidence)
                r["note"] = "Keyword found in your CV — " + ("upgraded." if ai_status else "AI gave no verdict.")
        elif ai_status is None:
            if sim is not None and sim >= config.SIM_VERIFY_MIN:
                status = "partial"
            r["note"] = "AI gave no verdict — decided by the keyword/similarity check."
        elif status in ("met", "partial") and not r["evidence_verified"] and not r["keyword_found"]:
            if status == "met" and (sim is None or sim < config.SIM_VERIFY_MIN):
                status = "partial"
                r["note"] = "AI said 'met' but the quoted evidence isn't in your CV — downgraded."
            elif status == "partial" and sim is not None and sim < config.SIM_LOW:
                status = "missing"
                r["note"] = "No supporting text found in your CV — downgraded."
        r["status"] = status
        r["evidence"] = evidence

    # ---- requirement checklist score (must-haves count double)
    total_w = sum(2 if r["type"] == "must" else 1 for r in reqs)
    req_score = (5 * sum((2 if r["type"] == "must" else 1) * STATUS_VALUE[r["status"]] for r in reqs) / total_w
                 if total_w else None)
    sem_score = (5 * sum((2 if r["type"] == "must" else 1) * _sim_to_unit(r["similarity"]) for r in reqs) / total_w
                 if sims and total_w else None)

    ai = ev.get("scores") or {}

    def ai_score(name: str) -> tuple[float, str]:
        item = ai.get(name) or {}
        if isinstance(item, (int, float)):
            return _clamp(item), ""
        return _clamp(item.get("score", 0)), str(item.get("reason") or "")

    categories: dict[str, dict[str, Any]] = {}

    s_ai, s_reason = ai_score("skills")
    parts = {"AI judgement": (s_ai, config.SKILLS_BLEND["llm"])}
    if req_score is not None:
        parts["Requirement checklist"] = (req_score, config.SKILLS_BLEND["requirements"])
    if sem_score is not None:
        parts["Semantic similarity"] = (sem_score, config.SKILLS_BLEND["semantic"])
    w_sum = sum(w for _, w in parts.values())
    categories["skills"] = {"score": sum(s * w for s, w in parts.values()) / w_sum, "reason": s_reason,
                            "parts": {k: round(v[0], 2) for k, v in parts.items()}}

    for name in ("experience", "education"):
        sc, reason = ai_score(name)
        categories[name] = {"score": sc, "reason": reason, "parts": {"AI judgement": round(sc, 2)}}

    l_ai, l_reason = ai_score("languages")
    cap = de["language_cap"]
    l_final = min(l_ai, cap) if cap is not None else l_ai
    lang_parts = {"AI judgement": round(l_ai, 2)}
    if cap is not None:
        lang_parts["German-level cap"] = cap
        if l_final < l_ai:
            l_reason += f" (Capped at {cap} because the German requirement is above your level.)"
    categories["languages"] = {"score": l_final, "reason": l_reason.strip(), "parts": lang_parts}

    kw_cov, kw_found, kw_missing = keyword_coverage(job["keywords"], cv_text)
    categories["keywords"] = {
        "score": 5 * kw_cov if job["keywords"] else categories["skills"]["score"],
        "reason": (f"{len(kw_found)} of {len(kw_found) + len(kw_missing)} ATS keywords from the ad appear in your CV."
                   if job["keywords"] else "No keywords extracted — using the skills score."),
        "parts": {"Keyword coverage": round(5 * kw_cov, 2)},
        "found": kw_found, "missing": kw_missing,
    }

    for c in categories.values():
        c["score"] = round(_clamp(c["score"]), 2)

    weighted = sum(categories[k]["score"] * w for k, w in weights.items())
    missing_must = [r for r in reqs if r["type"] == "must" and r["status"] == "missing"]
    penalty = min(config.MUST_HAVE_PENALTY * len(missing_must), config.MUST_HAVE_PENALTY_MAX)
    overall = round(_clamp(weighted - penalty), 2)

    return {
        "job": {k: v for k, v in job.items() if k != "requirements"},
        "requirements": reqs,
        "categories": categories,
        "weights": weights,
        "weighted_score": round(weighted, 2),
        "penalty": {"missing_must_have": len(missing_must), "points": round(penalty, 2)},
        "overall": overall,
        "cover_letter_eligible": overall > config.COVER_LETTER_THRESHOLD,
        "semantic_available": sims is not None,
        "strengths": [str(s) for s in (ev.get("strengths") or [])][:5],
        "gaps": [str(s) for s in (ev.get("gaps") or [])][:5],
        "cv_tips": [str(s) for s in (ev.get("cv_tips") or [])][:5],
        "germany": de,
    }
