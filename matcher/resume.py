"""Job-specific resume built from your complete ("master") CV — selected and rephrased, never invented.

Flow: the model picks the relevant parts of the complete CV (resume_chain), `clean_resume` enforces the
size limits, `resume_fact_check` compares every claim with the complete CV, and `generate_resume`
rewrites once with the problems listed if the check fails, keeping the draft with the fewest issues.
"""
from __future__ import annotations

import re
from typing import Any

from . import config
from .cover_letter import not_in_cv_terms
from .germany import CEFR
from .llm import Models
from .textutils import evidence_in_text, keyword_in_text, normalize

SIZES: dict[str, dict[str, Any]] = {
    "one_page": {"label": "1 page", "max_entries": 3, "max_projects": 2, "max_bullets": 3, "summary_words": 50},
    "two_pages": {"label": "2 pages", "max_entries": 5, "max_projects": 4, "max_bullets": 4, "summary_words": 70},
}

HEADINGS = {
    "de": {"summary": "Profil", "experience": "Berufserfahrung", "projects": "Projekte", "education": "Ausbildung",
           "skills": "Kenntnisse", "languages": "Sprachen", "certifications": "Zertifikate"},
    "en": {"summary": "Profile", "experience": "Experience", "projects": "Projects", "education": "Education",
           "skills": "Skills", "languages": "Languages", "certifications": "Certifications"},
}
SECTION_ORDER = ("summary", "experience", "projects", "education", "skills", "languages", "certifications")


def resume_inputs(cv_text: str, jd_text: str, analysis: dict[str, Any], size: str = "one_page",
                  extra_instructions: str = "") -> dict[str, Any]:
    """All placeholder values for prompts.RESUME_USER."""
    spec = SIZES.get(size, SIZES["one_page"])
    lang = analysis["germany"]["job_language"]
    job = analysis["job"]
    extra = extra_instructions.strip()
    return {
        "language": "German" if lang == "de" else "English",
        "size": spec["label"],
        **{k: spec[k] for k in ("max_entries", "max_projects", "max_bullets", "summary_words")},
        "do_not_claim": ", ".join(not_in_cv_terms(analysis, cv_text)) or "(none)",
        "title": job.get("job_title") or "the role",
        "company": job.get("company") or "the company",
        "must": "; ".join(r["name"] for r in analysis.get("requirements", []) if r["type"] == "must") or "(none)",
        "keywords": ", ".join(job.get("keywords", [])) or "(none)",
        "extra": f"Extra instructions from the candidate (follow them): {extra}" if extra else "",
        "jd": jd_text[:10000],
        "cv": cv_text[:24000],
    }


def _s(value: Any) -> str:
    return str(value or "").strip()


def _entry(item: dict[str, Any], max_bullets: int) -> dict[str, Any] | None:
    bullets = [_s(b).lstrip("•-–* ").strip() for b in (item.get("bullets") or [])]
    entry = {"title": _s(item.get("title")), "organisation": _s(item.get("organisation")),
             "location": _s(item.get("location")), "dates": _s(item.get("dates")),
             "bullets": [b for b in bullets if b][:max_bullets]}
    return entry if entry["title"] or entry["bullets"] else None


def clean_resume(data: dict[str, Any], language: str, size: str = "one_page") -> dict[str, Any]:
    """Enforce the size limits and tidy the model's answer."""
    spec = SIZES.get(size, SIZES["one_page"])
    mb = spec["max_bullets"]

    def entries(key: str, limit: int, bullets: int) -> list[dict[str, Any]]:
        out = [e for e in (_entry(x, bullets) for x in (data.get(key) or [])) if e]
        return out[:limit]

    summary = _s(data.get("summary"))
    words = summary.split()
    if len(words) > spec["summary_words"] * 1.3:  # hard stop for run-away summaries
        summary = " ".join(words[: spec["summary_words"]]).rstrip(",;") + " …"
    skills = []
    for group in (data.get("skills") or [])[:5]:
        items = list(dict.fromkeys(_s(i) for i in (group.get("items") or []) if _s(i)))
        if items:
            skills.append({"category": _s(group.get("category")), "items": items[:12]})
    return {
        "language": language,
        "size": size,
        "headline": _s(data.get("headline")),
        "summary": summary,
        "experience": entries("experience", spec["max_entries"], mb),
        "projects": entries("projects", spec["max_projects"], mb),
        "education": entries("education", 4, 2),
        "skills": skills,
        "languages": [_s(x) for x in (data.get("languages") or []) if _s(x)][:6],
        "certifications": [_s(x) for x in (data.get("certifications") or []) if _s(x)][:6],
    }


# --------------------------------------------------------------------------- fact check
_NUMBER = re.compile(r"(?<![\w.,/])(\d+(?:[.,]\d+)?)(?![\w])")
_DATE_TOKEN = re.compile(r"\b(?:\d{1,2}[./])?(?:19|20)\d{2}\b")


def _all_text(resume: dict[str, Any]) -> list[str]:
    parts = [resume.get("headline", ""), resume.get("summary", "")]
    for key in ("experience", "projects", "education"):
        for e in resume.get(key, []):
            parts += [e["title"], *e["bullets"]]
    return [p for p in parts if p]


def resume_fact_check(resume: dict[str, Any], cv_text: str, analysis: dict[str, Any]) -> list[str]:
    """Warnings about anything in the resume that the complete CV does not support."""
    cv_norm = normalize(cv_text)
    texts = _all_text(resume)
    body_norm = normalize(" ".join(texts + [i for g in resume.get("skills", []) for i in g["items"]]))
    warnings: list[str] = []

    for term in sorted(not_in_cv_terms(analysis, cv_text), key=len):
        if keyword_in_text(term, body_norm):
            warnings.append(f"Mentions **{term}**, which is not in your complete CV.")

    for group in resume.get("skills", []):
        for item in group["items"]:
            main = re.sub(r"\(.*?\)", "", item).strip()
            if main and not keyword_in_text(main, cv_norm) and not evidence_in_text(main, cv_text):
                warnings.append(f"Skill **{item}** is not in your complete CV.")

    for key in ("experience", "projects", "education"):
        for e in resume.get(key, []):
            org = e["organisation"]
            if org and not keyword_in_text(org, cv_norm) and not evidence_in_text(org, cv_text):
                warnings.append(f"Organisation **{org}** is not in your complete CV.")
            for d in _DATE_TOKEN.findall(e["dates"]):
                if not re.search(rf"\b{re.escape(d)}\b", cv_text):
                    warnings.append(f"Date **{d}** ({e['title'] or org}) is not in your complete CV.")

    for text in texts:
        for m in _NUMBER.finditer(text):
            n = m.group(1)
            if not re.search(rf"(?<![\d.,]){re.escape(n)}(?![\d])", cv_text):
                warnings.append(f"Number **{n}** does not appear in your complete CV.")

    # language levels: compare with the level written in the same part of the CV line as the language
    segments = re.split(r"[,;|·•\n]", cv_text)
    for line in resume.get("languages", []):
        name = line.split()[0].strip(":,()-–") if line.split() else ""
        if not name:
            continue
        cv_levels = {lvl.upper() for seg in segments if re.search(rf"\b{re.escape(name)}", seg, re.I)
                     for lvl in re.findall(r"\b([ABC][12])\b", seg, re.I)}
        for level in re.findall(r"\b([ABC][12])\b", line.upper()):
            if level in CEFR and level not in cv_levels:
                warnings.append(f"Language level **{line}** does not match your complete CV.")

    return list(dict.fromkeys(warnings))


# --------------------------------------------------------------------------- generate
def generate_resume(models: Models, cv_text: str, jd_text: str, analysis: dict[str, Any],
                    size: str = "one_page", extra_instructions: str = "",
                    max_rewrites: int | None = None) -> dict[str, Any]:
    """Tailored resume + fact check, rewritten automatically while unsupported claims remain."""
    from .chains import resume_chain

    chain = resume_chain(models)
    lang = analysis["germany"]["job_language"]
    limit = config.MAX_RESUME_REWRITES if max_rewrites is None else max_rewrites
    extra = extra_instructions
    best: tuple[dict[str, Any], list[str]] | None = None
    rewrites = 0
    while True:
        data = chain.invoke(resume_inputs(cv_text, jd_text, analysis, size, extra)).model_dump()
        resume = clean_resume(data, lang, size)
        issues = resume_fact_check(resume, cv_text, analysis)
        if best is None or len(issues) < len(best[1]):
            best = (resume, issues)
        if not issues or rewrites >= limit:
            break
        rewrites += 1
        extra = (extra_instructions + " Fix these problems from the previous draft (remove the unsupported "
                 "parts): " + " ".join(i.replace("**", "") for i in issues)).strip()
    return {"resume": best[0], "issues": best[1], "rewrites": rewrites}


def resume_to_text(resume: dict[str, Any], profile: dict[str, Any] | None = None) -> str:
    """Plain-text version (preview and TXT download)."""
    h = HEADINGS.get(resume.get("language", "en"), HEADINGS["en"])
    profile = profile or {}
    contact = " · ".join(_s(profile.get(k)) for k in ("postal_city", "phone", "email") if _s(profile.get(k)))
    lines = [x for x in (_s(profile.get("full_name")), resume.get("headline", ""), contact) if x]
    for key in SECTION_ORDER:
        value = resume.get(key)
        if not value:
            continue
        lines += ["", h[key].upper()]
        if key == "summary":
            lines.append(value)
        elif key == "skills":
            lines += [f"{g['category']}: {', '.join(g['items'])}" if g["category"] else ", ".join(g["items"])
                      for g in value]
        elif key in ("languages", "certifications"):
            lines += [f"- {x}" for x in value]
        else:
            for e in value:
                head = " — ".join(x for x in (e["title"], e["organisation"]) if x)
                meta = " · ".join(x for x in (e["dates"], e["location"]) if x)
                lines.append(head + (f"  ({meta})" if meta else ""))
                lines += [f"  • {b}" for b in e["bullets"]]
    return "\n".join(lines).strip()
