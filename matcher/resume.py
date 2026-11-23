"""Job-specific resume built from your complete ("master") CV — selected and rephrased, never invented.

Flow:
  1. the model picks the relevant parts of the complete CV (resume_chain)
  2. `clean_resume` enforces the size limits and merges roles the model split up (one employer = one role,
     its projects as subsections)
  3. `restore_details` puts back the original CV line wherever a bullet lost specifics (boards, tools,
     numbers) or picked up a number from another line; the skills come straight from the CV's skills section
  4. `resume_fact_check` compares every claim with the complete CV; `generate_resume` rebuilds once with
     the problems listed if the check fails, keeping the draft with the fewest issues.
"""
from __future__ import annotations

import re
from typing import Any

from . import config
from .cover_letter import _NUMBER_WORDS, not_in_cv_terms
from .germany import CEFR, detect_language
from .llm import Models
from .textutils import evidence_in_text, keyword_in_text, normalize

SIZES: dict[str, dict[str, Any]] = {
    "one_page": {"label": "1 page", "max_entries": 3, "max_subsections": 4, "sub_bullets": 2, "max_bullets": 3,
                 "max_projects": 3, "project_bullets": 2, "summary_words": 50},
    "two_pages": {"label": "2 pages", "max_entries": 5, "max_subsections": 6, "sub_bullets": 3, "max_bullets": 4,
                  "max_projects": 5, "project_bullets": 3, "summary_words": 70},
}
_SIZE_KEYS = ("max_entries", "max_subsections", "sub_bullets", "max_bullets", "max_projects", "project_bullets",
              "summary_words")

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
        **{k: spec[k] for k in _SIZE_KEYS},
        "do_not_claim": ", ".join(not_in_cv_terms(analysis, cv_text)) or "(none)",
        "title": job.get("job_title") or "the role",
        "company": job.get("company") or "the company",
        "must": "; ".join(r["name"] for r in analysis.get("requirements", []) if r["type"] == "must") or "(none)",
        "keywords": ", ".join(job.get("keywords", [])) or "(none)",
        "extra": f"Extra instructions from the candidate (follow them): {extra}" if extra else "",
        "jd": jd_text[:10000],
        "cv": cv_text[:24000],
    }


# --------------------------------------------------------------------------- clean-up
def _s(value: Any) -> str:
    return str(value or "").strip()


def _bullets(items: Any, limit: int) -> list[str]:
    out = [_s(b).lstrip("•-–*▪● ").strip() for b in (items or [])]
    return [b for b in out if b][:limit]


_YEAR_TOKEN = re.compile(r"(?:\d{1,2}[./])?(?:19|20)\d{2}")


def _split_dates(text: str, dates: str) -> tuple[str, str]:
    """'Personal Project, 2026 • Link' + '2026' -> ('Personal Project', '2026')."""
    text = re.sub(r"[\s,·•|–-]*\bLink\b\s*$", "", text, flags=re.I)
    for token in _YEAR_TOKEN.findall(text):
        if not dates:
            dates = token
        if token in dates:
            text = re.sub(rf"[\s,·•|(–-]*\b{re.escape(token)}\b\)?", "", text)
    return text.strip(" ,;·•|–-"), dates


def _entry(item: dict[str, Any], max_bullets: int, max_subsections: int = 0,
           sub_bullets: int = 0) -> dict[str, Any] | None:
    subs = []
    for sub in (item.get("subsections") or [])[:max_subsections]:
        sb = {"name": _s(sub.get("name")), "dates": _s(sub.get("dates")),
              "bullets": _bullets(sub.get("bullets"), sub_bullets)}
        if sb["name"] or sb["bullets"]:
            subs.append(sb)
    dates = _s(item.get("dates"))
    title, dates = _split_dates(_s(item.get("title")), dates)
    organisation, dates = _split_dates(_s(item.get("organisation")), dates)
    location, dates = _split_dates(_s(item.get("location")), dates)    # models sometimes put the year here
    entry = {"title": title, "organisation": organisation,
             "location": location, "dates": dates, "tech": _s(item.get("tech")),
             "bullets": _bullets(item.get("bullets"), max_bullets), "subsections": subs}
    return entry if entry["title"] or entry["bullets"] or subs else None


def _merge_split_roles(entries: list[dict[str, Any]], max_subsections: int) -> list[dict[str, Any]]:
    """Small models often turn the projects under ONE role into separate jobs at the same employer.

    Later entries with the same organisation become subsections of the first one."""
    merged: list[dict[str, Any]] = []
    by_org: dict[str, dict[str, Any]] = {}
    for e in entries:
        key = normalize(e["organisation"]).strip()
        first = by_org.get(key) if key else None
        if first is None:
            merged.append(e)
            if key:
                by_org[key] = e
            continue
        first["subsections"].append({"name": e["title"], "dates": e["dates"], "bullets": e["bullets"]})
        first["subsections"].extend(e["subsections"])
        first["subsections"] = first["subsections"][:max_subsections]
    return merged


def clean_resume(data: dict[str, Any], language: str, size: str = "one_page") -> dict[str, Any]:
    """Enforce the size limits, merge split roles and tidy the model's answer."""
    spec = SIZES.get(size, SIZES["one_page"])

    def entries(key: str, limit: int, bullets: int, subs: int = 0, sub_bullets: int = 0) -> list[dict[str, Any]]:
        out = [e for e in (_entry(x, bullets, subs, sub_bullets) for x in (data.get(key) or [])) if e]
        return out[:limit]

    experience = entries("experience", spec["max_entries"] + 3, spec["max_bullets"], spec["max_subsections"],
                         spec["sub_bullets"])
    experience = _merge_split_roles(experience, spec["max_subsections"])[: spec["max_entries"]]

    summary = _s(data.get("summary"))
    words = summary.split()
    if len(words) > spec["summary_words"] * 1.3:  # hard stop for run-away summaries
        summary = " ".join(words[: spec["summary_words"]]).rstrip(",;") + " …"
    skills = []
    for group in (data.get("skills") or [])[:8]:
        items = list(dict.fromkeys(_s(i) for i in (group.get("items") or []) if _s(i)))
        if items:
            skills.append({"category": _s(group.get("category")), "items": items[:15]})
    return {
        "language": language,
        "size": size,
        "headline": _s(data.get("headline")),
        "summary": summary,
        "experience": experience,
        "projects": entries("projects", spec["max_projects"], spec["project_bullets"]),
        "education": entries("education", 4, 2),
        "skills": skills,
        "languages": [_s(x) for x in (data.get("languages") or []) if _s(x)][:6],
        "certifications": [_s(x) for x in (data.get("certifications") or []) if _s(x)][:8],
    }


# --------------------------------------------------------------------------- skills from the CV
_SKILL_HEADER = re.compile(r"^(?:(?:technical|core|key|it)[ -])?(?:skills?|kenntnisse|fähigkeiten|"
                           r"skills?\s*(?:&|and)\s*tools|tech(?:nical)? stack|it-kenntnisse)\s*:?$", re.I)
_SPOKEN = re.compile(r"\b(?:english|german|deutsch|englisch|french|spanish|hindi|malayalam|tamil)\b", re.I)


def _is_heading(line: str) -> bool:
    letters = re.sub(r"[^A-Za-zÄÖÜäöüß]", "", line)
    return bool(letters) and letters.isupper() and len(line) <= 40


def parse_cv_skills(cv_text: str) -> list[dict[str, Any]]:
    """The skills section of the CV as groups: 'Embedded : JTAG, SWD, STM32' -> {category, items}."""
    lines = [ln.strip() for ln in cv_text.splitlines()]
    try:
        start = next(i for i, ln in enumerate(lines) if _SKILL_HEADER.match(ln))
    except StopIteration:
        return []
    groups: list[dict[str, Any]] = []
    for ln in lines[start + 1:]:
        if not ln:
            continue
        if _is_heading(ln):
            break
        category, _, rest = ln.partition(":") if ":" in ln else ("", "", ln)
        items = [i.strip(" •·-–") for i in re.split(r"\s*[,;|·•]\s*", rest) if i.strip(" •·-–")]
        if not items:
            continue
        category = category.strip(" •-–")
        if category.lower() in ("languages", "sprachen") and not any(_SPOKEN.search(i) for i in items):
            category = "Programming Languages"   # 'Languages : C, C++' are programming languages
        if groups and not category and not groups[-1]["category"]:
            groups[-1]["items"] += items       # wrapped line of an uncategorised list
        else:
            groups.append({"category": category, "items": items})
    return groups


def skills_for_job(cv_text: str, jd_text: str, model_skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Skills straight from the CV's skills section, job-relevant groups and items first.

    Without a recognisable skills section the model's groups are used, minus anything not in the CV."""
    jd_norm = normalize(jd_text)
    groups = parse_cv_skills(cv_text)
    if not groups:
        cv_norm = normalize(cv_text)
        groups = [{"category": g["category"],
                   "items": [i for i in g["items"] if keyword_in_text(re.sub(r"\(.*?\)", "", i).strip(), cv_norm)
                             or evidence_in_text(i, cv_text)]} for g in model_skills]
        groups = [g for g in groups if g["items"]]

    def relevant(item: str) -> bool:
        main = re.sub(r"\(.*?\)", "", item).strip()
        parts = [p.strip() for p in re.split(r"[-/&]|\band\b|\bund\b", main) if len(p.strip()) >= 3]
        return keyword_in_text(main, jd_norm) or any(keyword_in_text(p, jd_norm) for p in parts)

    ranked = []
    for pos, g in enumerate(groups):
        items = sorted(g["items"], key=lambda i: not relevant(i))   # stable: relevant first
        hits = sum(relevant(i) for i in items) + (2 if relevant(g["category"]) else 0)
        ranked.append((-hits, pos, {"category": g["category"], "items": items}))
    return [g for _, _, g in sorted(ranked, key=lambda t: t[:2])]


# --------------------------------------------------------------------------- keep the CV's details
_BULLET_START = re.compile(r"^[•\-–*▪●]\s*")
_HEADING_END = re.compile(r"(?:(?:19|20)\d{2}|present|heute|today|link)\s*$", re.I)


def cv_blocks(cv_text: str) -> list[tuple[str, str]]:
    """(bullet, heading) for every bullet point of the CV: wrapped lines joined back together, and the
    last non-bullet line above it (e.g. 'Edge AI deployment on Renesas MCUs 2025') as its heading."""
    out: list[tuple[str, str]] = []
    cur: str | None = None
    heading = ""
    for raw in cv_text.splitlines():
        line = raw.strip()
        if _BULLET_START.match(line):
            if cur:
                out.append((cur, heading))
            cur = _BULLET_START.sub("", line)
        elif cur is not None and line and not re.search(r"[.!?]$", cur) and not _HEADING_END.search(line) \
                and not _is_heading(line):
            cur += " " + line
        else:
            if cur:
                out.append((cur, heading))
            cur = None
            if line and (_HEADING_END.search(line) or _is_heading(line)):
                heading = line
    if cur:
        out.append((cur, heading))
    return [(re.sub(r"\s+", " ", c).strip(), h) for c, h in out if len(c) > 15]


def cv_lines(cv_text: str) -> list[str]:
    """The CV's bullet points with wrapped lines joined back together."""
    return [c for c, _ in cv_blocks(cv_text)]


def _words(text: str) -> set[str]:
    return {w[:6] for w in normalize(text).split() if len(w) > 3}


def _details(text: str) -> set[str]:
    """Specific tokens: numbers, model names (RA8D1, STM32), acronyms (JTAG), proper nouns mid-sentence."""
    tokens = re.findall(r"[\w+#/.-]*\w[\w+#]*", text)
    out = set()
    for i, t in enumerate(tokens):
        if any(c.isdigit() for c in t) or (len(t) >= 2 and t.isupper()) or (i > 0 and t[0].isupper()) \
                or re.match(r"^[a-z]+[A-Z]", t):
            out.add(t.lower())
    return out


def _best_line(bullet: str, lines: list[str]) -> tuple[int, float]:
    bw = _words(bullet)
    if not bw:
        return -1, 0.0
    scores = [len(bw & _words(ln)) / len(bw) for ln in lines]
    best = max(range(len(lines)), key=scores.__getitem__) if lines else -1
    return best, scores[best] if lines else 0.0


def _restore(bullets: list[str], lines: list[str], used: set[int]) -> list[str]:
    out = []
    for b in bullets:
        i, score = _best_line(b, lines)
        if i < 0 or score < 0.5:
            out.append(b)          # not clearly based on one CV line — the fact check looks at it
            continue
        if i in used:
            continue               # a second bullet made from the same CV line: drop the duplicate
        used.add(i)
        original = lines[i]
        lost = _details(original) - _details(b) - {w.lower() for w in re.findall(r"[\w+#/.-]+", b)}
        moved = bool(re.findall(r"\d+(?:[.,]\d+)?\s?%", b)) and not all(
            n in original for n in re.findall(r"\d+(?:[.,]\d+)?\s?%", b))
        out.append(original if (lost or moved) else b)
    return out


def _same_name(a: str, b: str) -> bool:
    wa = _words(a)
    return bool(wa) and len(wa & _words(b)) / len(wa) >= 0.6


def _regroup(entry: dict[str, Any], blocks: list[tuple[str, str]]) -> None:
    """Move role-level bullets into the subsection whose CV heading they come from; drop empty subsections."""
    if not entry.get("subsections"):
        return
    lines = [c for c, _ in blocks]
    keep = []
    for b in entry["bullets"]:
        i, score = _best_line(b, lines)
        target = None
        if i >= 0 and score >= 0.5:
            target = next((s for s in entry["subsections"] if _same_name(s["name"], blocks[i][1])), None)
        (target["bullets"] if target else keep).append(b)
    entry["bullets"] = keep
    entry["subsections"] = [s for s in entry["subsections"] if s["bullets"]]


def restore_details(resume: dict[str, Any], cv_text: str) -> dict[str, Any]:
    """Put the CV's own line back wherever the model generalised details away or moved a number,
    and file each bullet under the right project heading.

    Only when the resume is in the CV's language (a translated bullet can't be compared word by word)."""
    lang = detect_language(cv_text)
    blocks = cv_blocks(cv_text)
    if resume.get("language") != lang or not blocks:
        for e in resume.get("experience", []):
            e["subsections"] = [s for s in e.get("subsections", []) if s["bullets"]]
        return resume
    lines = [c for c, _ in blocks]
    used: set[int] = set()
    for key in ("experience", "projects"):
        for e in resume.get(key, []):
            e["bullets"] = _restore(e["bullets"], lines, used)
            for sub in e.get("subsections", []):
                sub["bullets"] = _restore(sub["bullets"], lines, used)
    for e in resume.get("experience", []):
        _regroup(e, blocks)
    return resume


_EDU_HEADER = re.compile(r"^(?:education|ausbildung|studium|bildungsweg|academic background)\b", re.I)


def _education_section(cv_text: str) -> str:
    lines = cv_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if _EDU_HEADER.match(ln.strip())), None)
    if start is None:
        return ""
    out = []
    for ln in lines[start + 1:]:
        if _is_heading(ln.strip()):
            break
        out.append(ln)
    return "\n".join(out)


def keep_backed_education(resume: dict[str, Any], cv_text: str) -> dict[str, Any]:
    """Education bullets (thesis, courses, grades) must come from the CV's education section."""
    section = _education_section(cv_text)
    for e in resume.get("education", []):
        e["bullets"] = [b for b in e["bullets"] if section and evidence_in_text(b, section)]
    return resume


# --------------------------------------------------------------------------- fact check
_NUMBER = re.compile(r"(?<![\w.,/])(\d+(?:[.,]\d+)?)(?![\w])")
_DATE_TOKEN = re.compile(r"\b(?:\d{1,2}[./])?(?:19|20)\d{2}\b")


def _all_text(resume: dict[str, Any]) -> list[str]:
    parts = [resume.get("headline", ""), resume.get("summary", "")]
    for key in ("experience", "projects", "education"):
        for e in resume.get(key, []):
            parts += [e["title"], *e["bullets"]]
            for sub in e.get("subsections", []):
                parts += [sub["name"], *sub["bullets"]]
    return [p for p in parts if p]


def _number_backed(n: str, cv_text: str) -> bool:
    if re.search(rf"(?<![\d.,]){re.escape(n)}(?![\d])", cv_text):
        return True
    words = [w for w, d in _NUMBER_WORDS.items() if d == n]     # "4" is backed by "four" / "vier"
    return any(re.search(rf"\b{w}\b", cv_text, re.I) for w in words)


def resume_fact_check(resume: dict[str, Any], cv_text: str, analysis: dict[str, Any]) -> list[str]:
    """Warnings about anything in the resume that the complete CV does not support."""
    cv_norm = normalize(cv_text)
    texts = _all_text(resume)
    tech_items = [t.strip() for e in resume.get("projects", []) + resume.get("experience", [])
                  for t in re.split(r"\s*[,;|·]\s*", e.get("tech", "")) if t.strip()]
    body_norm = normalize(" ".join(texts + tech_items + [i for g in resume.get("skills", []) for i in g["items"]]))
    warnings: list[str] = []

    for term in sorted(not_in_cv_terms(analysis, cv_text), key=len):
        if keyword_in_text(term, body_norm):
            warnings.append(f"Mentions **{term}**, which is not in your complete CV.")

    for item in [i for g in resume.get("skills", []) for i in g["items"]] + tech_items:
        main = re.sub(r"\(.*?\)", "", item).strip()
        if main and not keyword_in_text(main, cv_norm) and not evidence_in_text(main, cv_text):
            warnings.append(f"Skill **{item}** is not in your complete CV.")

    for key in ("experience", "projects", "education"):
        for e in resume.get(key, []):
            org = e["organisation"]
            if org and not keyword_in_text(org, cv_norm) and not evidence_in_text(org, cv_text):
                warnings.append(f"Organisation **{org}** is not in your complete CV.")
            for label, dates in [(e["title"] or org, e["dates"])] + [(s["name"], s["dates"])
                                                                     for s in e.get("subsections", [])]:
                for d in _DATE_TOKEN.findall(dates):
                    if not re.search(rf"\b{re.escape(d)}\b", cv_text):
                        warnings.append(f"Date **{d}** ({label}) is not in your complete CV.")

    lines = cv_lines(cv_text)
    for key in ("experience", "projects"):
        for e in resume.get(key, []):
            for b in e["bullets"] + [x for s in e.get("subsections", []) for x in s["bullets"]]:
                if lines and _best_line(b, lines)[1] < 0.5 and not evidence_in_text(b, cv_text):
                    warnings.append(f"Not clearly from your CV: “{b[:90]}{'…' if len(b) > 90 else ''}”")

    for text in texts:
        for m in _NUMBER.finditer(text):
            if not _number_backed(m.group(1), cv_text):
                warnings.append(f"Number **{m.group(1)}** does not appear in your complete CV.")

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
def finish_resume(data: dict[str, Any], cv_text: str, jd_text: str, language: str,
                  size: str = "one_page") -> dict[str, Any]:
    """Model answer -> final resume: limits, merged roles, CV skills, restored details."""
    resume = clean_resume(data, language, size)
    resume["skills"] = skills_for_job(cv_text, jd_text, resume["skills"])
    return keep_backed_education(restore_details(resume, cv_text), cv_text)


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
        resume = finish_resume(data, cv_text, jd_text, lang, size)
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
                if e.get("tech"):
                    lines.append(f"  {e['tech']}")
                lines += [f"  • {b}" for b in e["bullets"]]
                for sub in e.get("subsections", []):
                    lines.append(f"  {sub['name']}" + (f"  ({sub['dates']})" if sub["dates"] else ""))
                    lines += [f"    • {b}" for b in sub["bullets"]]
    return "\n".join(lines).strip()
