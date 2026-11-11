"""Rule-based checks for the German job market.

These run without the AI, so they are fast and predictable:
  * which language the job ad is written in
  * what German level the job asks for (CEFR A1..C2) vs. your level
  * job type (Werkstudent, internship, full-time ...) and weekly hours
  * a heads-up when hours could clash with the usual 20 h/week student limit
"""
from __future__ import annotations

import re
from typing import Any

from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

CEFR = ["A1", "A2", "B1", "B2", "C1", "C2"]

_LANG_WORD = r"(?:deutsch\w*|german)"
_STOP = r"[^.,;:!?\n]"  # stay inside one clause

# (descriptor regex, CEFR level). Order matters: more specific first.
_DESCRIPTORS: list[tuple[str, str]] = [
    (r"mutterspra\w*|native(?:[- ]level)?|native speaker", "C2"),
    (r"verhandlungssicher\w*|business[- ]fluent|business[- ]level|fluent(?:ly)?|flie(?:ß|ss)end\w*|"
     r"exzellent\w*|excellent|sehr gute\w*|very good|ausgezeichnet\w*|proficient|"
     r"strong command", "C1"),
    (r"gute\w*|good|solid\w*|fundierte\w*|working knowledge|conversational", "B2"),
    (r"grundkenntnisse|basic\w*|elementary|erste kenntnisse|beginner", "A2"),
]
_OPTIONAL = re.compile(
    r"von vorteil|wünschenswert|wuenschenswert|ein plus|is a plus|\bplus\b|nice to have|"
    r"advantage|preferred|desirable|ideally|idealerweise|optional|not required|nicht erforderlich|"
    r"keine voraussetzung", re.I)


def detect_language(text: str) -> str:
    """Return 'de' or 'en' (anything else falls back to 'en')."""
    try:
        lang = detect(text[:5000])
    except Exception:
        return "en"
    return "de" if lang == "de" else "en"


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    right_candidates = [i for i in (text.find(".", end), text.find("\n", end)) if i != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left:right].strip()


def find_german_requirement(jd_text: str) -> dict[str, Any] | None:
    """Find the German level a job ad asks for. Returns None if nothing is mentioned."""
    found: list[dict[str, Any]] = []
    # drop dots in abbreviations so "mind. B2" / "e.g. C1" stay inside one clause
    jd_text = re.sub(r"\b(mind|min|ca|bzw|z\.b|e\.g|i\.e|inkl|incl)\.", r"\1", jd_text, flags=re.I)
    lower = jd_text.lower()

    # 1) explicit CEFR level near the word German/Deutsch
    explicit = re.compile(
        rf"{_LANG_WORD}{_STOP}{{0,40}}?\b([abc][12])\b|\b([abc][12])\b{_STOP}{{0,30}}?{_LANG_WORD}", re.I)
    for m in explicit.finditer(lower):
        level = (m.group(1) or m.group(2)).upper()
        sentence = _sentence_around(jd_text, m.start(), m.end())
        found.append({"level": level, "optional": bool(_OPTIONAL.search(sentence)),
                      "evidence": sentence, "explicit": True})

    # 2) descriptive wording ("fließend Deutsch", "very good German", "gute Deutschkenntnisse")
    if not found:
        for pattern, level in _DESCRIPTORS:
            rx = re.compile(
                rf"\b(?:{pattern})\b{_STOP}{{0,40}}?{_LANG_WORD}|{_LANG_WORD}{_STOP}{{0,20}}?\b(?:{pattern})\b", re.I)
            for m in rx.finditer(lower):
                sentence = _sentence_around(jd_text, m.start(), m.end())
                found.append({"level": level, "optional": bool(_OPTIONAL.search(sentence)),
                              "evidence": sentence, "explicit": False})

    # 3) German mentioned as merely a plus without a level
    if not found:
        m = re.search(rf"{_LANG_WORD}{_STOP}{{0,40}}?(?:{_OPTIONAL.pattern})", lower, re.I)
        if m:
            found.append({"level": "A2", "optional": True,
                          "evidence": _sentence_around(jd_text, m.start(), m.end()), "explicit": False})

    if not found:
        return None
    required = [f for f in found if not f["optional"]]
    pool = required or found
    return max(pool, key=lambda f: CEFR.index(f["level"]))


def find_cv_german_level(cv_text: str) -> str | None:
    """Try to read the candidate's German level from the CV (e.g. 'German – A2')."""
    m = re.search(rf"{_LANG_WORD}{_STOP}{{0,30}}?\b([ABC][12])\b", cv_text, re.I)
    return m.group(1).upper() if m else None


def detect_job_type(jd_text: str) -> str:
    t = jd_text.lower()
    checks = [
        ("working student", r"werkstudent|working student|studentische[nr]? (?:hilfskraft|mitarbeit)|student assistant|\bhiwi\b"),
        ("thesis", r"abschlussarbeit|masterarbeit|bachelorarbeit|master'?s thesis|thesis"),
        ("internship", r"praktik|internship|\bintern\b"),
        ("minijob", r"minijob|geringfügig"),
        ("full-time", r"vollzeit|full[- ]time"),
        ("part-time", r"teilzeit|part[- ]time"),
    ]
    for label, rx in checks:
        if re.search(rx, t):
            return label
    return "unknown"


def detect_weekly_hours(jd_text: str) -> int | None:
    """Return the maximum weekly hours mentioned, e.g. '15–20 Std./Woche' -> 20."""
    rx = re.compile(
        r"(\d{1,2})\s*(?:(?:-|–|bis|to)\s*(\d{1,2})\s*)?(?:std\.?|stunden|hours|hrs|h)\b"
        r"(?:\s*(?:pro|per|/|a|in der|each)\s*(?:woche|week|wk))?", re.I)
    values: list[int] = []
    for m in rx.finditer(jd_text):
        nums = [int(x) for x in m.groups() if x]
        tail = jd_text[m.end():m.end() + 25].lower()
        context = jd_text[max(0, m.start() - 25):m.end() + 25].lower()
        if any(w in context for w in ("woche", "week", "wöchentlich", "weekly")) or "/" in tail[:3]:
            values.extend(n for n in nums if 1 <= n <= 60)
    return max(values) if values else None


def language_score_cap(required: str | None, candidate: str | None, optional: bool) -> float | None:
    """Maximum language score allowed when German is a hard requirement you don't meet."""
    if not required or optional or not candidate:
        return None
    gap = CEFR.index(required) - CEFR.index(candidate)
    if gap <= 0:
        return None
    return {1: 3.5, 2: 2.5}.get(gap, 1.5)


def run_checks(jd_text: str, cv_text: str, profile: dict[str, Any]) -> dict[str, Any]:
    flags: list[dict[str, str]] = []
    job_lang = detect_language(jd_text)
    req = find_german_requirement(jd_text)
    candidate = (profile.get("german_level") or "").upper() or find_cv_german_level(cv_text)
    job_type = detect_job_type(jd_text)
    hours = detect_weekly_hours(jd_text)

    flags.append({"level": "info", "title": f"Job ad is written in {'German' if job_lang == 'de' else 'English'}",
                  "detail": "The cover letter will be written in the same language."})

    if req is None:
        if job_lang == "de":
            flags.append({"level": "warning", "title": "No German level stated, but the ad is in German",
                          "detail": "Day-to-day German is probably expected — usually around B2 or higher."})
        else:
            flags.append({"level": "success", "title": "No German requirement found", "detail": ""})
    elif not candidate:
        flags.append({"level": "warning", "title": f"Job asks for German {req['level']}",
                      "detail": "Set your German level in 'My profile' so this can be checked."})
    else:
        gap = CEFR.index(req["level"]) - CEFR.index(candidate)
        need = "nice-to-have" if req["optional"] else "required"
        detail = f"Job: {req['level']} ({need}) · You: {candidate}. Found in ad: “{req['evidence'][:160]}”"
        if gap <= 0:
            flags.append({"level": "success", "title": "Your German level meets the requirement", "detail": detail})
        elif req["optional"]:
            flags.append({"level": "info", "title": f"German {req['level']} is only a plus", "detail": detail})
        elif gap == 1:
            flags.append({"level": "warning", "title": f"German gap: one level below ({candidate} → {req['level']})",
                          "detail": detail + " Mention your current course/progress in the letter."})
        else:
            flags.append({"level": "error", "title": f"Big German gap ({candidate} → {req['level']})",
                          "detail": detail + " This is often a hard filter for German employers."})

    is_student = bool(profile.get("is_student", True))
    other_hours = int(profile.get("other_job_hours") or 0)
    if is_student:
        if job_type == "full-time":
            flags.append({"level": "warning", "title": "Full-time role",
                          "detail": "While enrolled, working more than 20 h/week during lecture periods can affect "
                                    "student status and work permissions. Fine for after graduation or semester breaks — "
                                    "check your own situation."})
        elif hours is not None and hours + other_hours > 20:
            flags.append({"level": "warning", "title": f"Hours check: {hours} h + {other_hours} h other job = {hours + other_hours} h/week",
                          "detail": "That is above the usual 20 h/week limit for students during lecture periods. "
                                    "Check with your university / Ausländerbehörde if this applies to you."})
        elif hours is not None:
            flags.append({"level": "success", "title": f"Hours fit: {hours} h + {other_hours} h other job",
                          "detail": "Within the usual 20 h/week student limit."})
    if job_type != "unknown":
        flags.append({"level": "info", "title": f"Job type: {job_type}", "detail": ""})

    return {
        "job_language": job_lang,
        "german_requirement": req,
        "candidate_german_level": candidate,
        "language_cap": language_score_cap(req["level"] if req else None, candidate,
                                           req["optional"] if req else True),
        "job_type": job_type,
        "weekly_hours": hours,
        "flags": flags,
    }
