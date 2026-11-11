"""Cover letter generation (content) — layout/export lives in export.py."""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from .llm import Models
from .textutils import keyword_in_text, normalize

LENGTHS = {
    "short": (180, 3),
    "standard": (280, 4),
    "detailed": (380, 5),
}

TONES = {
    "formal": {
        "de": "classic formal German business style (Sie-Form), polite and precise",
        "en": "formal, professional business English",
    },
    "modern": {
        "de": "modern and personal, still professional (Sie-Form unless the ad uses 'du'; if the ad uses 'du', use 'du')",
        "en": "modern, warm and confident, still professional",
    },
}



def salutation_hint(lang: str, tone: str, contact: str) -> str:
    if lang == "de":
        if contact:
            return (f"Address '{contact}' by name: 'Sehr geehrte Frau …,' or 'Sehr geehrter Herr …,' "
                    f"(guess gender from the first name only if obvious, otherwise 'Guten Tag {contact},')."
                    if tone == "formal" else f"'Hallo {contact},' or 'Guten Tag {contact},'")
        return "'Sehr geehrte Damen und Herren,'" if tone == "formal" else "'Liebes [Company]-Team,' using the company name"
    if contact:
        return f"'Dear {contact},'"
    return "'Dear Hiring Team,'" if tone == "formal" else "'Dear [Company] team,' using the company name"


def default_closing(lang: str, tone: str) -> str:
    if lang == "de":
        return "Mit freundlichen Grüßen" if tone == "formal" else "Viele Grüße"
    return "Kind regards" if tone == "formal" else "Best regards"


def format_date(lang: str, d: date | None = None) -> str:
    d = d or date.today()
    if lang == "de":
        return f"{d.day:02d}.{d.month:02d}.{d.year}"
    return f"{d.day} {d.strftime('%B')} {d.year}"


def _strip_name(text: str, name: str) -> str:
    if name:
        text = re.sub(rf"[,\s]*{re.escape(name)}\s*$", "", text.strip(), flags=re.I)
    return text.strip()


def _clean_closing(closing: str, name: str, fallback: str) -> str:
    closing = _strip_name(closing, name)
    closing = closing.split("\n")[0].split(",")[0].strip()  # "Kind regards, Vinay" -> "Kind regards"
    return closing or fallback


def _german_rule(level: str | None) -> str:
    if level:
        return (f"the candidate's German level is {level}. If you mention German at all, state exactly {level} "
                "(e.g. 'German at {level} level, improving') — never a higher level.").replace("{level}", level)
    return "the German level is unknown — do not mention a German level."


def letter_inputs(cv_text: str, jd_text: str, analysis: dict[str, Any], profile: dict[str, Any],
                  tone: str = "formal", length: str = "standard", extra_instructions: str = "") -> dict[str, Any]:
    """All placeholder values for the cover letter prompt (see prompts.COVER_USER)."""
    lang = analysis["germany"]["job_language"]
    language = "German" if lang == "de" else "English"
    words, paragraphs = LENGTHS.get(length, LENGTHS["standard"])
    availability = str(profile.get("availability") or "").strip()
    reqs = analysis.get("requirements", [])
    extra = extra_instructions.strip()
    return {
        "language": language,
        "tone_text": TONES.get(tone, TONES["formal"])[lang],
        "words": words,
        "paragraphs": paragraphs,
        "availability_rule": (f"mention the candidate's availability exactly as: \"{availability}\"." if availability
                              else "do NOT mention availability, start dates or notice periods."),
        "do_not_claim": ", ".join(not_in_cv_terms(analysis, cv_text)) or "(none)",
        "education": "; ".join(sorted(set(_education_lines(cv_text)))) or "see CV",
        "german_rule": _german_rule(analysis["germany"].get("candidate_german_level")),
        "matches": "\n".join(f"- {r['name']} → {r['evidence']}" for r in reqs
                             if r["status"] in ("met", "partial") and r.get("evidence")) or "- (none)",
        "salutation_hint": salutation_hint(lang, tone, analysis["job"].get("contact_person", "")),
        "closing": default_closing(lang, tone),
        "extra": f"Extra instructions from the candidate (follow them): {extra}" if extra else "",
        "jd": jd_text[:10000],
        "strengths": "; ".join(analysis.get("strengths", [])),
        "cv": cv_text[:10000],
    }


def clean_letter(data: dict[str, Any], analysis: dict[str, Any], profile: dict[str, Any],
                 tone: str = "formal") -> dict[str, Any]:
    """Tidy the model's letter: no name in the closing, no closing formula inside the body."""
    lang = analysis["germany"]["job_language"]
    closing = default_closing(lang, tone)
    name = str(profile.get("full_name") or "").strip()
    paras = [str(p).strip() for p in (data.get("paragraphs") or []) if str(p).strip()]
    if paras:
        paras[-1] = _strip_name(paras[-1], name) or paras[-1]
        if paras[-1].lower().rstrip(",. ") in (closing.lower(), "kind regards", "best regards",
                                               "mit freundlichen grüßen", "viele grüße"):
            paras.pop()  # closing formula leaked into the body
    return {
        "language": lang,
        "subject": str(data.get("subject") or "").strip(),
        "salutation": str(data.get("salutation") or "").strip(),
        "paragraphs": paras,
        "closing": _clean_closing(str(data.get("closing") or ""), name, closing),
    }


def generate_cover_letter(models: Models, cv_text: str, jd_text: str, analysis: dict[str, Any],
                          profile: dict[str, Any], tone: str = "formal", length: str = "standard",
                          extra_instructions: str = "") -> dict[str, Any]:
    """One letter, no fix loop (the graph in graph.py adds the fact-check-and-rewrite loop)."""
    from .chains import cover_letter_chain

    result = cover_letter_chain(models).invoke(
        letter_inputs(cv_text, jd_text, analysis, profile, tone, length, extra_instructions))
    return clean_letter(result.model_dump(), analysis, profile, tone)


# --------------------------------------------------------------------------- fact check
_COMMON_WORDS = {"experience", "knowledge", "development", "develop", "design", "with", "using", "skills",
                 "erfahrung", "kenntnisse", "entwicklung", "and", "und", "oder", "tools", "systems", "system",
                 "technology", "technologies", "code", "testing", "tests", "software", "hardware", "management",
                 "configuration", "solutions", "components", "component", "mobile", "apps", "console",
                 "deliver", "collaborate", "designers", "team", "teams", "source", "versioning", "level"}


def _education_lines(cv_text: str) -> list[str]:
    rx = re.compile(r"\b(?:b\.?\s?tech|b\.?\s?e\.|b\.?\s?eng|b\.?\s?sc|bsc|bachelor\w*|m\.?\s?sc|msc|m\.?\s?tech|"
                    r"master\w*|ph\.?\s?d)\b", re.I)
    return [ln.strip()[:120] for ln in cv_text.splitlines() if rx.search(ln)][:4]


def not_in_cv_terms(analysis: dict[str, Any], cv_text: str) -> list[str]:
    """Tools/skills the job wants that the CV does not contain — the letter must not claim them."""
    cv_norm = normalize(cv_text)
    terms: list[str] = list(analysis.get("categories", {}).get("keywords", {}).get("missing", []))
    for r in analysis.get("requirements", []):
        if r["status"] != "missing":
            continue
        name = r["name"].strip()
        words = re.findall(r"[\w+#./-]+", name)
        if len(words) <= 4:
            terms.append(name)
        # the tool after "with/using/in/mit": "Continuous Integration with Jenkins" -> "Jenkins"
        m = re.search(r"\b(?:with|using|via|in|mit|über)\s+(.+)$", name, re.I)
        if m and len(m.group(1).split()) <= 3:
            terms.append(m.group(1))
        # acronyms / CamelCase tool names: RHEL, PyTest, TypeScript
        terms += [w for w in words if (w.isupper() and len(w) >= 2) or re.match(r"^[A-Z][a-z]+[A-Z]", w)]
    out: list[str] = []
    for t in terms:
        t = t.strip()
        if t and not keyword_in_text(t, cv_norm) and t.lower() not in {o.lower() for o in out}:
            out.append(t)
    return out


_LEARNING = re.compile(
    r"\b(?:learn\w*|eager|keen|look(?:ing)? forward|familiari[sz]e\w*|get up to speed|pick up|not yet|"
    r"new to|have not|haven't|no (?:prior|hands-on|direct)|deepen|expand|grow into|"
    r"lernen|erlernen|einarbeiten|einzuarbeiten|aneignen|anzueignen|vertiefen|noch nicht|"
    r"keine (?:erfahrung|kenntnisse)|freue mich darauf)\b", re.I)

_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                 "eight": "8", "nine": "9", "ten": "10", "ein": "1", "zwei": "2", "drei": "3", "vier": "4",
                 "fünf": "5", "sechs": "6", "sieben": "7", "acht": "8", "neun": "9", "zehn": "10"}


def fact_check(letter: dict[str, Any], cv_text: str, analysis: dict[str, Any],
               profile: dict[str, Any]) -> list[str]:
    """Warnings about claims in the letter that the CV / profile do not support."""
    body = " ".join([letter.get("subject", "")] + letter.get("paragraphs", []))
    body_norm = normalize(body)
    warnings: list[str] = []

    flagged: list[str] = []
    sentences = [normalize(x) for x in re.split(r"(?<=[.!?])\s+", body)]
    for term in sorted(not_in_cv_terms(analysis, cv_text), key=len):
        if not keyword_in_text(term, body_norm) or any(f.lower() in term.lower() for f in flagged):
            continue
        # "keen to learn Jenkins" is honest; only flag sentences that present it as experience
        claimed = [x for x in sentences if keyword_in_text(term, x) and not _LEARNING.search(x)]
        if claimed:
            flagged.append(term)
            warnings.append(f"Mentions **{term}**, which is not in your CV — make sure it's not claimed as experience.")

    if _education_lines(cv_text) and re.search(
            r"(?:do not|don't|without|lack\w*|no)\s+(?:have\s+)?(?:a\s+)?bachelor|kein\w*\s+bachelor", body, re.I):
        warnings.append("Says you **don't have a Bachelor's degree**, but your CV lists one.")

    if not str(profile.get("availability") or "").strip() and re.search(
            r"immediate|available (?:from|to start|starting)|start date|notice period|ab sofort|"
            r"sofort verfügbar|verfügbar ab|eintrittstermin", body, re.I):
        warnings.append("Mentions **availability**, but none is set in My profile.")

    level = analysis.get("germany", {}).get("candidate_german_level")
    for m in re.finditer(r"(?:german|deutsch)\w*[^.]{0,40}?\b([ABC][12])\b|\b([ABC][12])\b[^.]{0,25}?(?:german|deutsch)",
                         body, re.I):
        stated = (m.group(1) or m.group(2)).upper()
        if level and stated != level:
            warnings.append(f"States German **{stated}**, but your level is **{level}**.")

    for m in re.finditer(r"\b(\d+(?:[.,]\d+)?)\s?(%|prozent|percent)", body, re.I):
        if not re.search(rf"\b{re.escape(m.group(1))}\s?(?:%|prozent|percent)", cv_text, re.I):
            warnings.append(f"Number **{m.group(0).strip()}** does not appear in your CV.")
    for m in re.finditer(rf"\b(\d+|{'|'.join(_NUMBER_WORDS)})\+?\s?(?:years|jahre[n]?)\b", body, re.I):
        n = m.group(1).lower()
        forms = {n, _NUMBER_WORDS.get(n, n), *[w for w, d in _NUMBER_WORDS.items() if d == n]}
        if not any(re.search(rf"\b{re.escape(f)}\+?\s?(?:years|jahre)", cv_text, re.I) for f in forms):
            warnings.append(f"**{m.group(0).strip()}** does not match the experience stated in your CV.")

    return list(dict.fromkeys(warnings))


def letter_to_text(letter: dict[str, Any]) -> str:
    """Editable plain-text body: subject, salutation, paragraphs, closing."""
    parts = [letter.get("subject", ""), "", letter.get("salutation", ""), ""]
    parts += [p for para in letter.get("paragraphs", []) for p in (para, "")]
    parts.append(letter.get("closing", ""))
    return "\n".join(parts).strip()


def text_to_letter(text: str, language: str) -> dict[str, Any]:
    """Parse the edited text back: line 1 = subject, next = salutation, last = closing."""
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    if len(blocks) < 3:
        return {"language": language, "subject": blocks[0] if blocks else "", "salutation": "",
                "paragraphs": blocks[1:], "closing": ""}
    return {"language": language, "subject": blocks[0], "salutation": blocks[1],
            "paragraphs": blocks[2:-1], "closing": blocks[-1]}
