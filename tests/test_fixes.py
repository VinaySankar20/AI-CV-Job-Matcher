"""Regression tests for problems found on the first real run (Firmware Developer job)."""
from matcher import schemas
from matcher.cover_letter import _clean_closing, fact_check, generate_cover_letter, not_in_cv_terms
from matcher.pdf_reader import clean_text
from matcher.pipeline import analyse
from matcher.scorer import _similar_names, normalize_job
from tests.fake_llm import FakeModels

CV = """Vinay Example
EXPERIENCE
Software Developer, Quest Global (11/2021 - 09/2025)
- Developed backend firmware in C/C++ for networked HiFi music players
- Wrote 173 automated tests (pytest) with faked hardware
EDUCATION
09/2025 - Present M.Sc. Industrial Artificial Intelligence
08/2017 - 06/2021 B.Tech Computer Science and Engineering
SKILLS
Languages C, C++, Python, C#, JavaScript
Tools Git, SVN, Jira, Linux
Methods Agile-Scrum, CI, unit & regression testing
Languages English - fluent, German - B1 (improving)
"""

JD = "Firmware Developer at IBM. You have a Bachelor's degree, Linux and Git. Nice: Python, C or C++, Jenkins."


class SkippingModels(FakeModels):
    """Behaves like the 4B model: answers only the must-haves, even on the re-check."""

    def respond(self, schema, system, user):
        if schema is schemas.JobExtraction:
            return {"job_title": "Firmware Developer", "company": "IBM", "location": "", "contact_person": "",
                    "min_years_experience": 0, "keywords": ["Linux", "Git", "Python", "Jenkins"],
                    "must_have": [{"name": "Bachelor's Degree", "category": "education"},
                                  {"name": "Linux Operating System", "category": "skill"},
                                  {"name": "Code Versioning with Git", "category": "skill"}],
                    "nice_to_have": [{"name": "Python", "category": "skill"},
                                     {"name": "C or C++", "category": "skill"},
                                     {"name": "Source Code Versioning with Git", "category": "skill"},
                                     {"name": "Continuous Integration with Jenkins", "category": "skill"}]}
        if schema is schemas.Evaluation:
            return {"requirements": [
                        {"id": "M1", "status": "missing", "evidence": ""},
                        {"id": "M2", "status": "met", "evidence": "Tools Git, SVN, Jira, Linux"},
                        {"id": "M3", "status": "met", "evidence": "Tools Git, SVN, Jira, Linux"}],
                    "scores": {k: {"score": 4, "reason": ""} for k in ("skills", "experience", "education", "languages")},
                    "strengths": [], "gaps": [], "cv_tips": []}
        if schema is schemas.Recheck:           # the re-check pass: still skips everything
            return {"requirements": []}
        if schema is schemas.CoverLetter:
            return {"subject": "Application", "salutation": "Dear Hiring Team,",
                    "paragraphs": ["I applied CI with Jenkins.", "Kind regards, Vinay Example"],
                    "closing": "Kind regards, Vinay Example"}
        return super().respond(schema, system, user)


def test_inter_font_symbols_are_restored():
    raw = "C/C and AIAssisted Ollama on EK -RA8D1"
    assert clean_text(raw) == "C/C++ and AI-Assisted (Ollama) on EK-RA8D1"


def test_duplicates_removed():
    assert _similar_names("Code Versioning with Git", "Source Code Versioning with Git")
    assert _similar_names("Debugging and Testing", "Debug and Test Code")
    assert not _similar_names("Python", "Strong Python skills")
    job = normalize_job(SkippingModels().respond(schemas.JobExtraction, "", JD))
    assert [r["name"] for r in job["requirements"]].count("Source Code Versioning with Git") == 0


def test_skipped_and_wrong_verdicts_are_corrected():
    llm = SkippingModels()
    a = analyse(llm, CV, JD, {"german_level": ""})
    by = {r["name"]: r for r in a["requirements"]}
    assert llm.calls == ["JobExtraction", "Evaluation", "Recheck"]   # exactly one re-check, then on
    assert [t["node"] for t in a["trace"]].count("recheck") == 1
    assert by["Bachelor's Degree"]["status"] == "met"           # B.Tech counts
    assert "B.Tech" in by["Bachelor's Degree"]["evidence"]
    assert by["Python"]["status"] == "met"                      # keyword found, AI skipped it
    assert by["C or C++"]["status"] == "met"                    # alternative matched
    assert by["Continuous Integration with Jenkins"]["status"] == "missing"
    assert a["penalty"]["missing_must_have"] == 0


def test_letter_prompt_forbids_missing_skills_and_cleans_closing():
    llm = SkippingModels()
    a = analyse(llm, CV, JD, {})
    profile = {"full_name": "Vinay Example", "availability": "Immediate joining possible"}
    letter = generate_cover_letter(llm, CV, JD, a, profile)
    prompt = llm.prompts["CoverLetter"]
    assert "Jenkins" in prompt.split("NEVER claim")[1].split("\n")[0]
    assert "Immediate joining possible" in prompt
    assert "B1" in prompt
    assert letter["closing"] == "Kind regards"
    assert letter["paragraphs"] == ["I applied CI with Jenkins."]   # leaked closing removed
    warnings = fact_check(letter, CV, a, profile)
    assert any("Jenkins" in w for w in warnings)


def test_fact_check_catches_false_claims():
    a = {"requirements": [{"name": "Continuous Integration with Jenkins", "status": "missing"}],
         "categories": {"keywords": {"missing": ["Jenkins"]}}, "germany": {"candidate_german_level": "B1"}}
    letter = {"subject": "x", "paragraphs": [
        "I do not have a Bachelor's degree. I am immediately available. My German is at B2 level.",
        "I cut costs by 55% in six years."]}
    w = " ".join(fact_check(letter, CV, a, {"availability": ""}))
    assert "Bachelor" in w and "availability" in w and "B2" in w and "55%" in w and "six years" in w
    assert "Jenkins" in not_in_cv_terms(a, CV)
    ok = {"subject": "x", "paragraphs": ["I wrote 173 automated tests and know C++ and Git.",
                                         "I have not used Jenkins yet and am keen to learn it."]}
    assert fact_check(ok, CV, a, {"availability": ""}) == []       # honest "keen to learn" is fine


def test_clean_closing():
    assert _clean_closing("Kind regards, Vinay Example", "Vinay Example", "Kind regards") == "Kind regards"
    assert _clean_closing("", "X", "Mit freundlichen Grüßen") == "Mit freundlichen Grüßen"
