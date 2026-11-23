"""Tailored resume from the complete CV: size limits, fact check, rewrite loop, exports, storage."""
import io

from pypdf import PdfReader

from matcher import schemas
from matcher.export import resume_to_docx, resume_to_pdf
from matcher.pipeline import analyse
from matcher.resume import clean_resume, generate_resume, resume_fact_check, resume_to_text
from matcher.storage import Storage
from tests.fake_llm import SAMPLE_CV, SAMPLE_JD_DE, SAMPLE_JD_EN, FakeModels, honest_resume

PROFILE = {"full_name": "Max Mustermann", "postal_city": "10115 Berlin", "email": "max@example.com",
           "german_level": "A2"}


class InventingModels(FakeModels):
    """First draft invents things; the rewrite is honest."""

    def respond(self, schema, system, user):
        if schema is schemas.TailoredResume and self.calls.count("TailoredResume") == 1:
            bad = honest_resume()
            bad["skills"][0]["items"].append("Airflow")
            bad["experience"][0]["organisation"] = "Google"
            bad["experience"][0]["dates"] = "2021"
            bad["experience"][0]["bullets"][0] = "Cut report time by 40% with Python ETL scripts"
            bad["languages"] = ["German (C1)"]
            return bad
        return super().respond(schema, system, user)


def test_honest_resume_passes_and_respects_size_limits():
    llm = FakeModels()
    a = analyse(llm, SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    run = generate_resume(llm, SAMPLE_CV, SAMPLE_JD_EN, a, size="one_page")
    assert run["issues"] == [] and run["rewrites"] == 0
    resume = run["resume"]
    assert resume["language"] == "en" and len(resume["experience"][0]["bullets"]) == 3   # trimmed from 4
    prompt = llm.prompts["TailoredResume"]
    assert "COMPLETE CV" in prompt and "Airflow" in prompt                                  # do-not-claim list
    text = resume_to_text(resume, PROFILE)
    assert text.startswith("Max Mustermann") and "EXPERIENCE" in text and "Acme GmbH" in text


def test_inventions_are_flagged_and_rewritten():
    llm = InventingModels()
    a = analyse(llm, SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    run = generate_resume(llm, SAMPLE_CV, SAMPLE_JD_EN, a)
    assert llm.calls.count("TailoredResume") == 2 and run["rewrites"] == 1 and run["issues"] == []
    assert "Fix these problems" in llm.prompts["TailoredResume"] and "Google" in llm.prompts["TailoredResume"]


def test_fact_check_finds_each_kind_of_invention():
    a = analyse(FakeModels(), SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    bad = honest_resume()
    bad["skills"][0]["items"].append("Airflow")
    bad["experience"][0].update(organisation="Google", dates="2021",
                                bullets=["Cut report time by 40% with Python ETL scripts"])
    bad["languages"] = ["German (C1)"]
    issues = " ".join(resume_fact_check(clean_resume(bad, "en"), SAMPLE_CV, a))
    for expected in ("Airflow", "Google", "2021", "40", "German (C1)"):
        assert expected in issues, expected


def test_german_job_gets_german_headings_and_exports():
    llm = FakeModels()
    a = analyse(llm, SAMPLE_CV, SAMPLE_JD_DE, PROFILE)
    resume = generate_resume(llm, SAMPLE_CV, SAMPLE_JD_DE, a, size="two_pages")["resume"]
    assert resume["language"] == "de"
    assert "BERUFSERFAHRUNG" in resume_to_text(resume, PROFILE)
    pdf = resume_to_pdf(resume, PROFILE)
    text = "".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "Max Mustermann" in text and "Acme GmbH" in text and "BERUFSERFAHRUNG" in text
    from docx import Document
    doc = Document(io.BytesIO(resume_to_docx(resume, PROFILE)))
    doc_text = "\n".join(p.text for p in doc.paragraphs)
    assert "Max Mustermann" in doc_text and "Created Power BI dashboards for the sales team" in doc_text


def test_storage_master_cv_and_resume(tmp_path):
    s = Storage(tmp_path / "t.db")
    assert s.get_master_cv() is None
    s.save_master_cv("  " + SAMPLE_CV + "  ")
    assert s.get_master_cv()["text"] == SAMPLE_CV.strip()
    a = analyse(FakeModels(), SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    app_id = s.add_application(a, SAMPLE_JD_EN, cv_text=SAMPLE_CV)
    s.update_application(app_id, resume_json={"headline": "x", "language": "en"})
    assert s.get_application(app_id)["resume"]["headline"] == "x"
    s.clear_master_cv()
    assert s.get_master_cv() is None


def test_number_words_and_engineer_alias_are_not_false_alarms():
    cv = SAMPLE_CV + "\nPROFILE\nSoftware engineer with four years of experience.\n"
    a = analyse(FakeModels(), cv, SAMPLE_JD_EN, PROFILE)
    resume = clean_resume({**honest_resume(), "summary": "Software engineering background, 4 years of experience."},
                          "en")
    assert resume_fact_check(resume, cv, a) == []
    resume["summary"] = "5 years of experience."
    assert any("**5**" in w for w in resume_fact_check(resume, cv, a))


# ---------------------------------------------------------------- structure, details, skills
FULL_CV = """PROFILE
Embedded software engineer with four years of C/C++ firmware experience.
EXPERIENCE
Software Developer 11/2021 – 09/2025
Quest Global Engineering Services, Trivandrum, India
Edge AI hardware benchmarking 2024 – 2025
• Benchmarked six AI-capable MCU and MPU platforms: Renesas RA, STM32, NXP, Hailo, NVIDIA Jetson, TI AM69
measuring inference latency, energy per inference and real time responsiveness.
Networked audio firmware 2022 – 2024
• Debugged firmware directly on target hardware using JTAG and SWD to isolate faults.
• Upgraded UI firmware, improving interface responsiveness by 30%.
SKILLS
Languages : C, C++, Python
Embedded : JTAG, SWD, STM32
Methods : Agile-Scrum, CI, Unit & regression testing
CERTIFICATIONS & LANGUAGES
Languages : English – C2 • German – B1
"""
SCRUM_JD = "Working student software engineering. You join a Scrum team and help with testing. " * 4


def _entry(title, dates, bullets, org="Quest Global Engineering Services"):
    return {"title": title, "organisation": org, "location": "Trivandrum, India", "dates": dates, "tech": "",
            "bullets": bullets, "subsections": []}


def test_split_roles_are_merged_and_lost_details_restored():
    data = {"headline": "Embedded Software Engineer",
            "summary": "Embedded software engineer with four years of C/C++ firmware experience.",
            "projects": [], "education": [], "languages": ["English – C2", "German – B1"], "certifications": [],
            "skills": [{"category": "Programming", "items": ["C", "JavaScript"]}], "experience": [
        _entry("Software Developer", "11/2021 – 09/2025", []),
        _entry("Edge AI Hardware Benchmarking", "2024 – 2025",
               ["Benchmarked several AI platforms for latency and energy."]),        # specifics generalised away
        _entry("Networked Audio Firmware", "2022 – 2024",
               ["Debugged firmware on target hardware, improving responsiveness by 30%.",   # number moved here
                "Upgraded UI firmware, improving interface responsiveness by 30%."]),
    ]}
    from matcher.resume import finish_resume
    r = finish_resume(data, FULL_CV, SCRUM_JD, "en", "one_page")
    assert len(r["experience"]) == 1                                                  # one role …
    subs = r["experience"][0]["subsections"]
    assert [s["name"] for s in subs] == ["Edge AI Hardware Benchmarking", "Networked Audio Firmware"]  # … 2 projects
    assert "Renesas RA, STM32, NXP, Hailo, NVIDIA Jetson, TI AM69" in subs[0]["bullets"][0]
    assert "JTAG and SWD" in subs[1]["bullets"][0] and "30%" not in subs[1]["bullets"][0]
    assert subs[1]["bullets"][1].endswith("by 30%.")
    # skills come from the CV's skills section: all groups, Scrum/testing first, nothing invented
    assert r["skills"][0]["category"] == "Methods" and r["skills"][0]["items"][0] == "Agile-Scrum"
    assert {g["category"] for g in r["skills"]} == {"Methods", "Programming Languages", "Embedded"}
    assert "JavaScript" not in str(r["skills"])
    a = analyse(FakeModels(), FULL_CV, SAMPLE_JD_EN, PROFILE)
    assert resume_fact_check(r, FULL_CV, a) == []
    text = resume_to_text(r, PROFILE)
    assert "  Networked Audio Firmware  (2022 – 2024)" in text and text.count("Quest Global") == 1
    pdf = resume_to_pdf(r, PROFILE)
    pdf_text = "".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "Networked Audio Firmware" in pdf_text and "Agile-Scrum" in pdf_text
    from docx import Document
    doc_text = "\n".join(p.text for p in Document(io.BytesIO(resume_to_docx(r, PROFILE))).paragraphs)
    assert "Networked Audio Firmware" in doc_text and "JTAG and SWD" in doc_text


def test_cv_without_skills_section_keeps_only_backed_model_skills():
    from matcher.resume import parse_cv_skills, skills_for_job
    cv = "EXPERIENCE\n• Wrote Python and SQL scripts for reports.\n"
    assert parse_cv_skills(cv) == []
    out = skills_for_job(cv, "Python job", [{"category": "Programming", "items": ["Python", "SQL", "Rust"]}])
    assert out == [{"category": "Programming", "items": ["Python", "SQL"]}]


def test_bullets_filed_under_their_project_and_titles_cleaned():
    from matcher.resume import finish_resume
    data = {"headline": "", "summary": "", "skills": [], "languages": [], "certifications": [],
            "experience": [{**_entry("Software Developer", "11/2021 – 09/2025",
                                     ["Upgraded UI firmware, improving interface responsiveness by 30%."]),
                            "subsections": [{"name": "Networked audio firmware", "dates": "2022 – 2024",
                                             "bullets": []},
                                            {"name": "Edge AI hardware benchmarking", "dates": "", "bullets": []}]}],
            "projects": [{"title": "CNN classifier", "organisation": "Personal Project, 2026 • Link",
                          "location": "2026", "dates": "", "tech": "", "bullets": [], "subsections": []}],
            "education": [{"title": "B.Tech", "organisation": "", "location": "", "dates": "", "tech": "",
                           "bullets": ["Completed with a focus on edge-AI benchmarking."], "subsections": []}]}
    r = finish_resume(data, FULL_CV + "EDUCATION\nB.Tech Computer Science 2017 – 2021\n", SCRUM_JD, "en")
    role = r["experience"][0]
    assert role["bullets"] == [] and [s["name"] for s in role["subsections"]] == ["Networked audio firmware"]
    assert role["subsections"][0]["bullets"] == ["Upgraded UI firmware, improving interface responsiveness by 30%."]
    p = r["projects"][0]
    assert (p["organisation"], p["location"], p["dates"]) == ("Personal Project", "", "2026")
    assert r["education"][0]["bullets"] == []                     # not in the CV's education section
