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
