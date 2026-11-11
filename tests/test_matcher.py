import io

import pytest
from pypdf import PdfReader

from matcher import config, germany
from matcher.cover_letter import format_date, generate_cover_letter, letter_to_text, text_to_letter
from matcher.export import to_docx, to_pdf
from matcher.pdf_reader import PDFTextError, extract_pdf_text
from matcher.pipeline import analyse
from matcher.storage import Storage
from matcher.tailor import tailor_cv
from matcher.textutils import keyword_coverage, keyword_in_text, normalize
from tests.fake_llm import SAMPLE_CV, SAMPLE_JD_DE, SAMPLE_JD_EN, FakeModels

PROFILE = {"full_name": "Max Mustermann", "street": "Musterstraße 1", "postal_city": "10115 Berlin",
           "email": "max@example.com", "phone": "+49 151 000000", "german_level": "A2",
           "is_student": True, "other_job_hours": 0}


# ---------------------------------------------------------------- Germany rules
@pytest.mark.parametrize("text,level,optional", [
    ("Sehr gute Deutsch- und Englischkenntnisse in Wort und Schrift.", "C1", False),
    ("Gute Englischkenntnisse, Deutsch von Vorteil.", "A2", True),
    ("Fluent English is required. German is a plus.", "A2", True),
    ("Deutschkenntnisse (mind. B2) erforderlich", "B2", False),
    ("Verhandlungssichere Deutschkenntnisse", "C1", False),
    ("You speak German at C1 level.", "C1", False),
    ("Native-level German", "C2", False),
])
def test_german_requirement(text, level, optional):
    req = germany.find_german_requirement(text)
    assert req["level"] == level and req["optional"] == optional


def test_no_german_requirement():
    assert germany.find_german_requirement("We need strong Python skills.") is None


def test_hours_and_type():
    assert germany.detect_weekly_hours("Arbeitszeit: 15–20 Std./Woche") == 20
    assert germany.detect_weekly_hours("16-20 hours per week") == 20
    assert germany.detect_job_type("Werkstudent (m/w/d)") == "working student"
    assert germany.find_cv_german_level(SAMPLE_CV) == "A2"


def test_hours_flag_with_other_job():
    res = germany.run_checks(SAMPLE_JD_EN, SAMPLE_CV, {**PROFILE, "other_job_hours": 20})
    assert any("Hours check" in f["title"] and f["level"] == "warning" for f in res["flags"])


def test_language_cap():
    assert germany.language_score_cap("C1", "A2", False) == 1.5
    assert germany.language_score_cap("B2", "B1", False) == 3.5
    assert germany.language_score_cap("C1", "A2", True) is None


# ---------------------------------------------------------------- text utils
def test_keywords():
    norm = normalize(SAMPLE_CV)
    assert keyword_in_text("Power BI", norm)
    assert keyword_in_text("postgres", norm)          # alias of PostgreSQL
    assert keyword_in_text("machine-learning", normalize("Machine learning projects"))
    assert not keyword_in_text("Java", normalize("JavaScript developer"))
    cov, found, missing = keyword_coverage(["Python", "python", "Airflow"], SAMPLE_CV)
    assert cov == 0.5 and missing == ["Airflow"]


# ---------------------------------------------------------------- scoring
def test_good_match_english():
    a = analyse(FakeModels(), SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    assert a["overall"] > config.COVER_LETTER_THRESHOLD
    assert a["cover_letter_eligible"]
    assert a["germany"]["job_language"] == "en"
    assert a["semantic_available"]
    airflow = next(r for r in a["requirements"] if r["name"] == "Airflow")
    assert airflow["status"] != "met" and "downgraded" in airflow["note"]
    assert a["penalty"]["missing_must_have"] == 0
    assert set(a["categories"]) == set(config.DEFAULT_WEIGHTS)
    assert "Semantic similarity" in a["categories"]["skills"]["parts"]


def test_weak_match_german_gets_penalty_and_cap():
    a = analyse(FakeModels(), SAMPLE_CV, SAMPLE_JD_DE, PROFILE)
    assert a["germany"]["job_language"] == "de"
    assert a["germany"]["german_requirement"]["level"] == "C1"
    assert a["categories"]["languages"]["score"] <= 1.5
    # "Studium BWL": candidate has a B.Tech -> partial (degree exists, field differs), not missing
    bwl = next(r for r in a["requirements"] if r["name"] == "Studium BWL")
    assert bwl["status"] == "partial" and "field" in bwl["note"]
    assert a["penalty"]["missing_must_have"] == 3 and a["penalty"]["points"] == 0.75
    assert not a["cover_letter_eligible"]
    assert any(f["level"] == "error" for f in a["germany"]["flags"])


def test_works_without_embeddings():
    a = analyse(FakeModels(with_embeddings=False), SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    assert not a["semantic_available"]
    assert "Semantic similarity" not in a["categories"]["skills"]["parts"]
    assert 0 <= a["overall"] <= 5


# ---------------------------------------------------------------- outputs
def test_tailor_and_letter_roundtrip():
    llm = FakeModels()
    a = analyse(llm, SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    assert a["trace"][0]["node"] == "extract_requirements" and a["awaiting_letter_decision"]
    tips = tailor_cv(llm, SAMPLE_CV, a)
    assert tips and all(t["original_found_in_cv"] for t in tips)
    letter = generate_cover_letter(llm, SAMPLE_CV, SAMPLE_JD_EN, a, PROFILE, tone="modern", length="short")
    assert letter["language"] == "en" and len(letter["paragraphs"]) == 3
    back = text_to_letter(letter_to_text(letter), "en")
    assert back["subject"] == letter["subject"] and back["paragraphs"] == letter["paragraphs"]


def test_exports_open_and_contain_text():
    letter = {"language": "de", "subject": "Bewerbung als Werkstudent", "salutation": "Sehr geehrte Damen und Herren,",
              "paragraphs": ["Größe & Übung – 100 €.", "Zweiter Absatz."], "closing": "Mit freundlichen Grüßen"}
    rec = {"company": "FinanzPartner AG", "contact": "", "street": "Hauptstr. 5", "postal_city": "80331 München"}
    pdf = to_pdf(letter, PROFILE, rec, "Berlin", format_date("de"))
    text = PdfReader(io.BytesIO(pdf)).pages[0].extract_text()
    assert "Bewerbung als Werkstudent" in text and "Größe" in text and "FinanzPartner AG" in text
    docx_bytes = to_docx(letter, PROFILE, rec, "Berlin", "04.11.2026")
    from docx import Document
    doc_text = "\n".join(p.text for p in Document(io.BytesIO(docx_bytes)).paragraphs)
    assert "Mit freundlichen Grüßen" in doc_text and "Max Mustermann" in doc_text


def test_pdf_reader(tmp_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in SAMPLE_CV.splitlines():
        c.drawString(50, y, line)
        y -= 14
    c.save()
    text = extract_pdf_text(buf.getvalue())
    assert "Power BI" in text and "Mustermann" in text
    with pytest.raises(PDFTextError):
        extract_pdf_text(b"not a pdf")


# ---------------------------------------------------------------- storage
def test_storage(tmp_path):
    s = Storage(tmp_path / "t.db")
    assert s.get_profile()["is_student"] is True
    s.save_profile({**PROFILE})
    assert s.get_profile()["full_name"] == "Max Mustermann"
    s.save_cv("a.pdf", "old")
    s.save_cv("b.pdf", "new")
    assert s.get_active_cv()["filename"] == "b.pdf"
    a = analyse(FakeModels(), SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    app_id = s.add_application(a, SAMPLE_JD_EN, "https://example.com/job")
    s.update_application(app_id, status="applied", cover_letter_json={"subject": "x", "paragraphs": []})
    rows = s.list_applications()
    assert rows[0]["company"] == "DataCorp GmbH" and rows[0]["status"] == "applied" and rows[0]["has_letter"]
    assert s.get_application(app_id)["analysis"]["overall"] == a["overall"]
    s.delete_application(app_id)
    assert s.list_applications() == []
