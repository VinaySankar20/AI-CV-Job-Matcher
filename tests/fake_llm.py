"""A deterministic stand-in for Ollama so tests run anywhere."""
from __future__ import annotations

import hashlib
import re

from langchain_core.runnables import RunnableLambda

from matcher import schemas

SAMPLE_CV = """Max Mustermann
Musterstraße 1, 10115 Berlin · max@example.com
EDUCATION
M.Sc. Data Science, TU Berlin (since 10/2025)
B.Tech. Computer Science, University of Kerala (2019-2023)
EXPERIENCE
Data Analyst Intern, Acme GmbH (2023, 6 months)
- Built Python ETL scripts with pandas to clean sales data
- Created Power BI dashboards for the sales team
- Wrote SQL queries on PostgreSQL databases
PROJECTS
- Hand-gesture mouse controller with OpenCV and MediaPipe
- Trained a CNN image classifier in PyTorch (92% accuracy)
SKILLS
Python, SQL, pandas, PyTorch, OpenCV, Git, Docker, Power BI
LANGUAGES
English (C1), German – A2, Malayalam (native)
"""

SAMPLE_JD_EN = """Working Student Data Science (m/f/d) – Berlin
DataCorp GmbH is a fast-growing analytics company in Berlin.
Your tasks: build data pipelines in Python, create dashboards, support our ML team.
What you bring:
- Enrolled student in Computer Science, Data Science or similar
- Strong Python and SQL skills
- Experience with Power BI or Tableau
- First experience with machine learning (PyTorch or scikit-learn)
- Fluent English
Nice to have:
- Docker
- German is a plus
- Experience with Airflow
Working hours: 16–20 hours per week. Contact: Anna Schmidt.
"""

SAMPLE_JD_DE = """Werkstudent (m/w/d) Controlling – München
Die FinanzPartner AG sucht Verstärkung im Controlling.
Deine Aufgaben: Erstellung von Monatsberichten, Pflege von SAP-Daten, Abstimmung mit der Buchhaltung.
Dein Profil:
- Studium der Betriebswirtschaftslehre
- Sehr gute Kenntnisse in SAP FI/CO
- Verhandlungssichere Deutschkenntnisse in Wort und Schrift
- Erfahrung im Rechnungswesen
- Sehr gute Excel-Kenntnisse
Wünschenswert: Erfahrung mit DATEV.
Arbeitszeit: 20 Stunden pro Woche.
"""

_EXTRACT = {
    "DataCorp": {
        "job_title": "Working Student Data Science", "company": "DataCorp GmbH", "location": "Berlin",
        "contact_person": "Anna Schmidt", "min_years_experience": 0,
        "must_have": [
            {"name": "Enrolled in Computer Science or Data Science", "category": "education"},
            {"name": "Strong Python skills", "category": "skill"},
            {"name": "SQL", "category": "skill"},
            {"name": "Power BI or Tableau", "category": "skill"},
            {"name": "Machine learning with PyTorch", "category": "experience"},
            {"name": "Fluent English", "category": "language"},
        ],
        "nice_to_have": [
            {"name": "Docker", "category": "skill"},
            {"name": "Airflow", "category": "skill"},
            {"name": "German", "category": "language"},
        ],
        "keywords": ["Python", "SQL", "Power BI", "Tableau", "PyTorch", "scikit-learn", "Docker", "Airflow",
                     "Machine Learning", "Data Pipelines", "Dashboards"],
    },
    "FinanzPartner": {
        "job_title": "Werkstudent Controlling", "company": "FinanzPartner AG", "location": "München",
        "contact_person": "", "min_years_experience": 0,
        "must_have": [
            {"name": "Studium BWL", "category": "education"},
            {"name": "SAP FI/CO", "category": "skill"},
            {"name": "Verhandlungssicheres Deutsch", "category": "language"},
            {"name": "Erfahrung im Rechnungswesen", "category": "experience"},
            {"name": "Excel", "category": "skill"},
        ],
        "nice_to_have": [{"name": "DATEV", "category": "skill"}],
        "keywords": ["SAP", "FI/CO", "Excel", "DATEV", "Controlling", "Rechnungswesen"],
    },
}

_EVAL = {
    "DataCorp": {
        "requirements": [
            {"id": "M1", "status": "met", "evidence": "M.Sc. Data Science, TU Berlin (since 10/2025)"},
            {"id": "M2", "status": "met", "evidence": "Built Python ETL scripts with pandas to clean sales data"},
            {"id": "M3", "status": "met", "evidence": "Wrote SQL queries on PostgreSQL databases"},
            {"id": "M4", "status": "met", "evidence": "Created Power BI dashboards for the sales team"},
            {"id": "M5", "status": "met", "evidence": "Trained a CNN image classifier in PyTorch (92% accuracy)"},
            {"id": "M6", "status": "met", "evidence": "English (C1)"},
            {"id": "N1", "status": "met", "evidence": "Docker"},
            # hallucination: evidence not in CV -> should be downgraded
            {"id": "N2", "status": "met", "evidence": "Orchestrated Airflow DAGs in production"},
            {"id": "N3", "status": "partial", "evidence": "German – A2"},
        ],
        "scores": {
            "skills": {"score": 4.5, "reason": "Python, SQL, Power BI and PyTorch all present."},
            "experience": {"score": 3.8, "reason": "Internship plus relevant projects."},
            "education": {"score": 5, "reason": "Data Science Master's in progress."},
            "languages": {"score": 4.6, "reason": "English C1; German only a plus."},
        },
        "strengths": ["Python + SQL ETL experience", "Power BI dashboards", "PyTorch ML project"],
        "gaps": ["No Airflow", "Limited production experience"],
        "cv_tips": ["Quantify the dashboard impact", "Add scikit-learn if you have used it"],
    },
    "FinanzPartner": {
        "requirements": [
            {"id": "M1", "status": "missing", "evidence": ""},
            {"id": "M2", "status": "missing", "evidence": ""},
            {"id": "M3", "status": "missing", "evidence": ""},
            {"id": "M4", "status": "missing", "evidence": ""},
            {"id": "M5", "status": "partial", "evidence": "Power BI"},
            {"id": "N1", "status": "missing", "evidence": ""},
        ],
        "scores": {
            "skills": {"score": 1.0, "reason": "No SAP or accounting skills."},
            "experience": {"score": 1.0, "reason": "No controlling experience."},
            "education": {"score": 1.5, "reason": "Computer science, not business."},
            "languages": {"score": 3.0, "reason": "German A2 vs. business fluent."},
        },
        "strengths": ["Strong data skills"], "gaps": ["SAP", "German C1", "Accounting"],
        "cv_tips": ["This role is not a good fit"],
    },
}


def _which(text: str) -> str:
    return "FinanzPartner" if "FinanzPartner" in text else "DataCorp"


class FakeModels:
    """Same interface as matcher.llm.OllamaModels: structured() and embed()."""

    chat_model, embed_model, host = "fake-model", "fake-embed", "http://fake"

    def __init__(self, with_embeddings: bool = True):
        self.with_embeddings = with_embeddings
        self.calls: list[str] = []          # schema names, in call order
        self.prompts: dict[str, str] = {}   # last human prompt per schema name

    # ---- the canned answers (override in subclasses to simulate other model behaviour)
    def respond(self, schema, system: str, user: str) -> dict:
        if schema is schemas.JobExtraction:
            return _EXTRACT[_which(user)]
        if schema is schemas.Evaluation:
            return _EVAL[_which(user)]
        if schema is schemas.Recheck:
            return {"requirements": []}
        if schema is schemas.Tailoring:
            return {"suggestions": [
                {"original": "Created Power BI dashboards for the sales team",
                 "suggested": "Designed Power BI dashboards that gave the sales team a weekly KPI view",
                 "why": "Mirrors the 'dashboards' task.", "needs_confirmation": True},
                {"original": "Wrote SQL queries on PostgreSQL databases",
                 "suggested": "Wrote SQL queries on PostgreSQL to feed data pipelines",
                 "why": "Uses the ad's 'data pipelines' wording.", "needs_confirmation": False},
            ]}
        if schema is schemas.CoverLetter:
            german = "in German" in system
            return {
                "subject": "Bewerbung als Werkstudent" if german else "Application: Working Student Data Science",
                "salutation": "Sehr geehrte Damen und Herren," if german else "Dear Ms Schmidt,",
                "paragraphs": ["Para one about motivation – with umlauts äöüß and € sign.",
                               "Para two about Python, SQL and Power BI.",
                               "Para three about availability."],
                "closing": "Mit freundlichen Grüßen" if german else "Kind regards",
            }
        raise AssertionError(f"unexpected schema {schema}")

    def structured(self, schema, temperature: float = 0.0):
        def run(prompt_value):
            messages = prompt_value.to_messages()
            system, user = messages[0].content, messages[-1].content
            self.calls.append(schema.__name__)
            self.prompts[schema.__name__] = user
            return schema.model_validate(self.respond(schema, system, user))

        return RunnableLambda(run)

    def embed(self, texts):
        if not self.with_embeddings:
            return None
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str, dim: int = 256) -> list[float]:
        v = [0.0] * dim
        for w in re.findall(r"\w+", text.lower().split(":", 1)[-1]):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
        return v
