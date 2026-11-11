"""Pydantic models for everything the LLM returns.

LangChain's `with_structured_output` turns each model into a JSON schema that Ollama
enforces while generating, then validates the answer back into the model.

Every field is required on purpose: a required field is part of the enforced schema, so a
small model cannot simply leave out (for example) the list of requirement verdicts.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["skill", "experience", "education", "language", "other"]
Status = Literal["met", "partial", "missing"]


# ---------------------------------------------------------------- 1. job ad
class Requirement(BaseModel):
    name: str = Field(description="Short requirement, 2-8 words, in the language of the ad")
    category: Category


class JobExtraction(BaseModel):
    job_title: str
    company: str
    location: str
    contact_person: str
    must_have: list[Requirement]
    nice_to_have: list[Requirement]
    keywords: list[str]
    min_years_experience: float


# ---------------------------------------------------------------- 2. CV vs requirements
class Verdict(BaseModel):
    id: str
    status: Status
    evidence: str = Field(description="Exact words copied from the CV, or empty")


class CategoryScore(BaseModel):
    score: float = Field(description="0 to 5")
    reason: str


class Scores(BaseModel):
    skills: CategoryScore
    experience: CategoryScore
    education: CategoryScore
    languages: CategoryScore


class Evaluation(BaseModel):
    requirements: list[Verdict]
    scores: Scores
    strengths: list[str]
    gaps: list[str]
    cv_tips: list[str]


class Recheck(BaseModel):
    requirements: list[Verdict]


# ---------------------------------------------------------------- 3. CV tailoring
class Suggestion(BaseModel):
    original: str
    suggested: str
    why: str
    needs_confirmation: bool


class Tailoring(BaseModel):
    suggestions: list[Suggestion]


# ---------------------------------------------------------------- 4. cover letter
class CoverLetter(BaseModel):
    subject: str
    salutation: str
    paragraphs: list[str]
    closing: str
