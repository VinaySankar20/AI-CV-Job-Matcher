"""LangChain chains: prompt template | model with structured (Pydantic) output.

Each function returns a runnable you call with `.invoke({...placeholders...})`.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from . import prompts, schemas
from .llm import Models


def _chain(system: str, user: str, schema, models: Models, temperature: float, name: str) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
    return (prompt | models.structured(schema, temperature)).with_config(run_name=name)


def extract_chain(models: Models) -> Runnable:
    """job description -> JobExtraction"""
    return _chain(prompts.EXTRACT_SYSTEM, prompts.EXTRACT_USER, schemas.JobExtraction, models, 0.0, "extract")


def evaluate_chain(models: Models) -> Runnable:
    """CV + requirements -> Evaluation (verdict per requirement + category scores)"""
    return _chain(prompts.EVALUATE_SYSTEM, prompts.EVALUATE_USER, schemas.Evaluation, models, 0.0, "evaluate")


def recheck_chain(models: Models) -> Runnable:
    """CV + the requirements the model skipped -> Recheck"""
    return _chain(prompts.EVALUATE_SYSTEM, prompts.RECHECK_USER, schemas.Recheck, models, 0.0, "recheck")


def tailor_chain(models: Models) -> Runnable:
    """CV + job keywords -> Tailoring (rewrites of existing CV lines)"""
    return _chain(prompts.TAILOR_SYSTEM, prompts.TAILOR_USER, schemas.Tailoring, models, 0.3, "tailor")


def cover_letter_chain(models: Models) -> Runnable:
    """CV + job + verified matches + fact rules -> CoverLetter"""
    return _chain(prompts.COVER_SYSTEM, prompts.COVER_USER, schemas.CoverLetter, models, 0.5, "cover_letter")
