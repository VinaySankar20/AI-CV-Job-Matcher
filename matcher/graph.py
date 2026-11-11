"""The matching pipeline as a LangGraph state machine.

    extract_requirements → germany_checks → evaluate_cv
            │ model skipped requirements?  ── yes → recheck (max 1×) ─┐
            ▼ no  ◄────────────────────────────────────────────────────┘
    semantic_check → score
            │ score > 3.5 (or letter forced)?  ── no → END
            ▼ yes
    approve_letter   ← pauses here (human-in-the-loop) until the user decides
            │ user says "write"?  ── no → END
            ▼ yes
    write_letter → fact_check
            │ unsupported claims found?  ── yes → write_letter again (max 2 rewrites)
            ▼ no
    finalize → END   (keeps the draft with the fewest fact-check issues)

LLM nodes use the chains in chains.py; all other nodes are plain Python rules.
Every node appends one entry to `trace` (node, seconds, detail) — a local log of the run.
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import chains, config, germany, scorer
from .cover_letter import clean_letter, fact_check, letter_inputs
from .llm import Models


class MatchState(TypedDict, total=False):
    # ---- inputs
    cv_text: str
    jd_text: str
    profile: dict[str, Any]
    weights: dict[str, float]
    # ---- analysis phase
    job: dict[str, Any]                    # cleaned job extraction incl. requirements
    germany: dict[str, Any]                # rule-based German-market checks
    evaluation: dict[str, Any]             # model's scores, strengths, gaps, tips
    verdicts: dict[str, dict[str, Any]]    # requirement id -> {status, evidence}
    recheck_count: int
    sims: list[float] | None               # embedding similarity per requirement
    analysis: dict[str, Any]               # final scored result shown in the app
    # ---- cover letter phase
    force_letter: bool                     # user asked for a letter although score <= 3.5
    letter_request: dict[str, Any]         # the user's decision: write?, tone, length, extra
    letter: dict[str, Any] | None
    issues: list[str]                      # fact-check findings for the current draft
    rewrite_count: int
    best_letter: dict[str, Any] | None
    best_issues: list[str] | None
    # ---- local log, appended to by every node
    trace: Annotated[list[dict[str, Any]], operator.add]


NODE_LABELS = {
    "extract_requirements": "Reading the job description",
    "germany_checks": "Running Germany-specific checks",
    "evaluate_cv": "Comparing your CV with each requirement",
    "recheck": "Re-checking requirements the AI skipped",
    "semantic_check": "Checking semantic similarity (embeddings)",
    "score": "Calculating the score",
    "approve_letter": "Waiting for your decision on the cover letter",
    "write_letter": "Writing the cover letter",
    "fact_check": "Fact-checking the letter against your CV",
    "finalize": "Choosing the best draft",
}


def _traced(name: str, fn: Callable[[MatchState], dict[str, Any]]) -> Callable[[MatchState], dict[str, Any]]:
    """Wrap a node so it logs its duration and a one-line detail into state['trace']."""

    def node(state: MatchState) -> dict[str, Any]:
        start = time.perf_counter()
        update = fn(state)  # an interrupt() inside fn propagates; the node re-runs on resume
        detail = update.pop("_detail", "")
        update["trace"] = [{"node": name, "seconds": round(time.perf_counter() - start, 2), "detail": detail}]
        return update

    node.__name__ = name
    return node


def build_graph(models: Models, checkpointer=None):
    """Compile the graph. `checkpointer` saves every step so runs can pause and resume."""
    extract = chains.extract_chain(models)
    evaluate = chains.evaluate_chain(models)
    recheck_chain = chains.recheck_chain(models)
    cover = chains.cover_letter_chain(models)

    # ------------------------------------------------------------------ analysis nodes
    def extract_requirements(state: MatchState) -> dict[str, Any]:
        result = extract.invoke({"jd": state["jd_text"][:12000]})
        job = scorer.normalize_job(result.model_dump())
        must = sum(r["type"] == "must" for r in job["requirements"])
        return {"job": job, "_detail": f"{must} must-haves, {len(job['requirements']) - must} nice-to-haves, "
                                       f"{len(job['keywords'])} keywords"}

    def germany_checks(state: MatchState) -> dict[str, Any]:
        de = germany.run_checks(state["jd_text"], state["cv_text"], state.get("profile") or {})
        req = de["german_requirement"]
        return {"germany": de, "_detail": f"ad language {de['job_language']}, German required: "
                                          f"{req['level'] if req else 'not stated'}, type: {de['job_type']}"}

    def evaluate_cv(state: MatchState) -> dict[str, Any]:
        job = state["job"]
        result = evaluate.invoke({
            "years": job["min_years_experience"], "title": job["job_title"] or "the role",
            "company": job["company"] or "the company",
            "requirements": scorer.requirement_lines(job["requirements"]), "cv": state["cv_text"][:12000],
        }).model_dump()
        verdicts = scorer.collect_verdicts(result.pop("requirements"))
        skipped = scorer.skipped_requirements(job["requirements"], verdicts)
        return {"evaluation": result, "verdicts": verdicts, "recheck_count": 0,
                "_detail": f"{len(verdicts)} verdicts, {len(skipped)} skipped"}

    def recheck(state: MatchState) -> dict[str, Any]:
        skipped = scorer.skipped_requirements(state["job"]["requirements"], state["verdicts"])
        verdicts = dict(state["verdicts"])
        try:
            result = recheck_chain.invoke({"requirements": scorer.requirement_lines(skipped),
                                           "cv": state["cv_text"][:12000]})
            new = scorer.collect_verdicts(result.model_dump()["requirements"])
            wanted = {r["id"] for r in skipped}
            verdicts.update({k: v for k, v in new.items() if k in wanted})
            detail = f"asked again for {len(skipped)}, got {len(set(new) & wanted)}"
        except Exception as exc:  # not fatal: the keyword/similarity rules decide instead
            detail = f"re-check failed ({type(exc).__name__}); rules will decide"
        return {"verdicts": verdicts, "recheck_count": state.get("recheck_count", 0) + 1, "_detail": detail}

    def semantic_check(state: MatchState) -> dict[str, Any]:
        sims = scorer.semantic_similarities(models, state["cv_text"], state["job"]["requirements"])
        return {"sims": sims, "_detail": "embeddings unavailable — skipped" if sims is None
                else f"similarity {min(sims):.2f}–{max(sims):.2f}"}

    def score(state: MatchState) -> dict[str, Any]:
        analysis = scorer.build_analysis(state["job"], state["verdicts"], state["evaluation"], state.get("sims"),
                                         state["germany"], state["cv_text"], state.get("weights"))
        return {"analysis": analysis, "_detail": f"overall {analysis['overall']:.2f} / 5, "
                                                 f"{analysis['penalty']['missing_must_have']} must-haves missing"}

    # ------------------------------------------------------------------ cover letter nodes
    def approve_letter(state: MatchState) -> dict[str, Any]:
        # Human-in-the-loop: the graph stops here and is saved by the checkpointer.
        # It continues when the app resumes it with the user's decision.
        decision = interrupt({"question": "write_cover_letter", "overall": state["analysis"]["overall"],
                              "eligible": state["analysis"]["cover_letter_eligible"]})
        decision = decision if isinstance(decision, dict) else {"write": bool(decision)}
        return {"letter_request": decision, "force_letter": False, "letter": None, "issues": [],
                "rewrite_count": 0, "best_letter": None, "best_issues": None,
                "_detail": "write" if decision.get("write") else "declined"}

    def write_letter(state: MatchState) -> dict[str, Any]:
        req = state.get("letter_request") or {}
        tone, length = req.get("tone", "formal"), req.get("length", "standard")
        extra = str(req.get("extra") or "")
        is_rewrite = state.get("letter") is not None
        if is_rewrite and state.get("issues"):
            extra += (" Fix these problems from the previous draft: "
                      + " ".join(i.replace("**", "") for i in state["issues"]))
        profile = state.get("profile") or {}
        result = cover.invoke(letter_inputs(state["cv_text"], state["jd_text"], state["analysis"], profile,
                                            tone, length, extra))
        letter = clean_letter(result.model_dump(), state["analysis"], profile, tone)
        count = state.get("rewrite_count", 0) + (1 if is_rewrite else 0)
        words = len(" ".join(letter["paragraphs"]).split())
        return {"letter": letter, "rewrite_count": count,
                "_detail": (f"rewrite {count}" if is_rewrite else "first draft") + f", {words} words"}

    def check_facts(state: MatchState) -> dict[str, Any]:
        issues = fact_check(state["letter"], state["cv_text"], state["analysis"], state.get("profile") or {})
        best_issues = state.get("best_issues")
        update: dict[str, Any] = {"issues": issues}
        if best_issues is None or len(issues) < len(best_issues):
            update.update(best_letter=state["letter"], best_issues=issues)
        update["_detail"] = "passed" if not issues else f"{len(issues)} issue(s) found"
        return update

    def finalize(state: MatchState) -> dict[str, Any]:
        issues = state.get("best_issues") or []
        return {"letter": state.get("best_letter") or state.get("letter"), "issues": issues,
                "_detail": f"{state.get('rewrite_count', 0)} rewrite(s), {len(issues)} issue(s) left"}

    # ------------------------------------------------------------------ routing
    def after_evaluate(state: MatchState) -> str:
        skipped = scorer.skipped_requirements(state["job"]["requirements"], state["verdicts"])
        return "recheck" if skipped and state.get("recheck_count", 0) < config.MAX_RECHECKS else "semantic_check"

    def after_score(state: MatchState) -> str:
        return "approve_letter" if state["analysis"]["cover_letter_eligible"] or state.get("force_letter") else END

    def after_approval(state: MatchState) -> str:
        return "write_letter" if (state.get("letter_request") or {}).get("write") else END

    def after_fact_check(state: MatchState) -> str:
        retry = state.get("issues") and state.get("rewrite_count", 0) < config.MAX_LETTER_REWRITES
        return "write_letter" if retry else "finalize"

    # ------------------------------------------------------------------ wiring
    g = StateGraph(MatchState)
    for name, fn in [("extract_requirements", extract_requirements), ("germany_checks", germany_checks),
                     ("evaluate_cv", evaluate_cv), ("recheck", recheck), ("semantic_check", semantic_check),
                     ("score", score), ("approve_letter", approve_letter), ("write_letter", write_letter),
                     ("fact_check", check_facts), ("finalize", finalize)]:
        g.add_node(name, _traced(name, fn))

    g.add_edge(START, "extract_requirements")
    g.add_edge("extract_requirements", "germany_checks")
    g.add_edge("germany_checks", "evaluate_cv")
    g.add_conditional_edges("evaluate_cv", after_evaluate, ["recheck", "semantic_check"])
    g.add_conditional_edges("recheck", after_evaluate, ["recheck", "semantic_check"])
    g.add_edge("semantic_check", "score")
    g.add_conditional_edges("score", after_score, ["approve_letter", END])
    g.add_conditional_edges("approve_letter", after_approval, ["write_letter", END])
    g.add_edge("write_letter", "fact_check")
    g.add_conditional_edges("fact_check", after_fact_check, ["write_letter", "finalize"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
