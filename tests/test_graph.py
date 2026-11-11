"""LangGraph behaviour: pause/resume, the fact-check rewrite loop, checkpoints, routing."""
from matcher import config, schemas
from matcher.graph import NODE_LABELS
from matcher.pipeline import MatcherPipeline, sqlite_checkpointer
from tests.fake_llm import SAMPLE_CV, SAMPLE_JD_DE, SAMPLE_JD_EN, FakeModels

PROFILE = {"full_name": "Max Mustermann", "german_level": "A2"}


class LyingModels(FakeModels):
    """Writes letters that claim Airflow experience (not in the CV) for the first `bad_drafts` drafts."""

    def __init__(self, bad_drafts: int):
        super().__init__()
        self.bad_drafts = bad_drafts
        self.drafts = 0

    def respond(self, schema, system, user):
        data = super().respond(schema, system, user)
        if schema is schemas.CoverLetter:
            self.drafts += 1
            if self.drafts <= self.bad_drafts:
                data = {**data, "paragraphs": ["I ran Airflow pipelines in production for years.",
                                               *(["I also know Tableau very well."] if self.drafts == 1 else [])]}
        return data


def nodes(trace):
    return [t["node"] for t in trace]


def test_pauses_before_letter_and_resumes():
    models = FakeModels()
    pipe = MatcherPipeline(models)
    a = pipe.analyse(SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    assert a["awaiting_letter_decision"] and "CoverLetter" not in models.calls      # paused, nothing written
    assert nodes(a["trace"]) == ["extract_requirements", "germany_checks", "evaluate_cv", "semantic_check", "score"]

    events = []
    out = pipe.write_letter(a, SAMPLE_CV, SAMPLE_JD_EN, PROFILE, tone="modern", length="short",
                            on_step=lambda node, phase, entry: events.append((node, phase)))
    assert out["letter"]["salutation"] == "Dear Ms Schmidt," and out["issues"] == [] and out["rewrites"] == 0
    assert nodes(out["trace"]) == ["approve_letter", "write_letter", "fact_check", "finalize"]
    assert ("write_letter", "start") in events and ("fact_check", "done") in events
    assert "about 180 words" in models.prompts["CoverLetter"]                        # length option reached the prompt


def test_low_score_ends_without_pause_but_letter_can_be_forced():
    models = FakeModels()
    pipe = MatcherPipeline(models)
    a = pipe.analyse(SAMPLE_CV, SAMPLE_JD_DE, PROFILE)
    assert not a["cover_letter_eligible"] and not a["awaiting_letter_decision"]
    out = pipe.write_letter(a, SAMPLE_CV, SAMPLE_JD_DE, PROFILE)                     # "generate anyway"
    assert out["letter"]["language"] == "de" and out["letter"]["closing"] == "Mit freundlichen Grüßen"


def test_fact_check_loop_rewrites_until_clean():
    models = LyingModels(bad_drafts=1)
    pipe = MatcherPipeline(models)
    a = pipe.analyse(SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    out = pipe.write_letter(a, SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    assert out["rewrites"] == 1 and out["issues"] == []
    assert nodes(out["trace"]) == ["approve_letter", "write_letter", "fact_check", "write_letter", "fact_check",
                                   "finalize"]
    assert "Fix these problems" in models.prompts["CoverLetter"] and "Airflow" in models.prompts["CoverLetter"]


def test_fact_check_loop_stops_at_limit_and_keeps_best_draft():
    models = LyingModels(bad_drafts=99)
    pipe = MatcherPipeline(models)
    a = pipe.analyse(SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    out = pipe.write_letter(a, SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    assert out["rewrites"] == config.MAX_LETTER_REWRITES and models.drafts == 1 + config.MAX_LETTER_REWRITES
    assert len(out["issues"]) == 1 and "Airflow" in out["issues"][0]                 # draft 1 had 2 issues
    assert "Tableau" not in " ".join(out["letter"]["paragraphs"])


def test_regenerate_and_old_analysis_without_thread():
    models = FakeModels()
    pipe = MatcherPipeline(models)
    a = pipe.analyse(SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    first = pipe.write_letter(a, SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    again = pipe.write_letter(a, SAMPLE_CV, SAMPLE_JD_EN, {**PROFILE, "availability": "from 01.12.2026"})
    assert again["thread_id"] == first["thread_id"] and models.calls.count("CoverLetter") == 2
    assert "from 01.12.2026" in models.prompts["CoverLetter"]                        # uses the current profile
    assert models.calls.count("Evaluation") == 1                                     # analysis not repeated

    old = {k: v for k, v in a.items() if k not in ("thread_id", "trace", "awaiting_letter_decision")}
    out = MatcherPipeline(FakeModels()).write_letter(old, SAMPLE_CV, SAMPLE_JD_EN, PROFILE)   # e.g. from the tracker
    assert out["letter"]["subject"] and out["thread_id"]


def test_decline_finishes_the_run():
    pipe = MatcherPipeline(FakeModels())
    a = pipe.analyse(SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    pipe.decline_letter(a["thread_id"])
    state = pipe.graph.get_state(pipe._cfg(a["thread_id"]))
    assert state.next == () and state.values["letter_request"] == {"write": False}


def test_sqlite_checkpoints_survive_a_restart(tmp_path):
    db = tmp_path / "checkpoints.db"
    a = MatcherPipeline(FakeModels(), sqlite_checkpointer(db)).analyse(SAMPLE_CV, SAMPLE_JD_EN, PROFILE)

    models = FakeModels()
    restarted = MatcherPipeline(models, sqlite_checkpointer(db))                     # "app restarted"
    state = restarted.graph.get_state(restarted._cfg(a["thread_id"]))
    assert state.next == ("approve_letter",) and state.values["analysis"]["overall"] == a["overall"]
    out = restarted.write_letter(a, SAMPLE_CV, SAMPLE_JD_EN, PROFILE)
    assert out["letter"]["paragraphs"] and models.calls == ["CoverLetter"]           # no re-analysis needed


def test_graph_diagram_lists_every_node():
    diagram = MatcherPipeline(FakeModels()).mermaid()
    assert all(node in diagram for node in NODE_LABELS)
