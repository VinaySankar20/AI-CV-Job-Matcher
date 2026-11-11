"""Run the LangGraph pipeline from the app: analyse a job, then (after the user agrees) write the letter.

Each analysis is one graph "thread". The checkpointer saves every step, so the run pauses at the
human-in-the-loop node and resumes later — even after the app was restarted.
"""
from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from . import config
from .graph import NODE_LABELS, build_graph
from .llm import Models, friendly_error

# called as on_step(node_name, "start" | "done", trace_entry_or_None)
StepCallback = Callable[[str, str, dict | None], None]

_ANALYSIS_EXTRAS = ("thread_id", "trace", "awaiting_letter_decision", "model")


def sqlite_checkpointer(path=None):
    """Checkpointer that stores graph runs in a local SQLite file (stays on this computer)."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = path or config.CHECKPOINT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


class MatcherPipeline:
    def __init__(self, models: Models, checkpointer=None):
        self.models = models
        self.graph = build_graph(models, checkpointer or InMemorySaver())

    @staticmethod
    def _cfg(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _run(self, graph_input: Any, thread_id: str, on_step: StepCallback | None) -> None:
        """Run the graph until it finishes or pauses, reporting each node as it starts and ends."""
        try:
            for mode, event in self.graph.stream(graph_input, self._cfg(thread_id),
                                                 stream_mode=["tasks", "updates"]):
                if not on_step:
                    continue
                if mode == "tasks" and "input" in event:          # a node is starting
                    on_step(event["name"], "start", None)
                elif mode == "updates":
                    for node, update in event.items():
                        if node in NODE_LABELS and isinstance(update, dict):
                            on_step(node, "done", (update.get("trace") or [None])[-1])
        except Exception as exc:
            raise friendly_error(exc, self.models) from exc

    def _values(self, thread_id: str):
        return self.graph.get_state(self._cfg(thread_id))

    # ------------------------------------------------------------------ phase 1
    def analyse(self, cv_text: str, jd_text: str, profile: dict[str, Any] | None = None,
                weights: dict[str, float] | None = None, on_step: StepCallback | None = None,
                thread_id: str | None = None) -> dict[str, Any]:
        """Score the CV against the job. Stops before the cover letter (waits for the user)."""
        thread_id = thread_id or uuid.uuid4().hex
        self._run({"cv_text": cv_text, "jd_text": jd_text, "profile": profile or {},
                   "weights": weights or config.DEFAULT_WEIGHTS}, thread_id, on_step)
        state = self._values(thread_id)
        analysis = dict(state.values["analysis"])
        analysis["thread_id"] = thread_id
        analysis["trace"] = list(state.values.get("trace", []))
        analysis["awaiting_letter_decision"] = "approve_letter" in state.next
        analysis["model"] = getattr(self.models, "label", "")
        return analysis

    # ------------------------------------------------------------------ phase 2
    def write_letter(self, analysis: dict[str, Any], cv_text: str, jd_text: str, profile: dict[str, Any],
                     tone: str = "formal", length: str = "standard", extra: str = "",
                     on_step: StepCallback | None = None) -> dict[str, Any]:
        """Write the cover letter with the automatic fact-check-and-rewrite loop.

        Works for a run that is paused at the approval step, for a finished run (regenerate) and
        for an old analysis loaded from the tracker: the letter phase is (re)started from the
        point right after scoring, using the current profile.
        """
        thread_id = analysis.get("thread_id") or uuid.uuid4().hex
        core = {k: v for k, v in analysis.items() if k not in _ANALYSIS_EXTRAS}
        before = len(self._values(thread_id).values.get("trace", []))
        try:
            # act as if the "score" node had just produced this analysis, with the letter forced
            self.graph.update_state(self._cfg(thread_id),
                                    {"cv_text": cv_text, "jd_text": jd_text, "profile": profile or {},
                                     "analysis": core, "force_letter": True}, as_node="score")
        except Exception as exc:
            raise friendly_error(exc, self.models) from exc
        self._run(None, thread_id, None)                                    # runs up to the approval pause
        self._run(Command(resume={"write": True, "tone": tone, "length": length, "extra": extra}),
                  thread_id, on_step)
        values = self._values(thread_id).values
        return {"letter": values.get("letter"), "issues": values.get("issues") or [],
                "model": getattr(self.models, "label", ""),
                "rewrites": values.get("rewrite_count", 0), "thread_id": thread_id,
                "trace": list(values.get("trace", []))[before:]}

    def decline_letter(self, thread_id: str) -> None:
        """Tell a paused run that no letter is wanted, so it finishes cleanly."""
        if "approve_letter" in self._values(thread_id).next:
            self._run(Command(resume={"write": False}), thread_id, None)

    def mermaid(self) -> str:
        """The graph as a Mermaid diagram (for the README / the app)."""
        return self.graph.get_graph().draw_mermaid()


def analyse(models: Models, cv_text: str, jd_text: str, profile: dict[str, Any] | None = None,
            weights: dict[str, float] | None = None) -> dict[str, Any]:
    """One-off analysis without saving checkpoints (used by scripts and tests)."""
    return MatcherPipeline(models).analyse(cv_text, jd_text, profile, weights)
