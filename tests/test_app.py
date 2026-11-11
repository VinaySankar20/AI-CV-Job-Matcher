"""Smoke test of the Streamlit UI with the fake LLM (no Ollama needed)."""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from matcher import config, llm as llm_module
from matcher.storage import Storage
from tests.fake_llm import SAMPLE_CV, SAMPLE_JD_EN, FakeModels

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeOllama(FakeModels):
    provider, embed_model = "ollama", "nomic-embed-text"

    def __init__(self, chat_model=None, **kwargs):
        super().__init__()
        self.chat_model = chat_model or "qwen2.5:7b"
        self.label = f"Ollama · {self.chat_model}"

    def health(self):
        return {"ok": True, "error": "", "models": ["qwen2.5:7b", "mistral:latest", "nomic-embed-text:latest"],
                "chat_model": True, "embed_model": True}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(config, "CHECKPOINT_DB", tmp_path / "checkpoints.db")
    monkeypatch.setattr(llm_module, "OllamaModels", FakeOllama)
    st.cache_resource.clear()
    s = Storage(tmp_path / "app.db")
    s.save_profile({"full_name": "Max Mustermann", "german_level": "A2", "place": "Berlin"})
    s.save_cv("cv.pdf", SAMPLE_CV)
    return s


def test_full_flow(app):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Using saved CV" in s.value for s in at.success)

    at.text_area[0].set_value(SAMPLE_JD_EN)
    next(b for b in at.button if "Analyse match" in b.label).click().run()
    assert not at.exception
    assert any("/ 5" in m.value for m in at.metric)
    assert len(app.list_applications()) == 1

    assert any("Pipeline trace" in e.label for e in at.expander)
    assert any("paused here" in c.value for c in at.caption)          # human-in-the-loop pause

    next(b for b in at.button if "Write cover letter" in b.label).click().run()
    assert not at.exception
    editor = at.text_area(key="letter_editor")
    assert "Dear Ms Schmidt" in editor.value
    assert any("Fact check passed" in s.value for s in at.success)
    assert (config.CHECKPOINT_DB).exists()                             # graph run saved locally

    next(b for b in at.button if "Suggest CV rewrites" in b.label).click().run()
    assert not at.exception
    assert any("Before:" in m.value for m in at.markdown)
    assert app.get_application(app.list_applications()[0]["id"])["cover_letter"]["subject"]


@pytest.mark.parametrize("page", ["tracker", "profile"])
def test_other_pages_render(app, page, monkeypatch):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area[0].set_value(SAMPLE_JD_EN)
    next(b for b in at.button if "Analyse match" in b.label).click().run()
    monkeypatch.setenv("CVM_START_PAGE", page)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, at.exception
    if page == "tracker":
        assert any("Jobs analysed" in m.label for m in at.metric)
    else:
        assert at.text_input[0].value == "Max Mustermann"


def sidebar_select(at, label):
    return next(w for w in at.sidebar.selectbox if w.label == label)


def test_model_picker_lists_local_models_and_switches(app):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert sidebar_select(at, "Model").options[:2] == ["qwen2.5:7b", "mistral:latest"]   # embed model hidden
    sidebar_select(at, "Model").select("mistral:latest").run()
    assert not at.exception
    assert app.get_setting("model_choice") == {"provider": "ollama", "model": "mistral:latest"}
    at.text_area[0].set_value(SAMPLE_JD_EN)
    next(b for b in at.button if "Analyse match" in b.label).click().run()
    assert any("Scored by: Ollama · mistral:latest" in c.value for c in at.caption)


def test_cloud_needs_key_then_consent(app, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    sidebar_select(at, "Provider").select("openai").run()
    assert any("No API key for OpenAI" in e.value for e in at.sidebar.error)
    assert any("Still using" in c.value and "Ollama" in c.value for c in at.sidebar.caption)
    assert app.get_setting("model_choice") is None                       # nothing switched

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    at.run()
    assert any("cloud service" in w.value for w in at.sidebar.warning)
    assert app.get_setting("model_choice") is None                       # still waiting for consent
    next(b for b in at.sidebar.button if "I understand" in b.label).click().run()
    assert not at.exception
    assert app.get_setting("cloud_consent") == ["openai"]
    assert app.get_setting("model_choice") == {"provider": "openai", "model": "gpt-6.1-terra"}
    assert any("sent to OpenAI" in c.value for c in at.sidebar.caption)
