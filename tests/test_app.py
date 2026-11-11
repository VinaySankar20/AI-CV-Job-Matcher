"""Smoke test of the Streamlit UI with the fake LLM (no Ollama needed)."""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from matcher import config, llm as llm_module
from matcher.storage import Storage
from tests.fake_llm import SAMPLE_CV, SAMPLE_JD_EN, FakeLLM

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeOllama(FakeLLM):
    chat_model, embed_model = "qwen2.5:7b", "nomic-embed-text"

    def health(self):
        return {"ok": True, "error": "", "models": [], "chat_model": True, "embed_model": True}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(llm_module, "OllamaLLM", FakeOllama)
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

    next(b for b in at.button if "Generate cover letter" in b.label).click().run()
    assert not at.exception
    editor = at.text_area(key="letter_editor")
    assert "Dear Ms Schmidt" in editor.value

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
