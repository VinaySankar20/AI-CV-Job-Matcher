"""OllamaLLM behaviour with thinking / non-thinking models (Ollama client mocked)."""
from types import SimpleNamespace

import ollama
import pytest

from matcher.llm import LLMError, OllamaLLM


class FakeClient:
    def __init__(self, supports_think=True, content='{"ok": true}', thinking=None):
        self.supports_think, self.content, self.thinking = supports_think, content, thinking
        self.calls = []

    def chat(self, **kw):
        self.calls.append(kw)
        if "think" in kw and not self.supports_think:
            raise ollama.ResponseError('"mistral" does not support thinking', 400)
        return SimpleNamespace(message=SimpleNamespace(content=self.content, thinking=self.thinking))


def make(client):
    llm = OllamaLLM()
    llm.client = client
    return llm


def test_thinking_model_gets_think_false():
    c = FakeClient()
    assert make(c).chat_json("s", "u") == {"ok": True}
    assert c.calls[0]["think"] is False


def test_non_thinking_model_retries_without_flag_once():
    c = FakeClient(supports_think=False)
    llm = make(c)
    assert llm.chat_json("s", "u") == {"ok": True}
    assert llm.chat_json("s", "u") == {"ok": True}
    assert "think" in c.calls[0] and "think" not in c.calls[1] and "think" not in c.calls[2]


def test_answer_only_in_thinking_field():
    assert make(FakeClient(content="", thinking='{"ok": true}')).chat_json("s", "u") == {"ok": True}


def test_empty_answer_raises_clear_error():
    with pytest.raises(LLMError, match="empty answer"):
        make(FakeClient(content="")).chat_json("s", "u")
