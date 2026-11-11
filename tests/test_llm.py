"""OllamaModels against a tiny fake Ollama HTTP server.

This checks the real LangChain → Ollama wire format (JSON schema, thinking flag, embeddings)
without needing Ollama installed.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from matcher import schemas
from matcher.chains import extract_chain
from matcher.llm import LLMError, OllamaModels, friendly_error


class FakeOllamaServer:
    def __init__(self, capabilities, reject_think=False):
        self.requests: list[dict] = []
        self.capabilities = capabilities
        self.reject_think = reject_think
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, lines):
                body = "".join(json.dumps(x) + "\n" for x in lines).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send(200, [{"models": [{"model": "qwen2.5:7b", "name": "qwen2.5:7b"}]}])

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                outer.requests.append({"path": self.path, **payload})
                if self.path == "/api/show":
                    return self._send(200, [{"capabilities": outer.capabilities, "details": {}, "model_info": {}}])
                if self.path == "/api/embed":
                    return self._send(200, [{"model": "e", "embeddings": [[1.0, 0.0]] * len(payload["input"])}])
                if self.path == "/api/chat":
                    if outer.reject_think and payload.get("think") is not None:
                        return self._send(400, [{"error": '"m" does not support thinking'}])
                    answer = {"job_title": "Firmware Developer", "company": "IBM", "location": "", "contact_person": "",
                              "must_have": [{"name": "C++", "category": "skill"}], "nice_to_have": [],
                              "keywords": ["C++"], "min_years_experience": 2}
                    msg = {"model": "m", "created_at": "2026-01-01T00:00:00Z", "done": False,
                           "message": {"role": "assistant", "content": json.dumps(answer)}}
                    done = {"model": "m", "created_at": "2026-01-01T00:00:00Z", "done": True, "done_reason": "stop",
                            "message": {"role": "assistant", "content": ""}, "eval_count": 1, "prompt_eval_count": 1}
                    return self._send(200, [msg, done])
                self._send(404, [{"error": "not found"}])

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def chats(self):
        return [r for r in self.requests if r["path"] == "/api/chat"]

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server(request, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")   # never send local test traffic to a proxy
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    srv = FakeOllamaServer(**getattr(request, "param", {"capabilities": ["completion"]}))
    yield srv
    srv.close()


def test_structured_output_sends_schema_and_parses(server):
    models = OllamaModels(host=server.url, chat_model="qwen2.5:7b")
    job = extract_chain(models).invoke({"jd": "Firmware Developer at IBM, C++ required"})
    assert isinstance(job, schemas.JobExtraction) and job.must_have[0].name == "C++"
    chat = server.chats()[0]
    assert chat["format"]["properties"]["must_have"]["type"] == "array"     # JSON schema enforced by Ollama
    assert chat["options"]["temperature"] == 0.0 and chat["options"]["seed"] == 42
    assert chat.get("think") is None                                        # not a thinking model
    assert "Firmware Developer at IBM" in chat["messages"][-1]["content"]
    assert models.health()["chat_model"] is True


@pytest.mark.parametrize("server", [{"capabilities": ["completion", "thinking"]}], indirect=True)
def test_thinking_model_gets_thinking_switched_off(server):
    models = OllamaModels(host=server.url, chat_model="qwen3.5:4b")
    assert models.is_thinking_model()
    extract_chain(models).invoke({"jd": "x"})
    assert server.chats()[0]["think"] is False


@pytest.mark.parametrize("server", [{"capabilities": ["completion", "thinking"], "reject_think": True}],
                         indirect=True)
def test_falls_back_when_model_rejects_the_thinking_flag(server):
    models = OllamaModels(host=server.url, chat_model="odd-model")
    job = extract_chain(models).invoke({"jd": "x"})
    assert job.company == "IBM"
    assert [c.get("think") for c in server.chats()] == [False, None]


def test_embeddings_and_unreachable_server(server):
    models = OllamaModels(host=server.url)
    assert models.embed(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    down = OllamaModels(host="http://127.0.0.1:9")
    assert down.health()["ok"] is False
    assert down.embed(["a"]) is None
    with pytest.raises(Exception) as err:
        extract_chain(down).invoke({"jd": "x"})
    assert "Is the Ollama app running" in str(friendly_error(err.value, down))


def test_friendly_errors():
    assert "took longer" in str(friendly_error(TimeoutError("timed out")))
    assert "wrong format" in str(friendly_error(ValueError("1 validation error for Evaluation")))
    assert isinstance(friendly_error(LLMError("x")), LLMError)
