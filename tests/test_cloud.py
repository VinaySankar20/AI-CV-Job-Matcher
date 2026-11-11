"""CloudModels against fake OpenAI-style and Anthropic-style HTTP servers (no keys, no internet)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from matcher import config, schemas
from matcher.chains import extract_chain
from matcher.llm import CloudModels, LLMError, create_models, friendly_error

ANSWER = {"job_title": "Firmware Developer", "company": "IBM", "location": "", "contact_person": "",
          "must_have": [{"name": "C++", "category": "skill"}], "nice_to_have": [],
          "keywords": ["C++"], "min_years_experience": 2}


class NoEmbeddings:
    embed_model = "none"

    def embed(self, texts):
        return None


class FakeCloud:
    """Answers POST /chat/completions (OpenAI / OpenRouter) and POST /v1/messages (Anthropic)."""

    def __init__(self, reject=None, status=200):
        self.requests, self.reject, self.status = [], reject or (lambda body: None), status
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization")
                                       or self.headers.get("x-api-key"), **body})
                problem = outer.reject(body)
                if outer.status != 200 or problem:
                    code, payload = (outer.status if outer.status != 200 else 400), \
                        {"error": {"message": problem or "Incorrect API key provided", "type": "invalid_request_error"}}
                elif self.path.endswith("/messages"):
                    tool = body["tools"][0]["name"]
                    code, payload = 200, {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                                          "content": [{"type": "tool_use", "id": "tu_1", "name": tool, "input": ANSWER}],
                                          "stop_reason": "tool_use", "stop_sequence": None,
                                          "usage": {"input_tokens": 1, "output_tokens": 1}}
                else:
                    if body.get("tools"):
                        name = body["tools"][0]["function"]["name"]
                        message = {"role": "assistant", "content": None, "tool_calls": [
                            {"id": "call_1", "type": "function",
                             "function": {"name": name, "arguments": json.dumps(ANSWER)}}]}
                    else:
                        message = {"role": "assistant", "content": json.dumps(ANSWER)}
                    code, payload = 200, {"id": "c1", "object": "chat.completion", "created": 1, "model": body["model"],
                                          "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def cloud(request, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    srv = FakeCloud(**getattr(request, "param", {}))
    monkeypatch.setenv("OPENAI_BASE_URL", srv.url + "/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", srv.url)
    monkeypatch.setitem(config.PROVIDERS["openrouter"], "base_url", srv.url + "/api/v1")
    yield srv
    srv.httpd.shutdown()


def run(provider, model, key="sk-test"):
    models = CloudModels(provider, model, api_key=key, embedder=NoEmbeddings())
    return models, extract_chain(models).invoke({"jd": "Firmware Developer at IBM, C++ required"})


def test_openai_uses_strict_json_schema(cloud):
    models, job = run("openai", "gpt-6.1-terra")
    assert isinstance(job, schemas.JobExtraction) and job.must_have[0].name == "C++"
    req = cloud.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer sk-test"
    assert req["model"] == "gpt-6.1-terra" and req["temperature"] == 0.0
    fmt = req["response_format"]["json_schema"]
    assert fmt["strict"] is True and fmt["schema"]["additionalProperties"] is False
    assert models.label == "OpenAI · gpt-6.1-terra" and models.embed(["x"]) is None


def test_openrouter_goes_to_its_own_url(cloud):
    _, job = run("openrouter", "anthropic/claude-haiku-5.5", key="or-key")
    req = cloud.requests[0]
    assert req["path"] == "/api/v1/chat/completions" and req["auth"] == "Bearer or-key"
    assert req["model"] == "anthropic/claude-haiku-5.5" and job.company == "IBM"


@pytest.mark.parametrize("cloud", [{"reject": lambda b: "response_format json_schema is not supported"
                                    if "response_format" in b else None}], indirect=True)
def test_falls_back_to_tool_calling_without_json_schema_support(cloud):
    _, job = run("openrouter", "some/small-model")
    assert job.company == "IBM" and len(cloud.requests) == 2
    assert "response_format" in cloud.requests[0] and cloud.requests[1]["tools"]


@pytest.mark.parametrize("cloud", [{"reject": lambda b: "Unsupported value: 'temperature'"
                                    if "temperature" in b else None}], indirect=True)
def test_falls_back_when_model_rejects_temperature(cloud):
    _, job = run("openai", "reasoning-model")
    assert job.company == "IBM" and "temperature" not in cloud.requests[-1]


def test_anthropic_uses_tool_calling(cloud):
    _, job = run("anthropic", "claude-haiku-5-5", key="ant-key")
    req = cloud.requests[0]
    assert req["path"] == "/v1/messages" and "ant-key" in req["auth"]
    assert req["model"] == "claude-haiku-5-5" and req["max_tokens"] == config.CLOUD_MAX_TOKENS
    assert req["tools"][0]["name"] == "JobExtraction" and req["system"]
    assert job.keywords == ["C++"]


@pytest.mark.parametrize("cloud", [{"status": 401}], indirect=True)
def test_bad_key_gives_a_clear_message(cloud):
    models = CloudModels("openai", "gpt-6.1-terra", api_key="wrong", embedder=NoEmbeddings())
    with pytest.raises(Exception) as err:
        extract_chain(models).invoke({"jd": "x"})
    msg = str(friendly_error(err.value, models))
    assert "rejected the API key" in msg and "OPENAI_API_KEY" in msg


def test_health_and_factory(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    no_key = CloudModels("openrouter", "x/y", embedder=NoEmbeddings())
    assert no_key.health()["ok"] is False and "OPENROUTER_API_KEY" in no_key.health()["error"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    made = create_models("anthropic", "")
    assert made.chat_model == config.PROVIDERS["anthropic"]["models"][0] and made.health()["ok"]
    assert create_models("ollama", "qwen2.5:7b").label == "Ollama · qwen2.5:7b"
    with pytest.raises(LLMError):
        create_models("nope")
    assert "rate limit" in str(friendly_error(RuntimeError("Error code: 429"), no_key))
    assert "does not know the model" in str(friendly_error(RuntimeError("Error code: 404 model not found"), no_key))
