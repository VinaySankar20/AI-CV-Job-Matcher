"""Thin wrapper around the local Ollama server.

Any object with the same three methods (chat_json, chat_text, embed) can be used
instead, which is how the tests run without Ollama and how other providers
(Gemini, Claude) can be plugged in later.
"""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

from . import config


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or returns unusable output."""


class LLM(Protocol):
    def chat_json(self, system: str, user: str, schema: dict | None = None,
                  temperature: float = 0.0) -> dict: ...

    def chat_text(self, system: str, user: str, temperature: float = 0.4) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]] | None: ...


def extract_json(text: str) -> dict:
    """Parse JSON from a model answer, tolerating code fences or extra prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise LLMError("The model did not return valid JSON.")


class OllamaLLM:
    def __init__(self, host: str | None = None, chat_model: str | None = None,
                 embed_model: str | None = None, num_ctx: int | None = None):
        import ollama  # imported here so tests don't need a server

        self._ollama = ollama
        self.host = host or config.OLLAMA_HOST
        self.chat_model = chat_model or config.CHAT_MODEL
        self.embed_model = embed_model or config.EMBED_MODEL
        self.num_ctx = num_ctx or config.NUM_CTX
        self.think_supported: bool | None = None  # unknown until the first call
        self.client = ollama.Client(host=self.host, timeout=config.TIMEOUT_S)

    # ---------- status ----------
    def health(self) -> dict[str, Any]:
        """Check the server is running and which of our models are installed."""
        try:
            names = [m.model for m in self.client.list().models]
        except Exception as exc:  # connection refused etc.
            return {"ok": False, "error": f"Cannot reach Ollama at {self.host} ({exc})",
                    "chat_model": False, "embed_model": False, "models": []}

        def installed(model: str) -> bool:
            wanted = model if ":" in model else f"{model}:latest"
            return any(n == wanted or n == model for n in names)

        return {"ok": True, "error": "", "models": names,
                "chat_model": installed(self.chat_model),
                "embed_model": installed(self.embed_model)}

    # ---------- chat ----------
    def _chat(self, system: str, user: str, fmt: Any, temperature: float) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.chat_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            format=fmt,
            options={"temperature": temperature, "seed": 42, "num_ctx": self.num_ctx},
        )
        # Thinking models (Qwen3 / Qwen3.5, DeepSeek-R1 …) are much slower and can leave the
        # answer empty when thinking is on, so we switch it off. Models without thinking
        # support reject the flag — then we remember that and retry without it.
        if self.think_supported is not False:
            kwargs["think"] = config.THINK
        try:
            try:
                resp = self.client.chat(**kwargs)
            except self._ollama.ResponseError as exc:
                if "think" in kwargs and "think" in str(exc.error).lower():
                    self.think_supported = False
                    kwargs.pop("think")
                    resp = self.client.chat(**kwargs)
                else:
                    raise
        except self._ollama.ResponseError as exc:
            raise LLMError(f"Ollama error: {exc.error}. Did you run 'ollama pull {self.chat_model}'?") from exc
        except Exception as exc:
            if "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower():
                raise LLMError(f"The model '{self.chat_model}' took longer than {int(config.TIMEOUT_S)} s. "
                               "The first run loads the model — try again, or use a smaller model.") from exc
            raise LLMError(f"Cannot reach Ollama at {self.host}. Is the Ollama app running? ({exc})") from exc

        content = resp.message.content or ""
        if not content.strip():  # some thinking models still put everything in the thinking field
            content = getattr(resp.message, "thinking", None) or ""
        if not content.strip():
            raise LLMError(f"The model '{self.chat_model}' returned an empty answer. "
                           "Try again or use qwen2.5:7b.")
        return content

    def chat_json(self, system: str, user: str, schema: dict | None = None,
                  temperature: float = 0.0) -> dict:
        fmt = schema if schema else "json"
        last_error: Exception | None = None
        for _ in range(2):  # one retry on broken JSON
            try:
                return extract_json(self._chat(system, user, fmt, temperature))
            except LLMError as exc:
                last_error = exc
                if "reach Ollama" in str(exc) or "Ollama error" in str(exc):
                    raise
        raise LLMError(str(last_error))

    def chat_text(self, system: str, user: str, temperature: float = 0.4) -> str:
        return self._chat(system, user, None, temperature).strip()

    # ---------- embeddings ----------
    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Return embeddings, or None if the embedding model isn't available."""
        if not texts:
            return []
        try:
            resp = self.client.embed(model=self.embed_model, input=texts)
            return [list(v) for v in resp.embeddings]
        except Exception:
            return None
