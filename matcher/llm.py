"""Model access through LangChain.

`OllamaModels` runs everything locally. `CloudModels` uses OpenAI, Anthropic or OpenRouter for the
chat model (embeddings stay local on Ollama). Chains and graph nodes ask either one for
  * `structured(Schema, temperature)` – a runnable that returns a validated Pydantic object
  * `embed(texts)`                    – embedding vectors (or None if unavailable)
Tests swap in a fake object with the same two methods, so they run without Ollama.
"""
from __future__ import annotations

import os
from typing import Any, Protocol

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from . import config


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or returns unusable output."""


class Models(Protocol):
    def structured(self, schema: type[BaseModel], temperature: float = 0.0) -> Runnable: ...

    def embed(self, texts: list[str]) -> list[list[float]] | None: ...


class OllamaModels:
    provider = "ollama"

    def __init__(self, host: str | None = None, chat_model: str | None = None,
                 embed_model: str | None = None, num_ctx: int | None = None):
        import ollama  # only needed for the health check

        self.host = host or config.OLLAMA_HOST
        self.chat_model = chat_model or config.CHAT_MODEL
        self.embed_model = embed_model or config.EMBED_MODEL
        self.num_ctx = num_ctx or config.NUM_CTX
        self.client = ollama.Client(host=self.host, timeout=10)
        self.label = f"Ollama · {self.chat_model}"
        self._thinking_model: bool | None = None  # detected on first use
        self._embeddings = None

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

    def is_thinking_model(self) -> bool:
        """True for models with a 'thinking' mode (Qwen3/3.5, DeepSeek-R1 …)."""
        if self._thinking_model is None:
            try:
                caps = getattr(self.client.show(self.chat_model), "capabilities", None) or []
                self._thinking_model = "thinking" in caps
            except Exception:
                self._thinking_model = False
        return self._thinking_model

    # ---------- chat ----------
    def chat(self, temperature: float = 0.0, reasoning: bool | None = None):
        from langchain_ollama import ChatOllama

        return ChatOllama(model=self.chat_model, base_url=self.host, temperature=temperature, seed=42,
                          num_ctx=self.num_ctx, reasoning=reasoning,
                          client_kwargs={"timeout": config.TIMEOUT_S})

    def structured(self, schema: type[BaseModel], temperature: float = 0.0) -> Runnable:
        """A runnable: prompt -> validated `schema` object (Ollama enforces the JSON schema).

        Thinking is switched off for thinking models (faster, and the answer can't get lost in
        the thinking field) unless CVM_THINK=1. If a model rejects the flag, the fallback
        retries without it. Broken JSON is retried once.
        """
        if self.is_thinking_model():
            primary = self.chat(temperature, reasoning=config.THINK).with_structured_output(schema)
            plain = self.chat(temperature, reasoning=None).with_structured_output(schema)
            return primary.with_fallbacks([plain])
        return self.chat(temperature).with_structured_output(schema).with_retry(stop_after_attempt=2)

    # ---------- embeddings ----------
    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Return embeddings, or None if the embedding model isn't available."""
        if not texts:
            return []
        try:
            if self._embeddings is None:
                from langchain_ollama import OllamaEmbeddings

                self._embeddings = OllamaEmbeddings(model=self.embed_model, base_url=self.host)
            return self._embeddings.embed_documents(texts)
        except Exception:
            return None


class CloudModels:
    """Chat model from a cloud provider; embeddings still come from local Ollama.

    With a cloud model the CV and job ad are sent to that provider — the app asks for consent first.
    """

    def __init__(self, provider: str, chat_model: str, api_key: str | None = None, embedder: Any = None):
        self.spec = config.PROVIDERS[provider]
        self.provider = provider
        self.chat_model = chat_model
        self.api_key = api_key or os.getenv(self.spec["env_key"], "")
        self.host = self.spec["label"]
        self.label = f"{self.spec['label']} · {chat_model}"
        self.embedder = embedder if embedder is not None else OllamaModels()
        self.embed_model = getattr(self.embedder, "embed_model", config.EMBED_MODEL)

    def health(self) -> dict[str, Any]:
        local = self.embedder.health() if hasattr(self.embedder, "health") else {"embed_model": True}
        ok = bool(self.api_key)
        return {"ok": ok, "chat_model": ok, "models": [], "embed_model": bool(local.get("embed_model")),
                "error": "" if ok else f"No API key found. Add {self.spec['env_key']}=… to the .env file "
                                       "in the project folder and restart the app."}

    def chat(self, temperature: float | None = None):
        common = {"model": self.chat_model, "api_key": self.api_key, "timeout": config.TIMEOUT_S,
                  "max_retries": 1, "max_tokens": config.CLOUD_MAX_TOKENS}
        if temperature is not None:
            common["temperature"] = temperature
        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(**common)
        from langchain_openai import ChatOpenAI

        if self.spec["base_url"]:
            common["base_url"] = self.spec["base_url"]
        return ChatOpenAI(**common)

    def structured(self, schema: type[BaseModel], temperature: float = 0.0) -> Runnable:
        """prompt -> validated `schema` object, with fallbacks for models with fewer features.

        Order: strict JSON schema → tool calling (for models without JSON-schema support) →
        the same without a temperature (some reasoning models only accept their default).
        If every attempt fails, the first error is the one reported.
        """
        if self.provider == "anthropic":
            attempts = [self.chat(temperature).with_structured_output(schema),
                        self.chat(None).with_structured_output(schema)]
        else:
            attempts = [self.chat(temperature).with_structured_output(schema, method="json_schema", strict=True),
                        self.chat(temperature).with_structured_output(schema, method="function_calling"),
                        self.chat(None).with_structured_output(schema, method="json_schema", strict=True)]
        return attempts[0].with_fallbacks(attempts[1:])

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        return self.embedder.embed(texts)


def create_models(provider: str = "ollama", chat_model: str | None = None):
    """Build the model object for a provider chosen in the app."""
    if provider == "ollama":
        return OllamaModels(chat_model=chat_model or None)
    if provider not in config.PROVIDERS:
        raise LLMError(f"Unknown provider '{provider}'.")
    return CloudModels(provider, chat_model or config.PROVIDERS[provider]["models"][0])


def friendly_error(exc: Exception, models: Any = None) -> LLMError:
    """Turn a low-level failure into a message the user can act on."""
    if isinstance(exc, LLMError):
        return exc
    name = getattr(models, "chat_model", "the model")
    host = getattr(models, "host", config.OLLAMA_HOST)
    cloud = getattr(models, "provider", "ollama") != "ollama"
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if cloud:
        key = models.spec["env_key"]
        if "401" in low or "authentication" in low or "invalid api key" in low or "invalid x-api-key" in low \
                or "api_key" in low or "no auth" in low:
            return LLMError(f"{host} rejected the API key. Check {key} in your .env file and restart the app.")
        if "402" in low or "insufficient" in low or "credit" in low or "billing" in low or "quota" in low:
            return LLMError(f"{host}: no credit left on this account. Add credit or switch to another model.")
        if "429" in low or "rate limit" in low or "ratelimit" in low:
            return LLMError(f"{host}: rate limit reached. Wait a moment and try again.")
        if "404" in low or "not found" in low or "does not exist" in low or "not a valid model" in low:
            return LLMError(f"{host} does not know the model '{name}'. Check the exact model name.")
        if "timeout" in low or "timed out" in low:
            return LLMError(f"{host} did not answer within {int(config.TIMEOUT_S)} s. Try again.")
        if "connect" in low:
            return LLMError(f"Cannot reach {host}. Check your internet connection.")
        if "validation" in low or "parse" in low or "json" in low:
            return LLMError(f"The model '{name}' returned an answer in the wrong format. Try again or pick "
                            "another model (on OpenRouter, one that supports structured outputs).")
        return LLMError(f"{host} call failed: {text[:300]}")
    if "timeout" in low or "timed out" in low:
        return LLMError(f"The model '{name}' took longer than {int(config.TIMEOUT_S)} s. "
                        "The first run loads the model — try again, or use a smaller model.")
    if "connect" in low or "refused" in low:
        return LLMError(f"Cannot reach Ollama at {host}. Is the Ollama app running?")
    if "not found" in low and "model" in low:
        return LLMError(f"Model '{name}' is not installed. Run:  ollama pull {name}")
    if "validation" in low or "parse" in low or "json" in low:
        return LLMError(f"The model '{name}' returned an answer in the wrong format. Try again "
                        "(small models do this sometimes) or use qwen2.5:7b.")
    return LLMError(f"The model call failed: {text[:300]}")
