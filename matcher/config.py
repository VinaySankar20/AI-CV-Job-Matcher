"""Central settings. Override any of these with environment variables if needed."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("CVM_DATA_DIR", PROJECT_DIR / "data"))
DB_PATH = DATA_DIR / "cv_matcher.db"

# API keys for cloud providers are read from a local ".env" file in the project folder
# (git-ignored). Values already set in the environment win over the file.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env")
except ImportError:  # python-dotenv not installed: only real environment variables are used
    pass

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.getenv("CVM_CHAT_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("CVM_EMBED_MODEL", "nomic-embed-text")
NUM_CTX = int(os.getenv("CVM_NUM_CTX", "8192"))
# Thinking mode for models that support it (Qwen3/3.5, DeepSeek-R1). Off = faster, reliable JSON.
THINK = os.getenv("CVM_THINK", "0").lower() in ("1", "true", "yes")
# Give up on one model call after this many seconds (instead of loading forever)
TIMEOUT_S = float(os.getenv("CVM_TIMEOUT", "300"))

# Balanced weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "skills": 0.35,
    "experience": 0.25,
    "languages": 0.15,
    "education": 0.15,
    "keywords": 0.10,
}

CATEGORY_LABELS = {
    "skills": "Skills",
    "experience": "Experience",
    "languages": "Languages",
    "education": "Education",
    "keywords": "Keywords / ATS",
}

# A cover letter is generated only when the overall score is ABOVE this value
COVER_LETTER_THRESHOLD = 3.5

# Penalty for every must-have requirement that is completely missing
MUST_HAVE_PENALTY = 0.25
MUST_HAVE_PENALTY_MAX = 1.0

# How the skills category is blended from its three signals
SKILLS_BLEND = {"llm": 0.5, "requirements": 0.3, "semantic": 0.2}

# Cosine-similarity range mapped to 0..1 for the semantic check (nomic-embed-text).
# Calibrated on real runs: unrelated requirements score ~0.54-0.58, real matches ~0.60-0.72.
SIM_LOW = 0.55
SIM_HIGH = 0.72
# Below this similarity (and no keyword / verified evidence) an AI "met" claim is downgraded to "partial"
SIM_VERIFY_MIN = 0.62

# Model providers the app can use. "models" are only suggestions for the dropdown — any model
# name can be typed in the app. Edit these lists freely when providers release new models.
PROVIDERS = {
    "ollama": {"label": "Ollama (local)", "cloud": False, "env_key": None, "base_url": None,
               "models": [CHAT_MODEL]},
    "openai": {"label": "OpenAI", "cloud": True, "env_key": "OPENAI_API_KEY", "base_url": None,
               "models": ["gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-luna"],
               "key_url": "https://platform.openai.com/api-keys"},
    "anthropic": {"label": "Anthropic", "cloud": True, "env_key": "ANTHROPIC_API_KEY", "base_url": None,
                  "models": ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"],
                  "key_url": "https://console.anthropic.com/settings/keys"},
    "openrouter": {"label": "OpenRouter", "cloud": True, "env_key": "OPENROUTER_API_KEY",
                   "base_url": "https://openrouter.ai/api/v1",
                   "models": ["openai/gpt-6.1-terra", "anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5",
                              "google/gemini-4-argon", "deepseek/deepseek-v4.1-flash-1017"],
                   "key_url": "https://openrouter.ai/keys"},
}
DEFAULT_PROVIDER = "ollama"
CLOUD_MAX_TOKENS = 4096   # upper limit for one answer from a cloud model

# LangGraph loop limits
MAX_RECHECKS = 1          # how often to ask again for requirements the model skipped
MAX_LETTER_REWRITES = 2   # automatic rewrites when the fact check finds unsupported claims
CHECKPOINT_DB = DATA_DIR / "graph_checkpoints.db"   # saved graph runs (pause / resume)

APPLICATION_STATUSES = ["analysed", "applied", "interview", "offer", "rejected", "not applying"]
