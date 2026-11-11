"""Central settings. Override any of these with environment variables if needed."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("CVM_DATA_DIR", PROJECT_DIR / "data"))
DB_PATH = DATA_DIR / "cv_matcher.db"

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

APPLICATION_STATUSES = ["analysed", "applied", "interview", "offer", "rejected", "not applying"]
