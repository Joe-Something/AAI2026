"""Central configuration.

The Gemini API key is read from a secret named ``GEMINI_API_KEY``, checked in this order:
  1. Google Colab secrets (``google.colab.userdata``), when running in Colab
  2. The ``GEMINI_API_KEY`` environment variable
  3. A ``.env`` file in the project root

Three providers (``HELPDESK_PROVIDER`` = auto | gemini | local | offline):
  - gemini  : Google Gemini API (needs the key above)
  - local   : open-source models served by Ollama on this machine, no key and no data leaving it
              (default qwen2.5:7b for the agents, nomic-embed-text for embeddings)
  - offline : deterministic keyword classification, hashed bag-of-words embeddings and extractive
              answers; no model at all. Used by the unit tests and as an evaluation baseline.
``auto`` picks gemini when a key exists, else local when Ollama is running with the model pulled,
else offline. The active provider is labelled in every response and evaluation report.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

SECRET_NAME = "GEMINI_API_KEY"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "kb"
LOG_DIR = DATA_DIR / "logs"
RUNTIME_DIR = PROJECT_ROOT / "runtime"  # mutable state: ticket DB, directory copy, vector store


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_api_key() -> str | None:
    """Return the Gemini key from Colab secrets, env, or .env (never logged or printed)."""
    try:
        from google.colab import userdata  # type: ignore

        key = userdata.get(SECRET_NAME)
        if key:
            return key
    except Exception:
        # Not in Colab, or the secret exists but notebook access was not granted.
        pass
    _load_dotenv()
    return os.environ.get(SECRET_NAME) or None


@dataclass
class Settings:
    api_key: str | None = field(default_factory=get_api_key)
    # "gemini-flash-latest" tracks Google's current stable Flash model, so the project keeps
    # working when a pinned version is retired (gemini-2.5-* was shut down 2026-10-16).
    chat_model: str = field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-flash-latest"))
    embedding_models: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            m.strip()
            for m in os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2,gemini-embedding-001").split(",")
            if m.strip()
        )
    )
    force_offline: bool = field(default_factory=lambda: os.environ.get("HELPDESK_OFFLINE", "").lower() in ("1", "true", "yes"))
    provider_preference: str = field(default_factory=lambda: os.environ.get("HELPDESK_PROVIDER", "auto").lower())
    ollama_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"))
    local_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"))
    local_embedding_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))
    # Retrieval: chunks below this cosine similarity are treated as "not in the knowledge base".
    retrieval_top_k: int = 4
    # Gemini and offline embeddings live on different similarity scales, so each mode has its own default.
    min_relevance_override: float | None = field(
        default_factory=lambda: float(os.environ["HELPDESK_MIN_RELEVANCE"]) if os.environ.get("HELPDESK_MIN_RELEVANCE") else None
    )
    # Seconds between Gemini calls; raise this on the free tier to stay under requests-per-minute limits.
    min_call_interval: float = field(default_factory=lambda: float(os.environ.get("GEMINI_MIN_INTERVAL", "0")))
    # Optional: escalations also open a GitHub issue when both are set.
    github_token: str | None = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN") or None)
    github_repo: str | None = field(default_factory=lambda: os.environ.get("GITHUB_REPO") or None)

    @cached_property
    def provider(self) -> str:
        pref = self.provider_preference
        if self.force_offline or pref == "offline":
            return "offline"
        if pref == "gemini":
            return "gemini" if self.api_key else "offline"
        if pref == "local":
            return "local"
        if self.api_key:
            return "gemini"
        return "local" if self.ollama_ready() else "offline"

    def ollama_ready(self) -> bool:
        """True when the Ollama server answers and both local models are pulled."""
        try:
            with urllib.request.urlopen(f"{self.ollama_url}/api/tags", timeout=1.5) as r:
                names = {m["name"] for m in json.loads(r.read()).get("models", [])}
        except Exception:
            return False
        have = lambda m: m in names or f"{m}:latest" in names  # noqa: E731
        return have(self.local_model) and have(self.local_embedding_model)

    @property
    def offline(self) -> bool:
        return self.provider == "offline"

    @property
    def min_relevance(self) -> float:
        if self.min_relevance_override is not None:
            return self.min_relevance_override
        # Each embedding model has its own similarity scale; values calibrated on eval/scenarios.json.
        return {"offline": 0.15, "local": LOCAL_MIN_RELEVANCE, "gemini": 0.55}[self.provider]

    @property
    def mode_label(self) -> str:
        return {"offline": "offline (rule-based, no LLM)",
                "local": f"local:{self.local_model} via Ollama (no API key)",
                "gemini": f"gemini:{self.chat_model}"}[self.provider]


# nomic-embed-text on eval/scenarios.json: relevant questions score >= 0.676, off-topic ones <= 0.594.
LOCAL_MIN_RELEVANCE = 0.64


_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Re-read configuration (used by tests and notebooks after changing env vars)."""
    global _settings
    _settings = None
