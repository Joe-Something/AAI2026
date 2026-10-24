"""Central configuration.

The Gemini API key is read from a secret named ``GEMINI_API_KEY``, checked in this order:
  1. Google Colab secrets (``google.colab.userdata``), when running in Colab
  2. The ``GEMINI_API_KEY`` environment variable
  3. A ``.env`` file in the project root

If no key is found the system runs in *offline mode*: deterministic keyword classification,
hashed bag-of-words embeddings and extractive answers. Offline mode exists so the graph,
tools and tests can be exercised without network access or API quota; it is clearly
labelled in every response and in evaluation reports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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

    @property
    def offline(self) -> bool:
        return self.force_offline or not self.api_key

    @property
    def min_relevance(self) -> float:
        if self.min_relevance_override is not None:
            return self.min_relevance_override
        # Offline value calibrated on eval/scenarios.json (relevant >= 0.16, off-topic <= 0.12).
        return 0.15 if self.offline else 0.55

    @property
    def mode_label(self) -> str:
        return "offline (no GEMINI_API_KEY)" if self.offline else f"gemini:{self.chat_model}"


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
