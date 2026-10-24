"""Thin wrapper around the LLM providers with retries, pacing and metrics.

Agents call ``structured()`` for typed JSON output (validated by Pydantic), ``text()`` for
free-form answers and ``embed()`` for vectors, whichever provider is active:
  - gemini : Google Gemini API via the google-genai SDK (response_schema for structured output)
  - local  : Ollama's HTTP API on this machine (``format`` = the Pydantic JSON schema)
  - offline: ``embed()`` falls back to a deterministic hashed bag-of-words embedding and agents
             use their own rule-based paths (see ``helpdesk/offline.py``)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
from pydantic import BaseModel

from .config import Settings, settings

T = TypeVar("T", bound=BaseModel)

OFFLINE_DIM = 1024
EMBED_DIM = 768
_RETRYABLE = {429, 500, 502, 503, 504}


@dataclass
class CallStats:
    llm_calls: int = 0
    embed_calls: int = 0
    llm_seconds: float = 0.0
    retries: int = 0
    errors: list[str] = field(default_factory=list)


class LLM:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings()
        self.stats = CallStats()
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._client = None
        self._embedding_model: str | None = None
        self.provider = self.cfg.provider
        if self.provider == "gemini":
            from google import genai

            self._client = genai.Client(api_key=self.cfg.api_key)

    @property
    def offline(self) -> bool:
        return self.provider == "offline"

    # ------------------------------------------------------------------ generation
    def structured(self, prompt: str, schema: type[T], system: str) -> T:
        if self.provider == "local":
            return schema.model_validate_json(self._ollama_chat(prompt, system, 0.0, schema.model_json_schema()))
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
        )
        resp = self._generate(prompt, config)
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        return schema.model_validate_json(resp.text)

    def text(self, prompt: str, system: str, temperature: float = 0.2) -> str:
        if self.provider == "local":
            return self._ollama_chat(prompt, system, temperature).strip()
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
        return (self._generate(prompt, config).text or "").strip()

    def _generate(self, prompt: str, config):
        if self.offline:
            raise RuntimeError("LLM is offline; agents must use their offline path.")
        start = time.perf_counter()
        try:
            return self._with_retries(
                lambda: self._client.models.generate_content(model=self.cfg.chat_model, contents=prompt, config=config)
            )
        finally:
            self.stats.llm_calls += 1
            self.stats.llm_seconds += time.perf_counter() - start

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: list[str], task: str = "document") -> np.ndarray:
        """Return an L2-normalised (n, d) matrix. task is 'document' or 'query'."""
        if self.offline:
            return np.vstack([offline_embed(t) for t in texts])
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 100):  # API batch limit
            batch = texts[i : i + 100]
            vectors.extend(self._ollama_embed(batch, task) if self.provider == "local" else self._embed_batch(batch, task))
        self.stats.embed_calls += 1
        arr = np.asarray(vectors, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-9)

    def _embed_batch(self, batch: list[str], task: str) -> list[list[float]]:
        from google.genai import errors, types

        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
        candidates = [self._embedding_model] if self._embedding_model else list(self.cfg.embedding_models)
        last_err: Exception | None = None
        for model in candidates:
            # Not every embedding model accepts task_type; retry once without it.
            for config in (
                types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
                types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
            ):
                try:
                    vectors = self._embed_with(model, config, batch)
                    self._embedding_model = model
                    return vectors
                except errors.APIError as e:
                    last_err = e
                    if e.code == 404:
                        break  # model unavailable: try the next candidate
                    if e.code != 400:
                        raise
        raise RuntimeError(f"No embedding model worked ({candidates}): {last_err}")

    def _embed_with(self, model: str, config, batch: list[str]) -> list[list[float]]:
        from google.genai import types

        def call(contents):
            return self._with_retries(lambda: self._client.models.embed_content(model=model, contents=contents, config=config))

        # One Content per text so each gets its own vector.
        resp = call([types.Content(parts=[types.Part(text=t)]) for t in batch])
        vectors = [e.values for e in resp.embeddings]
        if len(vectors) == len(batch):
            return vectors
        # Multimodal models (gemini-embedding-2) can merge a multi-item request into one vector: embed one by one.
        return [call(t).embeddings[0].values for t in batch]

    @property
    def embedding_model(self) -> str:
        if self.offline:
            return "offline-hashed-bow"
        if self.provider == "local":
            return self.cfg.local_embedding_model
        return self._embedding_model or self.cfg.embedding_models[0]

    def warm_up(self) -> None:
        """Load the local model into GPU memory so the first real request isn't slowed by it."""
        if self.provider == "local":
            self._ollama_chat("Reply with OK.", "Be brief.", 0.0)
            self.stats.llm_calls -= 1  # not a real request
            self._ollama_embed(["warm up"], "query")

    # ------------------------------------------------------------------ local (Ollama)
    def _ollama_chat(self, prompt: str, system: str, temperature: float, schema: dict | None = None) -> str:
        payload = {
            "model": self.cfg.local_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": "30m",  # keep the model in GPU memory between requests
            # num_predict caps the reply: small models in JSON mode can otherwise loop forever.
            "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": 1024},
        }
        if schema:
            payload["format"] = _bounded(schema)  # Ollama constrains decoding to this JSON schema
        start = time.perf_counter()
        try:
            return self._ollama("/api/chat", payload)["message"]["content"]
        finally:
            self.stats.llm_calls += 1
            self.stats.llm_seconds += time.perf_counter() - start

    def _ollama_embed(self, batch: list[str], task: str) -> list[list[float]]:
        model = self.cfg.local_embedding_model
        if "nomic" in model:  # nomic-embed-text is trained with these task prefixes
            prefix = "search_query: " if task == "query" else "search_document: "
            batch = [prefix + t for t in batch]
        vectors = self._ollama("/api/embed", {"model": model, "input": batch, "keep_alive": "30m"})["embeddings"]
        if len(vectors) != len(batch):
            raise RuntimeError(f"Ollama returned {len(vectors)} embeddings for {len(batch)} texts")
        return vectors

    def _ollama(self, path: str, payload: dict, attempts: int = 3) -> dict:
        req = urllib.request.Request(f"{self.cfg.ollama_url}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:  # first call loads the model into memory
                    return json.loads(r.read())
            except TimeoutError as e:
                if attempt == attempts - 1:
                    raise RuntimeError(f"Ollama timed out on {path}") from e
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:200]
                if e.code < 500 or attempt == attempts - 1:
                    self.stats.errors.append(f"ollama {e.code}: {detail}")
                    raise RuntimeError(f"Ollama {e.code} on {path}: {detail}") from e
            except urllib.error.URLError as e:
                if attempt == attempts - 1:
                    raise RuntimeError(f"Ollama is not reachable at {self.cfg.ollama_url} ({e.reason}). "
                                       "Start it with `ollama serve` or the Ollama app.") from e
            self.stats.retries += 1
            time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------ plumbing
    def _with_retries(self, fn, attempts: int = 5):
        from google.genai import errors

        delay = 2.0
        for attempt in range(attempts):
            self._pace()
            try:
                return fn()
            except errors.APIError as e:
                if e.code not in _RETRYABLE or attempt == attempts - 1:
                    self.stats.errors.append(f"{e.code}: {str(e)[:120]}")
                    raise
                self.stats.retries += 1
                time.sleep(delay)
                delay = min(delay * 2, 30)

    def _pace(self) -> None:
        if self.cfg.min_call_interval <= 0:
            return
        with self._lock:
            wait = self._last_call + self.cfg.min_call_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


# ---------------------------------------------------------------------- local schema bounds
_LONG_FIELDS = {"answer": 1200, "summary": 300, "rationale": 300}


def _bounded(schema: dict) -> dict:
    """Copy of a JSON schema with length limits on strings and lists.

    Ollama enforces these while decoding, which stops a small model rambling until its token
    limit cuts the JSON off. Gemini gets the plain schema (it rejects some of these keywords).
    """
    out = json.loads(json.dumps(schema))
    for name, prop in out.get("properties", {}).items():
        if prop.get("type") == "string" and "enum" not in prop:
            prop["maxLength"] = _LONG_FIELDS.get(name, 200)
        elif prop.get("type") == "array":
            prop["maxItems"] = 6
    return out


# ---------------------------------------------------------------------- offline embedding
_SYNONYMS = {
    "pw": "password", "passwd": "password", "pwd": "password", "passcode": "password",
    "locked": "lockout", "lock": "lockout", "unlock": "lockout",
    "globalconnect": "vpn", "tunnel": "vpn",
    "outlook": "email", "mail": "email", "inbox": "email", "mailbox": "email",
    "wifi": "wi-fi", "wireless": "wi-fi", "internet": "wi-fi",
    "print": "printer", "printing": "printer", "scan": "printer", "scanner": "printer", "toner": "printer",
    "install": "software", "app": "software", "application": "software", "program": "software",
    "slow": "performance", "freezing": "performance", "freeze": "performance", "laggy": "performance",
    "authenticator": "mfa", "2fa": "mfa", "two-factor": "mfa",
    "phish": "phishing", "phishy": "phishing", "suspicious": "phishing", "scam": "phishing",
    "permission": "access", "permissions": "access", "share": "access", "drive": "access",
}
_STOP = set("a an the to of and or in on for my i me is it its be am are was can't cannot can how do does what why with this that at from your you we our".split())


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())
    out = []
    for w in words:
        if w in _STOP:
            continue
        w = _SYNONYMS.get(w, w)
        for suffix in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.append(_SYNONYMS.get(w, w))
    return out


def offline_embed(text: str) -> np.ndarray:
    toks = _tokens(text)
    feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    vec = np.zeros(OFFLINE_DIM, dtype=np.float32)
    for f in feats:
        vec[int(hashlib.md5(f.encode()).hexdigest(), 16) % OFFLINE_DIM] += 1.0
    vec = np.log1p(vec)
    norm = math.sqrt(float(vec @ vec)) or 1.0
    return vec / norm


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


def reset_llm() -> None:
    global _llm
    _llm = None
