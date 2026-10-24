"""RAG knowledge base: Markdown KB articles -> section chunks -> embeddings -> Chroma.

Chunking is by ``##`` section (with the article title prepended) so each chunk is a
self-contained answer unit and citations can point at a specific section.
The collection is rebuilt when the KB content or the embedding model changes
(tracked with a content hash stored in the collection metadata).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb

from .config import KB_DIR, RUNTIME_DIR
from .llm import LLM, get_llm


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    category: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, 1.0 = identical

    @property
    def citation(self) -> str:
        return f"{self.chunk.doc_id} · {self.chunk.title} › {self.chunk.section}"


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    meta: dict[str, str] = {}
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        for line in fm.strip().splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
        return meta, body
    return meta, raw


def load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        doc_id, title, category = meta["id"], meta["title"], meta.get("category", "general")
        # Split on level-2 headings; the text before the first one is the "Overview".
        parts = re.split(r"^## +(.+)$", body, flags=re.M)
        intro = re.sub(r"^# .+$", "", parts[0], flags=re.M).strip()
        sections = [("Overview", intro)] if intro else []
        sections += [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]
        for n, (section, text) in enumerate(sections):
            chunks.append(Chunk(f"{doc_id}#{n}", doc_id, title, section, category, f"{title} — {section}\n{text}"))
    return chunks


class KnowledgeBase:
    def __init__(self, llm: LLM | None = None, persist: bool = True):
        self.llm = llm or get_llm()
        self.chunks = load_chunks()
        self._by_id = {c.id: c for c in self.chunks}
        if persist:
            RUNTIME_DIR.mkdir(exist_ok=True)
            client = chromadb.PersistentClient(path=str(RUNTIME_DIR / "chroma"))
        else:
            client = chromadb.EphemeralClient()
        # One collection per embedding model so switching models never mixes vector spaces.
        name = "kb_" + re.sub(r"[^a-zA-Z0-9_-]", "_", self.llm.embedding_model)[:50]
        self.collection = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self._ensure_indexed()

    def _content_hash(self) -> str:
        return hashlib.sha256("".join(c.id + c.text for c in self.chunks).encode()).hexdigest()[:16]

    def _ensure_indexed(self) -> None:
        digest = self._content_hash()
        existing = self.collection.get(ids=["__meta__"], include=["metadatas"])
        if existing["ids"] and existing["metadatas"][0].get("hash") == digest:
            return
        if self.collection.count():
            self.collection.delete(ids=self.collection.get()["ids"])
        vectors = self.llm.embed([c.text for c in self.chunks], task="document")
        if len(vectors) != len(self.chunks):
            raise RuntimeError(f"Embedding model {self.llm.embedding_model} returned {len(vectors)} vectors for {len(self.chunks)} chunks")
        self.collection.add(
            ids=[c.id for c in self.chunks],
            embeddings=vectors.tolist(),
            documents=[c.text for c in self.chunks],
            metadatas=[{"doc_id": c.doc_id, "title": c.title, "section": c.section, "category": c.category} for c in self.chunks],
        )
        self.collection.add(ids=["__meta__"], embeddings=[vectors[0].tolist()], documents=[""], metadatas=[{"hash": digest}])

    def search(self, query: str, k: int = 4) -> list[Hit]:
        qvec = self.llm.embed([query], task="query")[0]
        res = self.collection.query(query_embeddings=[qvec.tolist()], n_results=k + 1, include=["distances"])
        hits = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            if cid == "__meta__":
                continue
            hits.append(Hit(self._by_id[cid], round(1.0 - dist, 4)))
        return hits[:k]

    def chunk(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    @property
    def size(self) -> dict[str, int]:
        return {"articles": len({c.doc_id for c in self.chunks}), "chunks": len(self.chunks)}


_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


def reset_kb() -> None:
    global _kb
    _kb = None
