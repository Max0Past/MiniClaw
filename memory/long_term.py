"""ChromaDB-backed vector store for persistent long-term knowledge."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import chromadb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "openclaw_memory"


@dataclass
class MemoryResult:
    """A single search result from the vector store."""

    id: str
    text: str
    distance: float
    metadata: dict


@dataclass
class MemoryRecord:
    """A stored record (for debug page listing)."""

    id: str
    text: str
    metadata: dict


class LongTermMemory:
    """Wraps ChromaDB with a simple store / query / list API.

    Uses ChromaDB's built-in default embedding function
    (all-MiniLM-L6-v2 via onnxruntime) -- no extra model pull needed.
    """

    def __init__(self, persist_dir: Path = CHROMA_DIR) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def store(self, text: str, metadata: dict | None = None) -> str:
        """Embed and store a text chunk. Returns the document ID."""
        doc_id = uuid.uuid4().hex[:12]
        meta = metadata or {}
        meta.setdefault("stored_at", datetime.now(timezone.utc).isoformat())
        self._collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[meta],
        )
        return doc_id

    def query(self, query_text: str, n_results: int = 5) -> list[MemoryResult]:
        """Retrieve the top-n most relevant chunks."""
        if self._collection.count() == 0:
            return []
        n = min(n_results, self._collection.count())
        results = self._collection.query(query_texts=[query_text], n_results=n)
        out: list[MemoryResult] = []
        for doc, dist, meta, doc_id in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["ids"][0],
        ):
            out.append(MemoryResult(id=doc_id, text=doc, distance=dist, metadata=meta))
        return out

    def get_all(self) -> list[MemoryRecord]:
        """Return every stored record (for debug page display)."""
        if self._collection.count() == 0:
            return []
        data = self._collection.get()
        out: list[MemoryRecord] = []
        for doc_id, doc, meta in zip(
            data["ids"],
            data["documents"],
            data["metadatas"],
        ):
            out.append(MemoryRecord(id=doc_id, text=doc, metadata=meta))
        return out

    def delete(self, doc_id: str) -> None:
        """Remove a document by ID."""
        self._collection.delete(ids=[doc_id])

    def count(self) -> int:
        """Return the number of stored documents."""
        return self._collection.count()
