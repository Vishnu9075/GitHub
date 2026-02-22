"""
Persistent stores:
- FAISS for vectors (semantic search)
- BM25 for lexical search (keyword search)
- Metadata store for citations/debugging
- Manifest for incremental indexing

Why hybrid matters:
- Embeddings can miss exact keywords (names, acronyms, IDs).
- BM25 can miss semantic paraphrases.
- Hybrid gets you robust retrieval across query styles.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss
import orjson
from rank_bm25 import BM25Okapi

from utils import sha256_text


def _tok(text: str) -> List[str]:
    # Simple tokenizer; for production you might use a better tokenizer.
    return [t for t in text.lower().split() if t]


class RAGStore:
    def __init__(self, storage_dir: str):
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)

        self.faiss_path = self.base / "faiss.index"
        self.meta_path = self.base / "meta.jsonl"      # one json per line
        self.bm25_path = self.base / "bm25.json"       # tokens + mapping
        self.manifest_path = self.base / "manifest.json"

        self.index: faiss.Index | None = None
        self.meta: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self.bm25_tokens: List[List[str]] = []

        self.manifest: Dict[str, Any] = {"docs": {}}  # source -> content_hash

    # ---------------------------
    # Load/save helpers
    # ---------------------------
    def load(self) -> None:
        if self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
        if self.meta_path.exists():
            self.meta = [
                orjson.loads(line)
                for line in self.meta_path.read_bytes().splitlines()
                if line.strip()
            ]
        if self.bm25_path.exists():
            data = orjson.loads(self.bm25_path.read_bytes())
            self.bm25_tokens = data["tokens"]
            self.bm25 = BM25Okapi(self.bm25_tokens)

        if self.manifest_path.exists():
            self.manifest = orjson.loads(self.manifest_path.read_bytes())

    def save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.faiss_path))

        # meta.jsonl (streaming-friendly and appendable)
        meta_bytes = b"\n".join(orjson.dumps(m) for m in self.meta)
        self.meta_path.write_bytes(meta_bytes + b"\n")

        # BM25 tokens must be persisted because BM25Okapi itself isn't trivially serializable.
        bm25_data = {"tokens": self.bm25_tokens}
        self.bm25_path.write_bytes(orjson.dumps(bm25_data))

        self.manifest_path.write_bytes(orjson.dumps(self.manifest))

    # ---------------------------
    # Incremental indexing
    # ---------------------------
    def should_index_source(self, source: str, combined_text: str) -> bool:
        """
        Determine if a doc changed.

        Why this matters:
        - Embeddings cost time and money (if hosted).
        - Incremental indexing is a standard production requirement.
        """
        h = sha256_text(combined_text)
        prev = self.manifest["docs"].get(source)
        return prev != h

    def update_manifest(self, source: str, combined_text: str) -> None:
        self.manifest["docs"][source] = sha256_text(combined_text)

    # ---------------------------
    # Build indices
    # ---------------------------
    def build_from(self, vectors: np.ndarray, metadocs: List[Dict[str, Any]], bm25_texts: List[str]) -> None:
        """
        Build fresh indices.

        vectors must be float32 and normalized (cosine similarity via inner product).
        """
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        dim = int(vectors.shape[1])
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

        self.meta = metadocs

        self.bm25_tokens = [_tok(t) for t in bm25_texts]
        self.bm25 = BM25Okapi(self.bm25_tokens)

    # ---------------------------
    # Search
    # ---------------------------
    def search_vector(self, qvec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Returns [(chunk_idx, score), ...]
        Score is cosine similarity if vectors normalized.
        """
        if self.index is None:
            raise RuntimeError("FAISS index not loaded/built.")
        scores, ids = self.index.search(qvec.astype(np.float32), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def search_bm25(self, query: str, k: int) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not loaded/built.")
        toks = _tok(query)
        scores = self.bm25.get_scores(toks)

        # Take top-k indices
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0.0]