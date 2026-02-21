"""
Central configuration.

Why this matters:
- In production you want settings to be consistent and controlled (chunk size, k, thresholds).
- Environment variables let you tune behavior without editing code.
"""

from __future__ import annotations
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    # Directories
    DATA_DIR: str = os.getenv("RAG_DATA_DIR", "data")
    STORAGE_DIR: str = os.getenv("RAG_STORAGE_DIR", "storage")

     # Chunking: smaller chunks improve precision; overlap preserves continuity across boundaries.
    CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "450"))
    CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))

    #retrieval
    TOP_K_VECTOR: int = int(os.getenv("RAG_TOP_K_VECTOR", "12"))   # retrieve more then rerank
    TOP_K_BM25: int = int(os.getenv("RAG_TOP_K_BM25", "12"))       # keyword candidate pool
    TOP_K_FINAL: int = int(os.getenv("RAG_TOP_K_FINAL", "5"))      # context size for LLM

    # Threshold: if similarity is below this, we prefer "I don't know".
    # This reduces hallucinations when retrieval is weak.
    MIN_VECTOR_SCORE: float = float(os.getenv("RAG_MIN_VECTOR_SCORE", "0.25"))


     # Models
    EMBED_MODEL: str = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # A lightweight reranker; strong boost in precision vs pure embedding similarity.
    RERANK_MODEL: str = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LLM
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    TEMPERATURE: float = float(os.getenv("RAG_TEMPERATURE", "0.2"))

    # Behavior toggles
    USE_RERANKER: bool = os.getenv("RAG_USE_RERANKER", "1") == "1"
    USE_OPENAI: bool = os.getenv("RAG_USE_OPENAI", "1") == "1"   # set 0 to disable generation


settings = Settings()