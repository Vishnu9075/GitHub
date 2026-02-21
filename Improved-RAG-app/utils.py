from __future__ import annotations
import hashlib
import time
from contextlib import contextmanager

def sha256_bytes(data: bytes) -> str:
     """Stable hashing for change detection (incremental indexing)."""
     return hashlib.sha256(data).hexdigest()

def sha256_text(text: str) -> str:
     return sha256_bytes(text.encode("utf-8", error= "ignore"))

@contextmanager
def timed(label: str):
    """Tiny timing helper; useful when you later add observability/metrics."""
    t0 = time.time()
    try:
        yield
    finally:
        dt = (time.time() - t0) * 1000
        print(f"[timing] {label} : {dt: 1f}ms")
