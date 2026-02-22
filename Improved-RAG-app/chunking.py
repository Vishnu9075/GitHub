"""
Chunking is the #1 reason simple RAG feels bad.

Best practice:
- Avoid huge chunks that contain multiple sections.
- Keep chunks "topic focused".
- Use overlap so key info isn't cut at boundaries.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Chunk:
    text: str
    metadata = Dict[str, str]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s", " ", text).strip()
    if not text:
        return []
    #naive sentence splitting

    return _SENT_SPLIT.split(text)


def make_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Build chunks by accumulating sentences.

    Why sentence-based chunking helps:
    - Resumes and docs often have bullet lines. Sentence/bullet grouping keeps chunks coherent.
    - Char-slicing can cut bullets in half, damaging retrieval.
    """
    sents = split_sentences(text)
    if not sents:
        return []
    
    chunks : List[str] = []
    buf : List[str] = []
    buf_len = 0

    for s in sents:
        #if adding sentence would exceed chunk, flush
        if buf and (buf_len + len(s) +1 > chunk_size):
            chunk = " ".join(buf).strip()
            if chunk:
                chunks.append(chunk)

            # overlap logic: keep last N chars from previous chunk
            # This reduces "lost context" for boundary facts.

            if overlap > 0 and chunk:
                tail = chunk[-overlap:]
                buf = [tail]
                buf_len = len(tail)
            else:
                buf = []
                bur_len = 0

        # flush remaining

        final = " ".join(buf).strip()
        if final:
            chunks.append(final)
        return chunks

def chunk_pages(pages, chunk_size: int, overlap: int) -> List[chunk]:
    """
    Convert loaded pages into chunks with metadata.

    Metadata is copied into each chunk so:
    - citations work
    - filtering by source/page works
    - debugging retrieval becomes possible
    """
    out : List[Chunk] = []
    for page in pages:
        for i, c in enumerate(make_chunks(page.text, chunk_size, overlap)):
            meta = dict(page.metadata)
            meta["chunk_id"] = str(i)
            out.append(Chunk(text=c, metadata= meta))

        return out