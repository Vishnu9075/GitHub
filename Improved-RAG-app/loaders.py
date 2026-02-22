"""
Load documents into text + metadata.

Best practice:
- Preserve metadata early (filename, page #, doc type).
- Metadata enables filtering and better citations later.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from pypdf import PdfReader


@dataclass
class DocPage:
    """A document can be represented as pages/segments with metadata."""
    text: str
    metadata: Dict[str, str]

def load_pdf(path: Path) -> List[DocPage]:
    reader = PdfReader(str(path))
    pages: List[DocPage]

    for i, page in enumerate(reader.pages):
        page_text = (page.extract_text() or "").strip()

        # If this is a scanned PDF, extract_text() may be empty.
        # In production you'd detect this and route to OCR.

        if not page_text:
            continue

        pages.append(DocPage(text = page_text, 
                             metadata= { "source": str(path), 
                                        "type": "pdf", 
                                        "page": str(i+1),},
                             ))
        
        return pages
    

def load_text_file(path: Path) -> List[DocPage]:
    text = path.read_text(encoding="utf-8", errors= "ignore").strip()
    if not text:
        return []
    return [
        DocPage(
            text = text,
            metadata= {
                "source": str(path),
                "type": path.suffix.lstrip(".") or "txt",
                "page" : "1",
            },
        )
    ]

def load_all(data_dir: str) -> List[DocPage]:
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Missing data directory: {base.resolve()}")
    
    out: List[DocPage] = []
    for p in base.rglob("*"):
        if p.is_dir():
            continue
        ext = p.suffix.lower()
        if ext == "pdf":
            out.extend(load_pdf(p))
        elif ext in [".txt", ".md"]:
            out.extend(load_text_file(p))

    return out