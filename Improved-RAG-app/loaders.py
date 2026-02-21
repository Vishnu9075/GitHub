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
    text = path.read_text(encoding="utf-8", errors= "ignore")