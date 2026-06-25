from __future__ import annotations

from io import BytesIO
from typing import List

from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF given its bytes."""
    reader = PdfReader(BytesIO(file_bytes))
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts).strip()

