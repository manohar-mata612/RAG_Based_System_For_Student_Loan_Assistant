"""
pdf_parser.py
-------------
Extracts raw text from PDF files using pypdf.
Works with both file paths and Streamlit UploadedFile objects.
"""

import io
from pypdf import PdfReader


def extract_text_from_pdf(file_source) -> list:
    """
    Extract text page-by-page from a PDF.

    Args:
        file_source: str file path  OR  file-like object (BytesIO / Streamlit UploadedFile)

    Returns:
        List of {"page": int, "text": str} dicts — one per non-empty page.
    """
    if isinstance(file_source, str):
        reader = PdfReader(file_source)
    else:
        file_source.seek(0)
        reader = PdfReader(io.BytesIO(file_source.read()))

    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": page_num, "text": text})

    return pages


def extract_full_text(file_source) -> str:
    """Return all page text joined into a single string."""
    return "\n\n".join(p["text"] for p in extract_text_from_pdf(file_source))
