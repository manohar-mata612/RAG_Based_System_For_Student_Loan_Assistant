"""
chunker.py
----------
Splits text into overlapping chunks ready for embedding.
Each chunk gets metadata so the UI can show source citations.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

CHUNK_SIZE    = 400   # ~400 tokens — good precision for retrieval
CHUNK_OVERLAP = 200    # overlap keeps context across boundaries


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_text(text: str, metadata: dict) -> list:
    """
    Split a string into LangChain Documents with attached metadata.

    Args:
        text:     Raw text to split.
        metadata: Dict added to every chunk (source, lender_name, doc_type, etc.)

    Returns:
        List of LangChain Document objects.
    """
    chunks = _splitter().split_text(text)
    return [
        Document(page_content=chunk, metadata={**metadata, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]


def chunk_pages(pages: list, base_metadata: dict) -> list:
    """
    Chunk a list of page dicts (from pdf_parser).
    Adds page_number to each chunk's metadata.
    """
    docs = []
    for page_info in pages:
        page_meta = {**base_metadata, "page_number": page_info["page"]}
        docs.extend(chunk_text(page_info["text"], page_meta))
    return docs
