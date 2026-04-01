"""
ingest.py
---------
One-time ingestion script.
Reads .txt files from data/corpus/, embeds them with the FREE local
HuggingFace model, and upserts vectors into Pinecone.

Run:
    python -m rag.ingest
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from rag.retriever import get_embeddings, EMBED_DIM
from langchain_pinecone import PineconeVectorStore
from utils.chunker import chunk_text

load_dotenv()

CORPUS_DIR  = Path(__file__).parent.parent /"rag"/"utils"/ "data" / "corpus"
NAMESPACE   = "policy-corpus"
INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME", "loan-assistant")
BATCH_SIZE  = 50   # smaller batches = safer with Pinecone free tier


def _parse_header(text: str, filename: str) -> dict:
    """Pull metadata from the header block at the top of each corpus file."""
    meta = {"source": filename, "lender_name": "unknown",
            "doc_type": "policy", "url": ""}
    for pattern, key in [
        (r"^SOURCE:\s*(.+)$",   "source"),
        (r"^URL.*?:\s*(\S+)",   "url"),
        (r"^LENDER:\s*(.+)$",   "lender_name"),
        (r"^DOC_TYPE:\s*(.+)$", "doc_type"),
    ]:
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            meta[key] = m.group(1).strip()
    return meta


def ensure_index(pc: Pinecone) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating index '{INDEX_NAME}' (dim={EMBED_DIM}, cosine) …")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


def run() -> None:
    for key, name in [("PINECONE_API_KEY", "PINECONE_API_KEY")]:
        if not os.getenv(key):
            raise EnvironmentError(f"{name} not set in .env")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    ensure_index(pc)

    print("\nLoading embedding model (downloads once to ~/.cache/huggingface) …")
    embeddings = get_embeddings()
    print("Embedding model ready.\n")

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )

    txt_files = sorted(CORPUS_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {CORPUS_DIR}. Add corpus files first.")
        return

    total = 0
    for txt_file in txt_files:
        print(f"Processing: {txt_file.name}")
        text = txt_file.read_text(encoding="utf-8")
        meta = _parse_header(text, txt_file.name)
        docs = chunk_text(text, meta)
        print(f"  {len(docs)} chunks")

        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i: i + BATCH_SIZE]
            vectorstore.add_documents(batch)
            print(f"  Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} docs)")

        total += len(docs)

    print(f"\nDone — {total} chunks in namespace '{NAMESPACE}'.")


if __name__ == "__main__":
    run()
