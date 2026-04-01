"""
retriever.py
------------
Builds Pinecone retrievers using free HuggingFace embeddings.

Embedding model: sentence-transformers/all-MiniLM-L6-v2
  - Runs locally on CPU (no API calls, no cost)
  - 384-dimensional vectors
  - Fast: ~50ms per batch on CPU
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM   = 384   # must match Pinecone index dimension
TOP_K       = 5


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the local embedding model.
    First call downloads the model (~90 MB) to ~/.cache/huggingface.
    All subsequent calls use the local cache — no network needed.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_retriever(namespace: str = "policy-corpus"):
    """
    Return a LangChain VectorStoreRetriever backed by Pinecone.

    Args:
        namespace: 'policy-corpus' for Mode A, session UUID for Mode B.
    """
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "loan-assistant"),
        embedding=get_embeddings(),
        namespace=namespace,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


def build_vectorstore(namespace: str) -> PineconeVectorStore:
    """Return raw vectorstore (needed for upsert + delete in Mode B)."""
    return PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "loan-assistant"),
        embedding=get_embeddings(),
        namespace=namespace,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )


def delete_namespace(namespace: str) -> None:
    """Delete all vectors in a Pinecone namespace (Mode B cleanup)."""
    from pinecone import Pinecone
    pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "loan-assistant"))
    index.delete(delete_all=True, namespace=namespace)
