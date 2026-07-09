"""
Step 4 — Indexing into ChromaDB
================================
After embedding, each chunk becomes a vector. We store those vectors in
ChromaDB — a vector database built for fast similarity search.

What ChromaDB stores per chunk:
  ┌────────────────────────────────────────────────────────┐
  │  id        : unique chunk identifier                    │
  │  embedding : [0.021, -0.134, 0.872, ...]  (the vector) │
  │  document  : the original chunk text                    │
  │  metadata  : {file_name, page_label, chunk_index, ...}  │
  └────────────────────────────────────────────────────────┘

At query time ChromaDB does Approximate Nearest Neighbor (ANN) search —
it finds the vectors closest to your query vector in milliseconds,
even with millions of stored chunks.
"""

import chromadb
import config
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.embedding import get_embed_model


def get_chroma_collection(collection_name: str = config.CHROMA_COLLECTION):
    """Connect to ChromaDB and return (or create) a named collection."""
    client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
    return client.get_or_create_collection(collection_name)


def build_index(
    nodes: list[BaseNode],
    collection_name: str = config.CHROMA_COLLECTION,
    clear_first: bool = True,
) -> VectorStoreIndex:
    """
    Embed all nodes and store them in ChromaDB.

    clear_first=True  — wipe the collection before indexing (full re-index)
    clear_first=False — add nodes to existing collection (incremental upload)
    """
    if clear_first:
        client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = get_chroma_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embed_model()

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    return index


def load_index(collection_name: str = config.CHROMA_COLLECTION) -> VectorStoreIndex:
    """
    Load an already-built index from ChromaDB — no re-embedding needed.
    Use this after the first ingest to avoid paying embedding costs twice.
    """
    collection = get_chroma_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embed_model()

    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
