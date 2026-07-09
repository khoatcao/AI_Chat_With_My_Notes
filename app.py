"""
AI Chat with My Notes — main chat interface.

Usage:
  # First time: ingest your notes
  docker compose run --rm app python step03_ingest.py

  # Then start chatting
  docker compose run --rm app python app.py

  # Or put your own notes in data/notes/ and ingest those:
  docker compose run --rm app python step03_ingest.py
"""

import os
import sys
import warnings
import logging

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
for logger_name in [
    "bm25s", "sentence_transformers", "transformers",
    "huggingface_hub", "huggingface_hub.utils._headers",
    "llama_index.retrievers.bm25", "llama_index",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import config
from llama_index.core import SimpleDirectoryReader
from src.chunking import sentence_chunks
from src.indexing import build_index, load_index, get_chroma_collection
from src.pipeline import query


def load_or_build_index():
    """Load existing index from ChromaDB, or build it if empty."""
    collection = get_chroma_collection()
    if collection.count() > 0:
        print(f"Loading existing index ({collection.count()} chunks from ChromaDB)...")
        return load_index(), None  # nodes not needed if already indexed
    else:
        print("No index found. Ingesting notes...")
        notes_dir = config.NOTES_DIR if os.path.exists(config.NOTES_DIR) else "data/sample"
        documents = SimpleDirectoryReader(notes_dir).load_data()
        nodes = sentence_chunks(documents, chunk_size=config.CHUNK_SIZE,
                                chunk_overlap=config.CHUNK_OVERLAP)
        print(f"Indexing {len(nodes)} chunks...")
        index = build_index(nodes)
        return index, nodes


def get_all_nodes_from_collection():
    """Fetch stored node text from ChromaDB for BM25 (hybrid search needs raw text)."""
    from llama_index.core.schema import TextNode
    collection = get_chroma_collection()
    results = collection.get(include=["documents", "metadatas"])
    nodes = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        nodes.append(TextNode(text=doc, metadata=meta or {}))
    return nodes


def main():
    if not config.OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    print("=" * 60)
    print("  AI Chat with My Notes")
    print("=" * 60)

    index, _ = load_or_build_index()
    nodes = get_all_nodes_from_collection()
    print(f"Ready. {len(nodes)} chunks indexed.")
    print("Type your question (or 'quit' to exit, 'sources' to toggle source display)\n")

    show_sources = True

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if user_input.lower() == "sources":
            show_sources = not show_sources
            print(f"Source display: {'ON' if show_sources else 'OFF'}\n")
            continue

        result = query(index, nodes, user_input)

        print(f"\nAssistant: {result['answer']}\n")

        if show_sources:
            # Only show sources with score > -5 (cross-encoder threshold for relevance)
            relevant = [s for s in result["sources"] if s["score"] > -5]
            if relevant:
                print("Sources used:")
                for i, src in enumerate(relevant, start=1):
                    print(f"  [{i}] {src['file']}")
                    print(f"       {src['text'][:120]}...")
            print()


if __name__ == "__main__":
    main()
