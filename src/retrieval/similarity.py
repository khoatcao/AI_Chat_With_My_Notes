"""
Step 5 — Similarity Search
===========================
At query time:
  1. Embed the user's question using the same model used at index time.
  2. Compute cosine similarity between the query vector and every stored vector.
  3. Return the top-K most similar chunks.

Cosine similarity measures the ANGLE between two vectors (not distance).
Score of 1.0 = identical direction = semantically identical text.
Score of 0.0 = perpendicular = completely unrelated.
"""

import config
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore


def similarity_search(
    index: VectorStoreIndex,
    query: str,
    top_k: int = config.TOP_K,
) -> list[NodeWithScore]:
    """
    Returns the top_k chunks most semantically similar to the query.
    Each NodeWithScore wraps the chunk text + metadata + a similarity score.
    """
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    return retriever.retrieve(query)


def print_results(nodes: list[NodeWithScore], query: str) -> None:
    print(f"\nQuery : {query}")
    print(f"{'─'*60}")
    for i, node in enumerate(nodes):
        score = node.score or 0.0
        text = node.get_content()
        source = node.metadata.get("file_name", "unknown")
        print(f"\n[{i+1}] score={score:.4f}  source={source}")
        print(f"     {text[:250].strip()}{'...' if len(text) > 250 else ''}")
