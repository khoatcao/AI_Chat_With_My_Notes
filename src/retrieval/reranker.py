"""
Step 8 — Reranking
===================
Retrieval (steps 5–7) uses EMBEDDING similarity — fast but approximate.
Reranking uses a CROSS-ENCODER — slower but far more accurate.

Bi-encoder (retrieval):  embed(query) · embed(chunk)   — independent vectors
Cross-encoder (reranking): model(query + chunk together) — reads both at once

Why two stages?
  You can't run a cross-encoder on millions of chunks — too slow.
  Solution: retriever fetches top-50 candidates cheaply,
            cross-encoder picks the best 5 precisely.

This is called the "retrieve-then-rerank" pattern.
"""

import config
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore


# cross-MiniLM is small (~80 MB), runs on CPU, free — good for dev.
# For production consider: cross-encoder/ms-marco-MiniLM-L-12-v2 (more accurate)
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(
    nodes: list[NodeWithScore],
    query: str,
    top_n: int = config.TOP_K,
) -> list[NodeWithScore]:
    """
    Reorder nodes using a cross-encoder that reads query+chunk together.
    Returns only top_n results — typically smaller than the retrieval top_k.

    Pattern: retrieve top_k=20, rerank, keep top_n=5.
    The cross-encoder score replaces the original cosine similarity score.
    """
    reranker = SentenceTransformerRerank(model=_CROSS_ENCODER_MODEL, top_n=top_n)
    # postprocessor expects a QueryBundle
    from llama_index.core.schema import QueryBundle
    return reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=query))
