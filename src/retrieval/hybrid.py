"""
Step 7 — Hybrid Search
========================
Pure vector search misses exact keyword matches.
Pure BM25 (keyword) misses semantic meaning.
Hybrid = both, merged with Reciprocal Rank Fusion (RRF).

Example where each shines:
  Vector: "what makes a model overfit?" → finds "generalization" chunks
  BM25:   "learning rate 0.001"         → finds exact string matches
  Hybrid: gets the best of both

RRF formula: score = Σ 1 / (rank_i + 60)
  Each retriever contributes a rank; higher-ranked results score higher.
  The constant 60 dampens the impact of very-top results.
"""

import config
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever


def hybrid_search(
    index: VectorStoreIndex,
    nodes: list[BaseNode],
    query: str,
    top_k: int = config.TOP_K,
    vector_weight: float = 0.6,
) -> list[NodeWithScore]:
    """
    Combine BM25 keyword search with vector similarity search.

    nodes       — the same node list used to build the index (BM25 needs raw text)
    vector_weight — how much to weight vector vs BM25 results (0.6 = 60% vector)
    """
    vector_retriever = index.as_retriever(similarity_top_k=top_k)

    # BM25 retriever works on raw text nodes — no embedding needed.
    # It tokenizes text and scores by term frequency / inverse document frequency.
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    # QueryFusionRetriever runs both retrievers and merges with RRF.
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=1,          # 1 = no query expansion (we'll cover that in production step)
        mode="reciprocal_rerank",
        retriever_weights=[vector_weight, 1 - vector_weight],
        use_async=False,
    )
    return hybrid_retriever.retrieve(query)
