"""
Step 6 — Metadata Filtering
============================
Similarity search finds semantically relevant chunks — but you often want to
NARROW results before or after: "only search my ML notes", "only pages from
this week", "only chunks tagged as definitions".

Metadata is stored alongside every vector in ChromaDB and can be filtered
WITHOUT re-embedding anything. This is fast and cheap.

Common metadata fields LlamaIndex adds automatically:
  file_name  — "machine_learning_notes.md"
  file_path  — "/app/data/sample/machine_learning_notes.md"
  file_type  — "text/markdown"
"""

import config
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore


def search_with_filter(
    index: VectorStoreIndex,
    query: str,
    file_name: str | None = None,
    top_k: int = config.TOP_K,
) -> list[NodeWithScore]:
    """
    Similarity search restricted to chunks from a specific file.

    FilterOperator.EQ  — exact match
    FilterCondition.AND — all filters must match (use OR for any-of)
    """
    filters = None
    if file_name:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="file_name",
                    value=file_name,
                    operator=FilterOperator.EQ,
                )
            ],
            condition=FilterCondition.AND,
        )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        filters=filters,
    )
    return retriever.retrieve(query)


def search_with_multi_filter(
    index: VectorStoreIndex,
    query: str,
    filters: list[dict],
    condition: str = "and",
    top_k: int = config.TOP_K,
) -> list[NodeWithScore]:
    """
    Search with arbitrary metadata filters.

    filters = [
        {"key": "file_name", "value": "rag_notes.md", "operator": "=="},
        {"key": "file_type", "value": "text/markdown",  "operator": "=="},
    ]
    """
    op_map = {"==": FilterOperator.EQ, "!=": FilterOperator.NE,
              ">": FilterOperator.GT,  ">=": FilterOperator.GTE,
              "<": FilterOperator.LT,  "<=": FilterOperator.LTE}
    cond = FilterCondition.AND if condition == "and" else FilterCondition.OR

    metadata_filters = MetadataFilters(
        filters=[
            MetadataFilter(key=f["key"], value=f["value"],
                           operator=op_map.get(f.get("operator", "=="), FilterOperator.EQ))
            for f in filters
        ],
        condition=cond,
    )
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k,
                                     filters=metadata_filters)
    return retriever.retrieve(query)
