"""
Step 10 — Production RAG Pipeline
===================================
Wires all previous steps into a single query() call:

  query
    → hybrid search (BM25 + vector, top_k=20)
    → rerank        (cross-encoder, keep top_n=5)
    → build prompt  (retrieved chunks injected as context)
    → LLM generate  (GPT-4o streams the answer)
    → return answer + source citations
"""

import config
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.llms.openai import OpenAI
from src.retrieval import hybrid_search, rerank


_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only
the provided context from the user's notes. If the context does not contain
enough information to answer the question, say so clearly.
Always cite which note your answer comes from."""


def build_llm() -> OpenAI:
    return OpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        api_base=config.OPENAI_API_BASE,
        temperature=0.1,
        max_tokens=1024,
    )


def query(
    index: VectorStoreIndex,
    nodes: list[BaseNode],
    question: str,
    retrieve_top_k: int = 20,
    rerank_top_n: int = 5,
    file_filter: list[str] | None = None,
) -> dict:
    """
    Full RAG pipeline: hybrid retrieve → rerank → LLM generate.

    Returns:
      {
        "answer"  : str,
        "sources" : [{"file": ..., "score": ..., "text": ...}],
        "question": str,
      }
    """
    # 1. Retrieve — apply metadata filter if specific files selected
    from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator, FilterCondition
    filters = None
    if file_filter:
        filters = MetadataFilters(
            filters=[MetadataFilter(key="file_name", value=f, operator=FilterOperator.EQ)
                     for f in file_filter],
            condition=FilterCondition.OR,
        )
    candidates = hybrid_search(index, nodes, question, top_k=min(retrieve_top_k, len(nodes)))

    # 2. Rerank — cross-encoder picks the best 5 from those 20
    top_nodes = rerank(candidates, question, top_n=rerank_top_n)

    # 3. Build context block from top chunks
    context_parts = []
    for i, node in enumerate(top_nodes, start=1):
        source = node.metadata.get("file_name", "unknown")
        context_parts.append(f"[Source {i}: {source}]\n{node.get_content()}")
    context = "\n\n---\n\n".join(context_parts)

    # 4. Compose prompt
    prompt = f"""{_SYSTEM_PROMPT}

CONTEXT FROM NOTES:
{context}

QUESTION: {question}

ANSWER:"""

    # 5. Generate with GPT-4o
    llm = build_llm()
    response = llm.complete(prompt)
    answer = response.text.strip()

    # 6. Build source citations
    sources = [
        {
            "file": n.metadata.get("file_name", "unknown"),
            "score": round(n.score or 0.0, 4),
            "text": n.get_content()[:200],
        }
        for n in top_nodes
    ]

    return {"answer": answer, "sources": sources, "question": question}
