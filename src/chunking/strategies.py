"""
Step 2 — Chunking Strategies
=============================
WHY chunking matters: LLMs have a context window limit and embedding models
work best on short, focused passages. Chunking splits your documents into
pieces small enough to embed and retrieve individually.

THREE strategies you should know:

1. FIXED-SIZE  — split every N tokens regardless of meaning.
                 Fast, simple. Bad at cutting mid-sentence.

2. SENTENCE    — split on sentence boundaries, then group until chunk_size.
                 Respects grammar. Best default choice.

3. SEMANTIC    — embed every sentence, then split where the topic *changes*
                 (cosine similarity drops). Slow but produces the most
                 coherent chunks. Good for long documents with multiple topics.
"""

from llama_index.core import Document
from llama_index.core.node_parser import (
    TokenTextSplitter,
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def fixed_size_chunks(documents: list[Document], chunk_size: int = 512, chunk_overlap: int = 64):
    """
    Split by token count. Overlap ensures context is not lost at boundaries.

    chunk_size   — max tokens per chunk
    chunk_overlap — how many tokens from the previous chunk to repeat
                    at the start of the next one (prevents context gaps)
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.get_nodes_from_documents(documents)


def sentence_chunks(documents: list[Document], chunk_size: int = 512, chunk_overlap: int = 64):
    """
    Split on sentence boundaries then pack sentences until chunk_size is hit.
    This is the recommended default — preserves grammar and readability.

    LlamaIndex's SentenceSplitter uses a sentence tokenizer under the hood,
    falling back to newlines and then character splits if needed.
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.get_nodes_from_documents(documents)


def semantic_chunks(
    documents: list[Document],
    embed_provider: str = "openai",
    embed_model_name: str = "text-embedding-3-small",
):
    """
    Split where the *topic changes*, not at arbitrary token counts.

    Algorithm:
      1. Split text into individual sentences.
      2. Embed each sentence.
      3. Compute cosine similarity between adjacent sentence embeddings.
      4. Split at positions where similarity drops below a threshold.

    breakpoint_percentile_threshold (default 95) — only split at the top 5%
    sharpest topic changes. Lower → more splits (smaller chunks).

    embed_provider: "openai" (default) or "huggingface" (free, runs locally)
    """
    if embed_provider == "huggingface":
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    else:
        embed_model = OpenAIEmbedding(model=embed_model_name)

    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=95,
    )
    return splitter.get_nodes_from_documents(documents)


def summarize_chunks(nodes: list, strategy_name: str) -> None:
    """Print a readable summary of chunk results."""
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"Total chunks: {len(nodes)}")
    print(f"{'='*60}")
    for i, node in enumerate(nodes):
        text = node.get_content()
        words = len(text.split())
        print(f"\n[Chunk {i+1}] — {words} words")
        print(f"  {text[:200].strip()}{'...' if len(text) > 200 else ''}")
