"""
Step 3 — Embedding
==================
An embedding model converts a chunk of text into a dense vector (list of floats).
Similar chunks end up with vectors that are geometrically close — this is what
makes semantic search possible.

  "What is gradient descent?"  →  [0.021, -0.134, 0.872, ...]  (1536 dims)
  "How does backprop work?"    →  [0.019, -0.141, 0.863, ...]  (close!)
  "My cat knocked over a vase" →  [0.412,  0.891, -0.023, ...]  (far away)

IMPORTANT: You must use the same embedding model at index time AND query time.
Mixing models scrambles the vector space — retrieval breaks silently.
"""

import config
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding


def get_embed_model() -> BaseEmbedding:
    """
    Factory — returns the embedding model defined in config.
    Switch between providers by changing EMBED_PROVIDER in your .env.
    """
    if config.EMBED_PROVIDER == "huggingface":
        # Free, runs locally in Docker — no API calls, no cost.
        # Slower on first use (downloads model weights ~90 MB).
        return HuggingFaceEmbedding(model_name=config.EMBED_MODEL)

    # Default: OpenAI text-embedding-3-small via Vocareum proxy
    return OpenAIEmbedding(
        model=config.EMBED_MODEL,
        api_key=config.OPENAI_API_KEY,
        api_base=config.OPENAI_API_BASE,
    )
