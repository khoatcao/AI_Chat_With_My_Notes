"""
Step 2 Demo — Run and compare all three chunking strategies.

  docker compose run --rm app python step02_chunking.py
"""

from llama_index.core import SimpleDirectoryReader
from src.chunking import (
    fixed_size_chunks,
    sentence_chunks,
    semantic_chunks,
    summarize_chunks,
)

# ── 1. Load documents ────────────────────────────────────────────────────────
# SimpleDirectoryReader walks a folder and returns a list of Document objects.
# Each Document holds the raw text + metadata (filename, path, etc.).
documents = SimpleDirectoryReader("data/sample").load_data()

print(f"Loaded {len(documents)} document(s)")
for doc in documents:
    print(f"  • {doc.metadata.get('file_name', 'unknown')} — {len(doc.text.split())} words")

# ── 2. Fixed-size chunking ───────────────────────────────────────────────────
# Cuts every 512 tokens. Fast. May split sentences awkwardly.
fixed_nodes = fixed_size_chunks(documents, chunk_size=512, chunk_overlap=64)
summarize_chunks(fixed_nodes, "Fixed-size (512 tokens, 64 overlap)")

# ── 3. Sentence-based chunking ───────────────────────────────────────────────
# Packs complete sentences until chunk_size is reached. Better coherence.
sentence_nodes = sentence_chunks(documents, chunk_size=512, chunk_overlap=64)
summarize_chunks(sentence_nodes, "Sentence-based (512 tokens, 64 overlap)")

# ── 4. Semantic chunking ─────────────────────────────────────────────────────
# Groups sentences by topic similarity. Calls the OpenAI embeddings API.
# Each sentence gets embedded, then splits happen where similarity drops.
import config
print("\n[Semantic chunking] calling OpenAI embeddings API — please wait...")
semantic_nodes = semantic_chunks(
    documents,
    embed_provider=config.EMBED_PROVIDER,
    embed_model_name=config.EMBED_MODEL,
)
summarize_chunks(semantic_nodes, "Semantic (topic-boundary splits)")

# ── 5. Side-by-side comparison ───────────────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"  Fixed-size  : {len(fixed_nodes):>3} chunks")
print(f"  Sentence    : {len(sentence_nodes):>3} chunks")
print(f"  Semantic    : {len(semantic_nodes):>3} chunks")
print()
print("KEY INSIGHT:")
print("  • Fixed-size is fastest but may cut mid-sentence.")
print("  • Sentence is the best default — readable + efficient.")
print("  • Semantic produces fewer, denser, topic-coherent chunks.")
print("  • In production: use sentence chunks with metadata tagging (Step 6).")
