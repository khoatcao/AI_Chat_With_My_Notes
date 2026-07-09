"""
Step 3+4 Demo — Chunk → Embed → Store in ChromaDB (full ingestion pipeline).

  docker compose up -d chromadb          # start ChromaDB first
  docker compose run --rm app python step03_ingest.py

After this script runs, your notes are indexed and ready for retrieval (Step 5).
"""

from llama_index.core import SimpleDirectoryReader
from src.chunking import sentence_chunks
from src.indexing import build_index, load_index, get_chroma_collection
import config

# ── 1. Clear existing index so re-running never creates duplicates ────────────
import chromadb as _chromadb
_client = _chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
try:
    _client.delete_collection(config.CHROMA_COLLECTION)
    print(f"Cleared existing collection '{config.CHROMA_COLLECTION}'.")
except Exception:
    pass

# ── 2. Load raw documents ─────────────────────────────────────────────────────
import os
notes_has_files = os.path.exists(config.NOTES_DIR) and any(os.scandir(config.NOTES_DIR))
notes_dir = config.NOTES_DIR if notes_has_files else "data/sample"
print(f"Loading documents from '{notes_dir}'...")
documents = SimpleDirectoryReader(notes_dir, recursive=True).load_data()
print(f"  Loaded {len(documents)} document(s)")
for doc in documents:
    print(f"  • {doc.metadata.get('file_name', 'unknown')} — {len(doc.text.split())} words")

# ── 2. Chunk into nodes ───────────────────────────────────────────────────────
# We use sentence-based chunking — best default for note-taking content.
# Each node = one chunk of text + its metadata (file name, position, etc.)
print(f"\nChunking (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})...")
nodes = sentence_chunks(documents, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
print(f"  Produced {len(nodes)} chunks")

# ── 3. Inspect a chunk before embedding ──────────────────────────────────────
# This shows you EXACTLY what gets embedded — the text the vector represents.
print(f"\nSample chunk (what the embedding model sees):")
print(f"  {'-'*56}")
sample = nodes[0].get_content()
print(f"  {sample[:300].strip()}{'...' if len(sample) > 300 else ''}")
print(f"  {'-'*56}")
print(f"  Metadata: {nodes[0].metadata}")

# ── 4. Embed + store in ChromaDB ─────────────────────────────────────────────
# build_index() calls the embedding model once per chunk, then writes:
#   - the vector (list of floats) to ChromaDB
#   - the original text + metadata alongside it
# Progress bar shows which chunks are being embedded.
print(f"\nEmbedding {len(nodes)} chunks with {config.EMBED_MODEL}")
print(f"Storing in ChromaDB at {config.CHROMA_HOST}:{config.CHROMA_PORT}...")
index = build_index(nodes)
print("  Done.")

# ── 5. Verify what's in ChromaDB ─────────────────────────────────────────────
collection = get_chroma_collection()
count = collection.count()
print(f"\nChromaDB collection '{config.CHROMA_COLLECTION}' now holds {count} vectors.")

# ── 6. Peek at a raw vector ───────────────────────────────────────────────────
# This shows you what an embedding actually looks like — a long list of floats.
# The model maps meaning to geometry: similar texts → similar vectors.
result = collection.get(limit=1, include=["embeddings", "documents", "metadatas"])
if result["embeddings"] is not None and len(result["embeddings"]) > 0:
    vec = result["embeddings"][0]
    print(f"\nRaw vector for first chunk:")
    print(f"  Dimensions : {len(vec)}")
    print(f"  First 8 values: {[round(v, 4) for v in vec[:8]]}")
    print(f"  Text preview: {result['documents'][0][:100]}...")

print("\nIngestion complete. Run step05_retrieval.py next to query this index.")
