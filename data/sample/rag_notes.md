# Retrieval-Augmented Generation (RAG)

## What is RAG?
RAG is a test technique that combines a retrieval system with a language model. Instead of relying purely on what the LLM memorized during training, RAG fetches relevant documents at query time and injects them into the prompt as context.

The core idea: the LLM's knowledge is static (frozen at training cutoff), but your document store can be updated at any time without retraining the model.

## Why RAG instead of fine-tuning?
- **Cost**: fine-tuning requires GPU time and labeled data. RAG only needs document ingestion.
- **Freshness**: your data stays up-to-date — just re-index new documents.
- **Transparency**: you can show which source passages the answer came from.
- **Hallucination reduction**: grounding the model in real retrieved text reduces made-up answers.

## The RAG Pipeline

### Step 1 — Ingestion (offline)
1. Load documents (PDFs, Markdown, web pages)
2. Chunk documents into smaller passages
3. Embed each chunk into a dense vector
4. Store vectors in a vector database with metadata

### Step 2 — Retrieval (online, at query time)
1. Embed the user's query using the same embedding model
2. Run similarity search against the vector database
3. Return the top-K most similar chunks
4. (Optional) Re-rank the results with a cross-encoder

### Step 3 — Generation
1. Insert retrieved chunks into the prompt as context
2. Ask the LLM to answer the query grounded in the context
3. Return the answer (optionally with source citations)

## Chunking
Chunking is the most underrated step in RAG. The quality of your chunks directly determines retrieval quality.

Common strategies:
- **Fixed-size**: split every N tokens. Fast, simple, ignores sentence boundaries.
- **Sentence-based**: split on sentence boundaries, then pack until size limit. Best default.
- **Semantic**: embed sentences and split where topic similarity drops. Most coherent but slowest.

Rule of thumb: chunk_size should match the granularity of the questions users will ask. For FAQ-style Q&A, small chunks (128–256 tokens) work well. For multi-paragraph reasoning, larger chunks (512–1024) are better.

## Embedding Models
The embedding model maps text to a dense vector in high-dimensional space. Similar texts cluster together.

- **`BAAI/bge-small-en-v1.5`**: small, fast, runs on CPU. Good for development.
- **`BAAI/bge-large-en-v1.5`**: more accurate, needs more RAM.
- **OpenAI `text-embedding-3-small`**: strong quality, API cost per call.

Always use the **same embedding model** for indexing and querying — mixing models will give garbage retrieval results.

## Vector Databases
A vector database stores embeddings and enables fast approximate nearest-neighbor (ANN) search.

- **ChromaDB**: local, no server needed, perfect for development.
- **Pinecone**: managed cloud, scales to billions of vectors.
- **Weaviate**: open-source, supports hybrid search natively.
- **pgvector**: Postgres extension — good if you already use Postgres.

## Hybrid Search
Combines dense (semantic) retrieval with sparse (keyword) retrieval.

- Dense: embedding similarity — good at semantic matching ("what makes a neural network overfit?")
- Sparse (BM25): keyword matching — good at exact matches ("learning rate 0.001")
- Hybrid: weighted combination of both scores, then re-rank

Hybrid search almost always outperforms pure vector search in production.

## Common Failure Modes
1. **Chunks too large**: retrieved passage contains the answer buried in noise.
2. **Chunks too small**: not enough context for the LLM to form a good answer.
3. **Wrong embedding model**: domain mismatch (e.g., a general model on medical text).
4. **Missing metadata**: can't filter by date, author, or topic at retrieval time.
5. **No re-ranking**: top-K by cosine similarity isn't always the most useful order.
