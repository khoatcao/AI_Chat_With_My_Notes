# AI Chat with My Notes

A RAG (Retrieval-Augmented Generation) application that lets you chat with your own documents. Upload notes, PDFs, Word files, or Markdown files and ask questions — answers are grounded in your content, not the internet.

![UI Screenshot](image.png)

---

## Features

- **Document upload** — supports `.md`, `.txt`, `.pdf`, and `.docx` files
- **Hybrid search** — combines dense vector search (embeddings) with BM25 keyword search for better retrieval
- **Cross-encoder reranking** — reranks retrieved chunks before sending to the LLM
- **Selective search** — filter queries to one or more specific documents
- **Incremental indexing** — upload new files without re-indexing everything
- **Source citations** — every answer shows which document chunks were used

---

## Architecture

```
Frontend (React/Vite)  →  FastAPI backend  →  ChromaDB (vector store)
                                          →  OpenAI (LLM + embeddings)
                                          →  BM25 index (keyword search)
```

- **Embeddings**: OpenAI `text-embedding-3-small` (default) or local HuggingFace `BAAI/bge-small-en-v1.5`
- **LLM**: OpenAI `gpt-4o` (configurable)
- **Vector store**: ChromaDB (runs as a separate Docker service)
- **Chunking**: Sentence-aware chunking with configurable size and overlap

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- An OpenAI API key

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 2. Start the stack

```bash
docker compose up
```

This starts three services:
| Service   | URL                        | Description              |
|-----------|----------------------------|--------------------------|
| Frontend  | http://localhost:5173      | React chat UI            |
| API       | http://localhost:8001      | FastAPI backend          |
| ChromaDB  | http://localhost:8000      | Vector database          |

### 3. Chat

Open http://localhost:5173, upload your documents via the sidebar, and start asking questions.

---

## Configuration

All settings are controlled via environment variables in `.env`:

| Variable          | Default                          | Description                              |
|-------------------|----------------------------------|------------------------------------------|
| `OPENAI_API_KEY`  | *(required)*                     | Your OpenAI API key                      |
| `OPENAI_API_BASE` | `https://openai.vocareum.com/v1` | API base URL (change for Azure/proxy)    |
| `EMBED_PROVIDER`  | `openai`                         | `openai` or `huggingface` (free, local)  |
| `EMBED_MODEL`     | `text-embedding-3-small`         | Embedding model name                     |
| `LLM_MODEL`       | `gpt-4o`                         | LLM model name                           |
| `CHROMA_HOST`     | `localhost`                      | ChromaDB host                            |
| `CHROMA_PORT`     | `8000`                           | ChromaDB port                            |

---

## Jupyter Notebooks

For experimentation and evaluation, start the notebook service:

```bash
docker compose --profile notebook up jupyter
```

Then open http://localhost:8889.

---

## Tech Stack

- **Frontend**: React, Vite
- **Backend**: FastAPI, Python
- **RAG framework**: LlamaIndex
- **Vector store**: ChromaDB
- **Embeddings**: OpenAI / HuggingFace (sentence-transformers)
- **LLM**: OpenAI GPT-4o
- **Evaluation**: RAGAS
