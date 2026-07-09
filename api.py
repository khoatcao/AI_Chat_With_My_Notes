import os, warnings, logging, shutil
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
for _name in ["bm25s","sentence_transformers","transformers",
               "huggingface_hub","llama_index"]:
    logging.getLogger(_name).setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import TextNode

import config
from src.chunking import sentence_chunks
from src.embedding import get_embed_model
from src.indexing import load_index, get_chroma_collection, build_index
from src.pipeline import query as rag_query
from src.retrieval import search_with_filter, search_with_multi_filter

NOTES_DIR = Path("data/notes")
NOTES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}

app = FastAPI(title="AI Chat with My Notes")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_index = None
_nodes = []


def _reload_index():
    global _index, _nodes
    collection = get_chroma_collection()
    if collection.count() == 0:
        _index = None
        _nodes = []
        return
    _index = load_index()
    results = collection.get(include=["documents", "metadatas"])
    _nodes = [
        TextNode(text=d, metadata=m or {})
        for d, m in zip(results["documents"], results["metadatas"])
    ]


@app.on_event("startup")
async def startup():
    # Try sample notes if data/notes is empty
    if not any(NOTES_DIR.iterdir()) if NOTES_DIR.exists() else True:
        sample = Path("data/sample")
        if sample.exists():
            for f in sample.glob("*"):
                shutil.copy(f, NOTES_DIR / f.name)
    _reload_index()
    print(f"[API] Ready — {len(_nodes)} chunks loaded.")


# ── Documents ────────────────────────────────────────────────────────────────

@app.get("/documents")
async def list_documents():
    """List all indexed documents with chunk counts."""
    collection = get_chroma_collection()
    results = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in results["metadatas"]:
        name = (meta or {}).get("file_name", "unknown")
        counts[name] = counts.get(name, 0) + 1
    docs = [{"file": k, "chunks": v} for k, v in sorted(counts.items())]
    return {"documents": docs, "total_chunks": sum(counts.values())}


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload one or more files and index them incrementally."""
    uploaded = []
    collection = get_chroma_collection()

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"{upload.filename}: unsupported type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        dest = NOTES_DIR / upload.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)

        # Remove old chunks for this file (incremental update)
        try:
            existing = collection.get(where={"file_name": upload.filename})
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

        # Chunk + embed + store just this file
        docs = SimpleDirectoryReader(input_files=[str(dest)]).load_data()
        nodes = sentence_chunks(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        build_index(nodes, clear_first=False)   # incremental — don't clear

        uploaded.append({"file": upload.filename, "chunks": len(nodes)})

    _reload_index()
    return {"uploaded": uploaded, "total_chunks": len(_nodes)}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Remove a document from the index and disk."""
    collection = get_chroma_collection()
    try:
        existing = collection.get(where={"file_name": filename})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    path = NOTES_DIR / filename
    if path.exists():
        path.unlink()

    _reload_index()
    return {"deleted": filename, "total_chunks": len(_nodes)}


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    selected_files: list[str] = []   # empty = search all files


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if _index is None:
        raise HTTPException(status_code=503, detail="No documents indexed yet. Upload files first.")

    # Filter nodes for BM25 if specific files selected
    nodes_for_bm25 = _nodes
    if req.selected_files:
        nodes_for_bm25 = [
            n for n in _nodes
            if n.metadata.get("file_name") in req.selected_files
        ]
        if not nodes_for_bm25:
            raise HTTPException(status_code=400, detail="Selected files not found in index.")

    result = rag_query(
        _index,
        nodes_for_bm25,
        req.question,
        file_filter=req.selected_files if req.selected_files else None,
    )

    seen, relevant_sources = set(), []
    for s in sorted(result["sources"], key=lambda x: float(x["score"]), reverse=True):
        snippet = s["text"][:80]
        if snippet not in seen and float(s["score"]) > -5:
            seen.add(snippet)
            relevant_sources.append({**s, "score": float(s["score"])})
        if len(relevant_sources) >= 3:
            break

    return {"answer": result["answer"], "sources": relevant_sources}


@app.get("/health")
async def health():
    return {"status": "ok", "chunks": len(_nodes)}
