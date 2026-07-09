import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openai.vocareum.com/v1")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# Embedding — "openai" uses text-embedding-3-small (paid)
#              "huggingface" uses BAAI/bge-small-en-v1.5 (free, local)
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5

NOTES_DIR = "data/notes"
CHROMA_COLLECTION = "my_notes"
