"""
Entry point — run this to verify your setup is working.
Each step of the RAG pipeline will be wired in here progressively.
"""
import config


def main():
    print("RAG pipeline config loaded:")
    print(f"  LLM model    : {config.LLM_MODEL}")
    print(f"  Embed model  : {config.EMBED_MODEL} ({config.EMBED_PROVIDER})")
    print(f"  ChromaDB     : {config.CHROMA_HOST}:{config.CHROMA_PORT}")
    print(f"  Chunk size   : {config.CHUNK_SIZE} tokens (overlap {config.CHUNK_OVERLAP})")
    print(f"  Top-K        : {config.TOP_K}")

    if not config.OPENAI_API_KEY:
        print("\n[!] OPENAI_API_KEY is not set — copy .env.example to .env and add your key.")
    else:
        print("\n[OK] OPENAI_API_KEY is set.")


if __name__ == "__main__":
    main()
