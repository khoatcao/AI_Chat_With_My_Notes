"""
Step 9 Demo — Retrieval Evaluation.

  docker compose run --rm app python step09_evaluation.py
"""

from src.indexing import load_index
from src.evaluation import evaluate_retrieval, SAMPLE_QA_PAIRS

print("Loading index from ChromaDB...")
index = load_index()

print(f"Running evaluation on {len(SAMPLE_QA_PAIRS)} test questions...")
result = evaluate_retrieval(index, SAMPLE_QA_PAIRS, top_k=5)
result.print_summary()

print("\nWHAT THESE NUMBERS MEAN:")
print("  Hit Rate > 0.8 → your retrieval is good enough for production")
print("  MRR > 0.7      → correct chunks are consistently ranking in top 2")
print("  If scores are low: try smaller chunk_size, more overlap, or hybrid search")
