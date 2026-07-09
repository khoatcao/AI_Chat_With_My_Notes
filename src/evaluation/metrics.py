"""
Step 9 — Retrieval Evaluation
==============================
How do you know if your retriever is actually good?
You measure it. Two essential metrics:

HIT RATE @ K
  "For each test question, is the right answer chunk in the top-K results?"
  Score = (# questions where correct chunk retrieved) / (# total questions)
  Range: 0–1. Higher is better. Simple and fast to compute.

MRR (Mean Reciprocal Rank)
  "How HIGH in the list does the correct chunk appear?"
  MRR = mean of (1 / rank_of_first_correct_chunk)
  If correct chunk is rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33, not found → 0
  Better than hit rate — penalizes correct answers buried at rank 5 vs rank 1.

CONTEXT PRECISION (RAGAS)
  "Are the retrieved chunks actually relevant to the question?"
  Uses an LLM to judge relevance — catches cases where hit rate is high
  but retrieved chunks don't actually help answer the question.
"""

from dataclasses import dataclass
from llama_index.core import VectorStoreIndex
from src.retrieval import similarity_search


@dataclass
class EvalResult:
    hit_rate: float
    mrr: float
    details: list[dict]

    def print_summary(self):
        print(f"\n{'='*50}")
        print(f"RETRIEVAL EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"  Hit Rate @ {len(self.details[0]['retrieved_sources'])} : {self.hit_rate:.3f}  "
              f"({'%.1f' % (self.hit_rate*100)}% of questions found correct chunk)")
        print(f"  MRR              : {self.mrr:.3f}  "
              f"(average rank of correct chunk: {'%.1f' % (1/self.mrr) if self.mrr > 0 else '∞'})")
        print(f"\nPer-question breakdown:")
        for d in self.details:
            hit = "✓" if d["hit"] else "✗"
            print(f"  {hit} [{d['rank_str']:>8}]  {d['question'][:60]}")


def evaluate_retrieval(
    index: VectorStoreIndex,
    qa_pairs: list[dict],
    top_k: int = 5,
) -> EvalResult:
    """
    Evaluate retrieval quality against a set of question-answer pairs.

    qa_pairs format:
      [
        {"question": "What is gradient descent?",
         "expected_source": "machine_learning_notes.md"},
        ...
      ]

    expected_source is the file_name that SHOULD appear in the top-K results.
    """
    hits = 0
    reciprocal_ranks = []
    details = []

    for pair in qa_pairs:
        question = pair["question"]
        expected = pair["expected_source"]

        nodes = similarity_search(index, question, top_k=top_k)
        sources = [n.metadata.get("file_name", "") for n in nodes]

        # Find rank of expected source (1-indexed)
        rank = None
        for i, src in enumerate(sources, start=1):
            if expected in src:
                rank = i
                break

        hit = rank is not None
        rr = (1 / rank) if rank else 0.0

        hits += int(hit)
        reciprocal_ranks.append(rr)
        details.append({
            "question": question,
            "expected_source": expected,
            "hit": hit,
            "rank": rank,
            "rank_str": f"rank {rank}" if rank else "not found",
            "retrieved_sources": sources,
        })

    return EvalResult(
        hit_rate=hits / len(qa_pairs),
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks),
        details=details,
    )


# ── Built-in test set for the sample notes ───────────────────────────────────
SAMPLE_QA_PAIRS = [
    {"question": "What is gradient descent?",
     "expected_source": "machine_learning_notes.md"},
    {"question": "What is the difference between overfitting and underfitting?",
     "expected_source": "machine_learning_notes.md"},
    {"question": "How do transformers use self-attention?",
     "expected_source": "machine_learning_notes.md"},
    {"question": "What is RAG and why use it instead of fine-tuning?",
     "expected_source": "rag_notes.md"},
    {"question": "What are common chunking strategies for RAG?",
     "expected_source": "rag_notes.md"},
    {"question": "What is hybrid search in a RAG pipeline?",
     "expected_source": "rag_notes.md"},
    {"question": "What embedding model should I use for development?",
     "expected_source": "rag_notes.md"},
]
