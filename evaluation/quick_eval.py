# evaluation/quick_eval.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.retrieval.dense_retriever  import DenseRetriever
from backend.retrieval.sparse_retriever import SparseRetriever
from backend.retrieval.hybrid_fusion    import HybridRetriever, reciprocal_rank_fusion

# ── Test questions with known relevant pages ──────────────────
# These are questions where we KNOW which page has the answer
TEST_CASES = [
    {
        "question": "What is the maximum fine for a GDPR violation?",
        "relevant_pages": [83, 28]
    },
    {
        "question": "What is the right to erasure under GDPR?",
        "relevant_pages": [43, 13, 14]
    },
    {
        "question": "What are the obligations of a data controller?",
        "relevant_pages": [43, 44, 45]
    },
    {
        "question": "What is the right to data portability?",
        "relevant_pages": [45, 44]
    },
    {
        "question": "When must a data breach be reported?",
        "relevant_pages": [52, 17]
    },
    {
        "question": "What is the lawful basis for processing personal data?",
        "relevant_pages": [13, 14, 36]
    },
    {
        "question": "What are the rights of data subjects under GDPR?",
        "relevant_pages": [43, 44, 45, 13]
    },
    {
        "question": "What is a data processing agreement?",
        "relevant_pages": [56, 57, 30]
    },
    {
        "question": "What is the role of a data protection officer?",
        "relevant_pages": [58, 59, 19]
    },
    {
        "question": "What are the conditions for consent under GDPR?",
        "relevant_pages": [36, 37, 13]
    }
]

def recall_at_k(results: list[dict], relevant_pages: list[int], k: int = 5) -> float:
    """
    Recall@K — did we retrieve at least one relevant page in top K?
    1.0 = yes, 0.0 = no
    """
    retrieved_pages = [
        r["metadata"].get("page_number", -1)
        for r in results[:k]
    ]
    for page in relevant_pages:
        if page in retrieved_pages:
            return 1.0
    return 0.0


def run_evaluation():
    print("="*60)
    print("RETRIEVAL EVALUATION")
    print("="*60)

    # initialize retrievers
    dense  = DenseRetriever()
    sparse = SparseRetriever()
    hybrid = HybridRetriever()

    dense_scores  = []
    sparse_scores = []
    hybrid_scores = []

    for i, test in enumerate(TEST_CASES, 1):
        question      = test["question"]
        relevant_pages = test["relevant_pages"]

        print(f"\n[{i}/{len(TEST_CASES)}] {question[:50]}...")

        # dense only
        dense_results  = dense.retrieve(question, top_k=5)
        dense_score    = recall_at_k(dense_results, relevant_pages)
        dense_scores.append(dense_score)

        # sparse only
        sparse_results = sparse.retrieve(question, top_k=5)
        sparse_score   = recall_at_k(sparse_results, relevant_pages)
        sparse_scores.append(sparse_score)

        # hybrid
        hybrid_results = hybrid.retrieve(question, top_k=5)
        hybrid_score   = recall_at_k(hybrid_results, relevant_pages)
        hybrid_scores.append(hybrid_score)

        print(f"  Dense: {dense_score} | Sparse: {sparse_score} | Hybrid: {hybrid_score}")

    # calculate averages
    dense_recall  = round(sum(dense_scores)  / len(dense_scores)  * 100, 1)
    sparse_recall = round(sum(sparse_scores) / len(sparse_scores) * 100, 1)
    hybrid_recall = round(sum(hybrid_scores) / len(hybrid_scores) * 100, 1)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Dense only  (vector) : Recall@5 = {dense_recall}%")
    print(f"Sparse only (BM25)   : Recall@5 = {sparse_recall}%")
    print(f"Hybrid RRF           : Recall@5 = {hybrid_recall}%")
    print(f"\nImprovement over dense : +{round(hybrid_recall - dense_recall, 1)}%")
    print(f"Improvement over sparse: +{round(hybrid_recall - sparse_recall, 1)}%")
    print(f"{'='*60}")

    return {
        "dense":  dense_recall,
        "sparse": sparse_recall,
        "hybrid": hybrid_recall
    }


if __name__ == "__main__":
    results = run_evaluation()