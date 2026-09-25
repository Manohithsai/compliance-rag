# backend/retrieval/hybrid_fusion.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.retrieval.dense_retriever  import DenseRetriever
from backend.retrieval.sparse_retriever import SparseRetriever


def reciprocal_rank_fusion(
    dense_results:  list[dict],
    sparse_results: list[dict],
    k:              int = 60,
    top_n:          int = 10
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) — combines two ranked lists
    into one without needing to tune any weights.

    WHY RRF OVER SIMPLE SCORE COMBINATION?

    Problem with adding scores directly:
    - Dense scores are cosine similarities (0.0 to 1.0)
    - BM25 scores are term frequencies (can be 0 to 20+)
    - They're on completely different scales
    - Adding them directly gives BM25 unfair dominance

    RRF solution:
    - Ignore raw scores entirely
    - Only use RANK POSITION (1st, 2nd, 3rd...)
    - Each result gets score = 1 / (k + rank)
    - k=60 is the standard value from the original RRF paper
    - Results appearing high in BOTH lists score very high
    - Results appearing in only one list score moderately

    This is parameter-free — no weights to tune.
    Works well across completely different retrieval systems.

    FORMULA: RRF(d) = Σ 1/(k + rank(d))
    """
    # map text → RRF score + metadata
    fused_scores = {}
    doc_data     = {}

    # process dense results
    for rank, result in enumerate(dense_results, start=1):
        text = result["text"]
        rrf_score = 1 / (k + rank)

        if text not in fused_scores:
            fused_scores[text] = 0
            doc_data[text]     = result

        fused_scores[text] += rrf_score

    # process sparse results
    for rank, result in enumerate(sparse_results, start=1):
        text = result["text"]
        rrf_score = 1 / (k + rank)

        if text not in fused_scores:
            fused_scores[text] = 0
            doc_data[text]     = result

        fused_scores[text] += rrf_score

    # sort by fused score descending
    sorted_texts = sorted(
        fused_scores.keys(),
        key     = lambda t: fused_scores[t],
        reverse = True
    )[:top_n]

    # build final results
    final_results = []
    for text in sorted_texts:
        result = doc_data[text].copy()
        result["rrf_score"] = round(fused_scores[text], 6)
        result["source"]    = "hybrid_rrf"
        final_results.append(result)

    return final_results


class HybridRetriever:
    """
    Combines DenseRetriever + SparseRetriever using RRF fusion.

    This is the main retriever used by the rest of the pipeline.
    Everything downstream (reranker, agent, generator) calls this.
    """

    def __init__(self):
        print("[HybridRetriever] Initializing...")
        self.dense  = DenseRetriever()
        self.sparse = SparseRetriever()
        print("[HybridRetriever] Ready.\n")

    def retrieve(
        self,
        query:          str,
        top_k:          int  = 10,
        dense_top_k:    int  = 20,
        sparse_top_k:   int  = 20,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline:
        1. Dense retrieval  → top 20 semantic candidates
        2. Sparse retrieval → top 20 keyword candidates
        3. RRF fusion       → merged top 10

        We retrieve 20 from each so RRF has enough candidates
        to work with. Final output is top_k=10.
        """
        print(f"[HybridRetriever] Query: '{query[:70]}...'")

        # run both retrievers
        dense_results  = self.dense.retrieve(query,  top_k=dense_top_k)
        sparse_results = self.sparse.retrieve(query, top_k=sparse_top_k)

        print(f"[HybridRetriever] Dense:  {len(dense_results)} results")
        print(f"[HybridRetriever] Sparse: {len(sparse_results)} results")

        # fuse with RRF
        fused = reciprocal_rank_fusion(
            dense_results  = dense_results,
            sparse_results = sparse_results,
            top_n          = top_k
        )

        print(f"[HybridRetriever] Fused:  {len(fused)} results\n")
        return fused

    def retrieve_with_breakdown(self, query: str, top_k: int = 5) -> dict:
        """
        Same as retrieve() but also returns individual retriever
        results for comparison and evaluation purposes.

        Use this during eval to measure:
        - Dense only performance
        - Sparse only performance
        - Hybrid performance
        """
        dense_results  = self.dense.retrieve(query,  top_k=top_k)
        sparse_results = self.sparse.retrieve(query, top_k=top_k)
        hybrid_results = reciprocal_rank_fusion(
            dense_results, sparse_results, top_n=top_k
        )

        return {
            "query":   query,
            "dense":   dense_results,
            "sparse":  sparse_results,
            "hybrid":  hybrid_results
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    retriever = HybridRetriever()

    queries = [
        "What is the right to erasure under GDPR?",
        "What is the maximum fine for GDPR violation?",
        "What are the obligations of a data controller?"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        breakdown = retriever.retrieve_with_breakdown(query, top_k=3)

        # show top result from each system side by side
        print(f"\n--- DENSE TOP RESULT ---")
        if breakdown["dense"]:
            d = breakdown["dense"][0]
            print(f"  Score : {d['score']:.4f}")
            print(f"  Page  : {d['metadata'].get('page_number','?')}")
            print(f"  Text  : {d['text'][:150]}...")

        print(f"\n--- SPARSE TOP RESULT ---")
        if breakdown["sparse"]:
            s = breakdown["sparse"][0]
            print(f"  Score : {s['score']:.4f}")
            print(f"  Page  : {s['metadata'].get('page_number','?')}")
            print(f"  Text  : {s['text'][:150]}...")

        print(f"\n--- HYBRID (RRF) TOP RESULT ---")
        if breakdown["hybrid"]:
            h = breakdown["hybrid"][0]
            print(f"  RRF Score : {h['rrf_score']:.6f}")
            print(f"  Page      : {h['metadata'].get('page_number','?')}")
            print(f"  Text      : {h['text'][:150]}...")