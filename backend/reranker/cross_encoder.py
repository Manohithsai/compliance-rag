# backend/reranker/cross_encoder.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from flashrank import Ranker, RerankRequest


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using FlashRank cross-encoder.

    WHY RERANKING?

    The hybrid retriever uses bi-encoders — query and document
    are embedded SEPARATELY then compared with cosine similarity.

    Fast: O(1) at query time since doc vectors are pre-computed.
    But imprecise: the model never sees query + document together.

    A cross-encoder reads BOTH query and document simultaneously
    enabling full attention between them. Much more accurate.

    Too slow to run over thousands of documents — but perfect
    for reranking a shortlist of 10-20 candidates.

    THE PATTERN:
    Bi-encoder  → retrieves top 20 broadly  (fast, recall-focused)
    Cross-encoder → reranks to top 5 precisely (slow, precision-focused)

    WHY FLASHRANK?
    - Runs locally — no API key, no cost
    - Fast enough on CPU
    - No sensitive compliance data sent externally
    - Multiple model options (nano for speed, small for accuracy)
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        print(f"[Reranker] Loading cross-encoder: {model_name}")
        print(f"[Reranker] First load downloads the model (~50MB)...")
        self.ranker = Ranker(model_name=model_name)
        print(f"[Reranker] Ready.")

    def rerank(
        self,
        query:    str,
        chunks:   list[dict],
        top_k:    int = 5
    ) -> list[dict]:
        """
        Rerank chunks by relevance to query.

        Takes the hybrid retriever's output (10-20 chunks)
        and returns the most relevant top_k.

        Each chunk must have a 'text' key.
        All other metadata is preserved.
        """
        if not chunks:
            return []

        # FlashRank expects list of dicts with 'text' key
        passages = [
            {
                "id":   i,
                "text": chunk["text"]
            }
            for i, chunk in enumerate(chunks)
        ]

        # build rerank request
        request = RerankRequest(query=query, passages=passages)

        # run reranking
        results = self.ranker.rerank(request)

        # map scores back to original chunks
        reranked = []
        for result in results[:top_k]:
            original_idx   = result["id"]
            original_chunk = chunks[original_idx].copy()

            # add reranker score
            original_chunk["rerank_score"]    = result["score"]
            original_chunk["retrieval_score"] = original_chunk.get(
                "rrf_score",
                original_chunk.get("score", 0)
            )
            original_chunk["source"] = "reranked"

            reranked.append(original_chunk)

        return reranked

    def rerank_with_scores(
        self,
        query:  str,
        chunks: list[dict],
        top_k:  int = 5
    ) -> dict:
        """
        Same as rerank() but returns full comparison data.
        Used in evaluation to measure reranker improvement.
        """
        reranked = self.rerank(query, chunks, top_k=top_k)

        return {
            "query":           query,
            "before_rerank":   chunks[:top_k],
            "after_rerank":    reranked,
            "total_input":     len(chunks),
            "total_output":    len(reranked)
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from backend.retrieval.hybrid_fusion import HybridRetriever

    # get hybrid results first
    retriever = HybridRetriever()
    reranker  = CrossEncoderReranker()

    queries = [
        "What is the right to erasure under GDPR?",
        "What is the maximum fine for GDPR violation?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        # Step 1 — hybrid retrieval (top 10)
        hybrid_results = retriever.retrieve(query, top_k=10)

        # Step 2 — rerank to top 5
        comparison = reranker.rerank_with_scores(
            query  = query,
            chunks = hybrid_results,
            top_k  = 5
        )

        print(f"\n--- BEFORE RERANK (Hybrid top 3) ---")
        for i, r in enumerate(comparison["before_rerank"][:3]):
            print(f"\n  #{i+1} RRF Score: {r.get('rrf_score', 0):.6f}")
            print(f"       Page: {r['metadata'].get('page_number','?')}")
            print(f"       Text: {r['text'][:120]}...")

        print(f"\n--- AFTER RERANK (Cross-encoder top 3) ---")
        for i, r in enumerate(comparison["after_rerank"][:3]):
            print(f"\n  #{i+1} Rerank Score: {r['rerank_score']:.6f}")
            print(f"       Page: {r['metadata'].get('page_number','?')}")
            print(f"       Text: {r['text'][:120]}...")

        # key insight — did ranking change?
        before_pages = [
            r['metadata'].get('page_number','?')
            for r in comparison["before_rerank"][:3]
        ]
        after_pages = [
            r['metadata'].get('page_number','?')
            for r in comparison["after_rerank"][:3]
        ]

        print(f"\n  Pages before rerank : {before_pages}")
        print(f"  Pages after rerank  : {after_pages}")

        if before_pages != after_pages:
            print(f"  ✅ Reranker changed the order — adding value")
        else:
            print(f"  ➡️  Same order — reranker confirmed hybrid ranking")