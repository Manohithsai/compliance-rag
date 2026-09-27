# backend/reranker/cross_encoder.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from flashrank import Ranker, RerankRequest


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using FlashRank cross-encoder.

    POSITION IN PIPELINE:
    Multi-query retrieval → [RERANKER] → Generator

    Takes 18 unique chunks from multi-query retrieval.
    Returns precise top 5 for the LLM.

    WHY HERE:
    Multi-query gives us high RECALL (finds everything relevant)
    Cross-encoder gives us high PRECISION (ranks best ones first)
    Together = best of both worlds
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        print(f"[Reranker] Loading model: {model_name}")
        print(f"[Reranker] First run downloads ~50MB model...")
        self.ranker = Ranker(model_name=model_name)
        print(f"[Reranker] Ready.")

    def rerank(
        self,
        query:  str,
        chunks: list[dict],
        top_k:  int = 5
    ) -> list[dict]:
        """
        Rerank chunks by relevance to query.

        Input:  18 chunks from multi-query retrieval
        Output: top 5 most relevant chunks

        Each chunk is scored by reading query + chunk
        together — full attention, much more accurate
        than cosine similarity alone.
        """
        if not chunks:
            return []

        # FlashRank expects list of dicts with id + text
        passages = [
            {"id": i, "text": chunk["text"]}
            for i, chunk in enumerate(chunks)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        reranked = []
        for result in results[:top_k]:
            original_chunk = chunks[result["id"]].copy()
            original_chunk["rerank_score"]    = result["score"]
            original_chunk["retrieval_score"] = original_chunk.get(
                "rrf_score",
                original_chunk.get("score", 0)
            )
            original_chunk["source"] = "reranked"
            reranked.append(original_chunk)

        return reranked


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    from backend.retrieval.hybrid_fusion        import HybridRetriever
    from backend.query.multi_query_retriever    import multi_query_retrieve

    retriever = HybridRetriever()
    reranker  = CrossEncoderReranker()

    query = "What is the right to erasure under GDPR?"

    print(f"\n{'='*60}")
    print(f"FULL PIPELINE TEST")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # Step 1 — multi query retrieval
    retrieval_result = multi_query_retrieve(query, retriever, top_k=10)
    chunks_before    = retrieval_result["final_chunks"]

    print(f"\nBEFORE RERANK — Top 3:")
    for i, c in enumerate(chunks_before[:3], 1):
        score = c.get("rrf_score", c.get("score", 0))
        print(f"\n  #{i} Score: {score:.4f}")
        print(f"     Page : {c['metadata'].get('page_number','?')}")
        print(f"     Text : {c['text'][:120]}...")

    # Step 2 — rerank
    chunks_after = reranker.rerank(query, chunks_before, top_k=5)

    print(f"\nAFTER RERANK — Top 3:")
    for i, c in enumerate(chunks_after[:3], 1):
        print(f"\n  #{i} Rerank Score: {c['rerank_score']:.4f}")
        print(f"     Page        : {c['metadata'].get('page_number','?')}")
        print(f"     Text        : {c['text'][:120]}...")

    # show if order changed
    before_pages = [c['metadata'].get('page_number','?') for c in chunks_before[:3]]
    after_pages  = [c['metadata'].get('page_number','?') for c in chunks_after[:3]]

    print(f"\nPages before rerank : {before_pages}")
    print(f"Pages after rerank  : {after_pages}")

    if before_pages != after_pages:
        print(f"✅ Reranker changed the order — adding value")
    else:
        print(f"➡️  Same order — reranker confirmed retrieval ranking")