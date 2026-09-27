# backend/query/multi_query_retriever.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.query.rewriter          import rewrite_query
from backend.retrieval.hybrid_fusion import HybridRetriever


def deduplicate_results(all_results: list[dict]) -> list[dict]:
    """
    When we run multiple queries, same chunk may appear
    multiple times across different query results.
    We keep only the highest scoring occurrence.
    """
    seen  = {}

    for result in all_results:
        text  = result["text"]
        score = result.get("rrf_score", result.get("score", 0))

        if text not in seen:
            seen[text] = result
        else:
            existing_score = seen[text].get(
                "rrf_score",
                seen[text].get("score", 0)
            )
            if score > existing_score:
                seen[text] = result

    return list(seen.values())


def multi_query_retrieve(
    query:      str,
    retriever:  HybridRetriever,
    top_k:      int = 10
) -> dict:
    """
    Full query rewriting + multi-query retrieval pipeline.

    Steps:
    1. Rewrite original query into 6 variants
    2. Run each variant through hybrid retrieval
    3. Merge all results
    4. Deduplicate
    5. Return top_k unique chunks

    WHY THIS IS POWERFUL:
    Original query retrieves some chunks.
    Step-back query retrieves broader context chunks.
    HyDE retrieves chunks closest to ideal answer.
    Multi-queries catch chunks any single phrasing missed.

    Union of all these = much higher recall than
    single query retrieval.
    """
    print(f"\n[MultiQuery] Starting retrieval for: '{query[:60]}'")

    # Step 1 — rewrite
    rewritten = rewrite_query(query)
    all_queries = rewritten["all_queries"]

    print(f"\n[MultiQuery] Running {len(all_queries)} queries...")

    # Step 2 — retrieve for each query
    all_results = []

    for i, q in enumerate(all_queries, 1):
        print(f"\n[MultiQuery] Query {i}/{len(all_queries)}: '{q[:60]}...'")

        results = retriever.retrieve(q, top_k=5)
        all_results.extend(results)

        print(f"[MultiQuery] Got {len(results)} results")

    print(f"\n[MultiQuery] Total raw results: {len(all_results)}")

    # Step 3 — deduplicate
    unique_results = deduplicate_results(all_results)
    print(f"[MultiQuery] After dedup: {len(unique_results)} unique chunks")

    # Step 4 — sort by score and take top_k
    unique_results.sort(
        key     = lambda x: x.get("rrf_score", x.get("score", 0)),
        reverse = True
    )
    final_results = unique_results[:top_k]

    print(f"[MultiQuery] Returning top {len(final_results)} chunks\n")

    return {
        "original_query": query,
        "queries_used":   all_queries,
        "total_retrieved": len(all_results),
        "unique_chunks":  len(unique_results),
        "final_chunks":   final_results
    }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":

    retriever = HybridRetriever()

    query = "What is the right to erasure under GDPR?"

    print(f"\n{'='*60}")
    print(f"MULTI-QUERY RETRIEVAL TEST")
    print(f"Query: {query}")
    print(f"{'='*60}")

    result = multi_query_retrieve(query, retriever, top_k=5)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Queries used     : {len(result['queries_used'])}")
    print(f"Total retrieved  : {result['total_retrieved']}")
    print(f"Unique chunks    : {result['unique_chunks']}")
    print(f"Returned to LLM  : {len(result['final_chunks'])}")

    print(f"\nTOP 3 CHUNKS:")
    for i, chunk in enumerate(result["final_chunks"][:3], 1):
        print(f"\n  #{i}")
        print(f"  Page  : {chunk['metadata'].get('page_number','?')}")
        print(f"  Score : {chunk.get('rrf_score', chunk.get('score',0)):.4f}")
        print(f"  Text  : {chunk['text'][:150]}...")