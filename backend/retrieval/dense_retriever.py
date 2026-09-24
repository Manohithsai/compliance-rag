# backend/retrieval/dense_retriever.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.ingestion.embedder import embed_query
from backend.ingestion.indexer  import (
    get_chroma_client,
    get_or_create_collection,
    search_similar,
    get_parent_chunk,
    CHILD_COLLECTION,
    PARENT_COLLECTION
)


class DenseRetriever:
    """
    Dense retrieval using ChromaDB + nomic-embed-text embeddings.

    HOW IT WORKS:
    1. Embed the query into a 768-dim vector
    2. Find the most similar chunk vectors in ChromaDB
    3. Return the parent chunks for full context

    STRENGTHS:
    - Understands semantic meaning
    - Finds relevant chunks even when exact words differ
    - "delete my account" matches "right to erasure"

    WEAKNESSES:
    - Can miss exact legal terms like "Article 17(3)(b)"
    - Struggles with specific clause numbers and codes
    - This is why we pair it with BM25
    """

    def __init__(self):
        self.client            = get_chroma_client()
        self.child_collection  = get_or_create_collection(
            self.client, CHILD_COLLECTION
        )
        self.parent_collection = get_or_create_collection(
            self.client, PARENT_COLLECTION
        )
        print("[DenseRetriever] Ready.")

    def retrieve(
        self,
        query:         str,
        top_k:         int = 10,
        use_parents:   bool = True
    ) -> list[dict]:
        """
        Retrieve top_k most relevant chunks for a query.

        use_parents=True means:
        - Retrieve precise child chunks
        - But return their parent chunks to the LLM
        - Best of both worlds: precision + context
        """
        # Step 1 — embed the query
        query_vector = embed_query(query)

        # Step 2 — search child chunks (precise)
        child_hits = search_similar(
            query_embedding = query_vector,
            collection      = self.child_collection,
            top_k           = top_k
        )

        if not use_parents:
            return child_hits

        # Step 3 — fetch parent chunks (contextual)
        results = []
        seen_parents = set()  # avoid duplicate parents

        for hit in child_hits:
            parent = get_parent_chunk(
                hit["metadata"],
                self.parent_collection
            )

            if parent:
                parent_id = hit["metadata"].get("parent_id", "")

                # skip if we already have this parent
                if parent_id in seen_parents:
                    continue

                seen_parents.add(parent_id)

                results.append({
                    "text":        parent["text"],
                    "metadata":    parent["metadata"],
                    "score":       hit["score"],  # child's score
                    "child_text":  hit["text"],   # what actually matched
                    "source":      "dense"
                })
            else:
                # fallback to child if parent not found
                results.append({**hit, "source": "dense"})

        return results


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    retriever = DenseRetriever()

    queries = [
        "What is the right to erasure under GDPR?",
        "What are the obligations of a data controller?",
        "What is the maximum fine for GDPR violation?"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        results = retriever.retrieve(query, top_k=3)

        for i, r in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  Score  : {r['score']:.4f}")
            print(f"  Page   : {r['metadata'].get('page_number', '?')}")
            print(f"  Source : {r['source']}")
            print(f"  Text   : {r['text'][:200]}...")