# backend/ingestion/indexer.py

import os
import sys
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "./data/processed/chroma")

# Collection names — we keep parents and children separate
PARENT_COLLECTION = "parent_chunks"
CHILD_COLLECTION  = "child_chunks"


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Create and return a persistent ChromaDB client.

    WHY PERSISTENT?
    Without persistence, ChromaDB stores everything in memory.
    Every time you restart, you'd have to re-embed all documents.
    With persistence, embeddings are saved to disk and reloaded instantly.
    For 594 chunks at ~2 mins embedding time — this matters a lot.
    """
    os.makedirs(CHROMA_PERSIST_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    print(f"[Indexer] ChromaDB connected at: {CHROMA_PERSIST_PATH}")
    return client


def get_or_create_collection(
    client:          chromadb.PersistentClient,
    collection_name: str
) -> chromadb.Collection:
    """
    Get existing collection or create a new one.
    Safe to call multiple times — won't duplicate.
    """
    collection = client.get_or_create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "cosine"}
        # WHY COSINE?
        # Cosine similarity measures the ANGLE between vectors.
        # This is better than Euclidean distance for text because
        # it ignores magnitude and focuses purely on direction (meaning).
        # Two texts about "data deletion" will point in the same
        # direction even if one is longer than the other.
    )
    return collection


def index_chunks(
    embedded_chunks: list[dict],
    collection:      chromadb.Collection,
    batch_size:      int = 50
) -> int:
    """
    Store embedded chunks into ChromaDB.

    ChromaDB expects three parallel lists:
    - ids:        unique string IDs
    - embeddings: list of vectors
    - documents:  the actual text
    - metadatas:  dict of metadata per chunk

    We batch to avoid memory issues with large documents.
    Returns count of chunks successfully indexed.
    """
    total   = len(embedded_chunks)
    indexed = 0

    for i in range(0, total, batch_size):
        batch = embedded_chunks[i : i + batch_size]

        ids        = [c["id"]        for c in batch]
        embeddings = [c["embedding"] for c in batch]
        documents  = [c["text"]      for c in batch]
        metadatas  = [c["metadata"]  for c in batch]

        try:
            collection.upsert(
                ids        = ids,
                embeddings = embeddings,
                documents  = documents,
                metadatas  = metadatas
            )
            # WHY UPSERT NOT ADD?
            # If you index the same document twice,
            # upsert updates existing records instead of
            # throwing a duplicate ID error.
            # Critical for when documents are updated.

            indexed += len(batch)
            print(f"[Indexer] Indexed {indexed}/{total} chunks", end="\r")

        except Exception as e:
            print(f"[Indexer] Error indexing batch {i}: {e}")

    print()
    return indexed


def search_similar(
    query_embedding: list[float],
    collection:      chromadb.Collection,
    top_k:           int = 10
) -> list[dict]:
    """
    Search ChromaDB for chunks most similar to the query embedding.

    Returns top_k most relevant chunks with their text and metadata.
    This is the DENSE retrieval part of our hybrid search.

    top_k=10 because:
    - We retrieve broadly here (10 candidates)
    - Then reranker narrows to top 5
    - Better to retrieve more and rerank than miss relevant chunks
    """
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"]
    )

    # format into clean list of dicts
    hits = []
    for i, doc in enumerate(results["documents"][0]):
        hits.append({
            "text":     doc,
            "metadata": results["metadatas"][0][i],
            "score":    1 - results["distances"][0][i],
            # WHY 1 - distance?
            # ChromaDB returns cosine DISTANCE (0=identical, 2=opposite)
            # We convert to SIMILARITY (1=identical, -1=opposite)
            # Higher score = more relevant
            "source":   "dense"
        })

    return hits


def get_parent_chunk(
    child_metadata:   dict,
    parent_collection: chromadb.Collection
) -> dict | None:
    """
    Given a child chunk's metadata, fetch its parent chunk.

    This is the PARENT DOCUMENT RETRIEVAL pattern:
    1. We retrieve precise child chunks
    2. We look up their parents for full context
    3. We send parents to the LLM

    Without this, the LLM sees only 128-word fragments.
    With this, it sees the full 512-word context window.
    """
    parent_id = child_metadata.get("parent_id", "")

    if not parent_id:
        return None

    try:
        result = parent_collection.get(
            ids     = [parent_id],
            include = ["documents", "metadatas"]
        )

        if result["documents"]:
            return {
                "text":     result["documents"][0],
                "metadata": result["metadatas"][0],
                "source":   "parent_lookup"
            }

    except Exception as e:
        print(f"[Indexer] Parent lookup failed for {parent_id}: {e}")

    return None


def get_collection_stats(collection: chromadb.Collection) -> dict:
    """
    Get basic stats about a collection.
    Useful for debugging and the eval dashboard.
    """
    count = collection.count()
    return {
        "name":        collection.name,
        "total_chunks": count,
    }


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

    from backend.ingestion.chunker  import process_document
    from backend.ingestion.embedder import embed_chunks, embed_query

    # find PDF
    raw_dir   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    pdf_path  = os.path.join(raw_dir, pdf_files[0])

    # Step 1 — chunk
    parents, children = process_document(pdf_path)

    # Step 2 — embed (first 20 only for speed during test)
    print("[Test] Embedding first 20 child chunks...")
    embedded_children = embed_chunks(children[:20], chunk_type="child")

    print("[Test] Embedding first 20 parent chunks...")
    embedded_parents  = embed_chunks(parents[:20],  chunk_type="parent")

    # Step 3 — index
    client           = get_chroma_client()
    child_collection = get_or_create_collection(client, CHILD_COLLECTION)
    parent_collection= get_or_create_collection(client, PARENT_COLLECTION)

    print("\n[Test] Indexing child chunks...")
    indexed_children = index_chunks(embedded_children, child_collection)

    print("[Test] Indexing parent chunks...")
    indexed_parents  = index_chunks(embedded_parents,  parent_collection)

    # Step 4 — search
    print("\n[Test] Running a test search...")
    query     = "What is the right to erasure under GDPR?"
    query_vec = embed_query(query)

    hits = search_similar(query_vec, child_collection, top_k=3)

    print(f"\n[Test] Top 3 results for: '{query}'")
    print("=" * 60)
    for i, hit in enumerate(hits):
        print(f"\nResult {i+1}:")
        print(f"  Score : {hit['score']:.4f}")
        print(f"  Page  : {hit['metadata']['page_number']}")
        print(f"  Text  : {hit['text'][:150]}...")

    # Step 5 — parent lookup
    print("\n[Test] Fetching parent of top result...")
    parent = get_parent_chunk(hits[0]["metadata"], parent_collection)
    if parent:
        print(f"  Parent text: {parent['text'][:200]}...")
    else:
        print("  No parent found (expected if parent wasn't in test batch)")

    # Step 6 — stats
    print("\n[Test] Collection stats:")
    print(f"  Children: {get_collection_stats(child_collection)}")
    print(f"  Parents : {get_collection_stats(parent_collection)}")

    print("\n[Indexer] All tests passed.")