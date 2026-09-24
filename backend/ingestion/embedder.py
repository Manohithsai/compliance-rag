# backend/ingestion/embedder.py

import ollama
from typing import List
from dotenv import load_dotenv
import os
import time

from backend.ingestion.chunker import Chunk

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def get_embedding(text: str) -> list[float]:
    """
    Convert a single piece of text into a vector (list of floats).

    WHY EMBEDDINGS?
    Computers can't compare meaning of text directly.
    Embeddings convert text into numbers where:
    - Similar meaning = vectors close together in space
    - Different meaning = vectors far apart

    Example:
    "right to erasure" and "delete my data" will have
    very similar vectors even though words are different.
    This is what makes semantic search work.

    We use nomic-embed-text which runs locally via Ollama.
    No API key. No cost. No data leaving your machine.
    Critical for compliance documents with sensitive content.
    """
    try:
        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text
        )
        return response["embedding"]

    except Exception as e:
        print(f"[Embedder] Error embedding text: {e}")
        raise


def embed_chunks(
    chunks: list[Chunk],
    batch_size: int = 10,
    chunk_type: str = "child"
) -> list[dict]:
    """
    Embed a list of chunks and return them ready for ChromaDB.

    We process in batches to avoid overwhelming Ollama.
    batch_size=10 is safe for most machines.

    Returns a list of dicts with:
    - id: unique chunk ID
    - embedding: the vector
    - text: original text
    - metadata: all the extra info
    """
    results   = []
    total     = len(chunks)
    failed    = 0

    print(f"[Embedder] Embedding {total} {chunk_type} chunks using {EMBEDDING_MODEL}...")
    print(f"[Embedder] This may take a few minutes on first run...")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]

        for chunk in batch:
            try:
                embedding = get_embedding(chunk.text)

                results.append({
                    "id":        chunk.chunk_id,
                    "embedding": embedding,
                    "text":      chunk.text,
                    "metadata":  {
                        **chunk.metadata,
                        "chunk_id":   chunk.chunk_id,
                        "parent_id":  chunk.parent_id or "",
                        "doc_name":   chunk.doc_name,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                    }
                })

            except Exception as e:
                print(f"[Embedder] Skipping chunk {chunk.chunk_id}: {e}")
                failed += 1

        # progress update every batch
        completed = min(i + batch_size, total)
        print(f"[Embedder] Progress: {completed}/{total} chunks embedded", end="\r")

    print()  # newline after progress
    print(f"[Embedder] Done. {len(results)} embedded, {failed} failed.")

    return results


def embed_query(query: str) -> list[float]:
    """
    Embed a user query for similarity search.

    Kept separate from embed_chunks because:
    1. Query embedding happens at runtime (every user question)
    2. Chunk embedding happens at indexing time (once per document)
    3. We may want different logic for each later
    """
    print(f"[Embedder] Embedding query: '{query[:60]}...'")
    return get_embedding(query)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    # add project root to path
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

    from backend.ingestion.chunker import process_document

    # find PDF
    raw_dir   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs in data/raw/")
        sys.exit(1)

    pdf_path = os.path.join(raw_dir, pdf_files[0])

    # get chunks
    parents, children = process_document(pdf_path)

    # only embed first 5 child chunks for testing
    print("\n[Test] Embedding first 5 child chunks only...")
    test_chunks = children[:5]

    embedded = embed_chunks(test_chunks, chunk_type="child")

    print(f"\n[Test] Results:")
    print(f"  Chunks embedded : {len(embedded)}")
    print(f"  Embedding dim   : {len(embedded[0]['embedding'])}")
    print(f"  Sample ID       : {embedded[0]['id']}")
    print(f"  Sample text     : {embedded[0]['text'][:100]}...")

    # test query embedding
    print("\n[Test] Embedding a sample query...")
    query_vec = embed_query("What is the right to erasure under GDPR?")
    print(f"  Query embedding dim: {len(query_vec)}")
    print(f"  First 5 values     : {query_vec[:5]}")

    print("\n[Embedder] All tests passed.")