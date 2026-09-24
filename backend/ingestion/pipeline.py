# backend/ingestion/pipeline.py
# This is the master ingestion script.
# Run this once per document to fully index it.

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.ingestion.chunker  import process_document
from backend.ingestion.embedder import embed_chunks
from backend.ingestion.indexer  import (
    get_chroma_client,
    get_or_create_collection,
    index_chunks,
    get_collection_stats,
    PARENT_COLLECTION,
    CHILD_COLLECTION
)


def ingest_document(pdf_path: str) -> dict:
    """
    Full ingestion pipeline for one PDF.
    Chunk → Embed → Index → Done.

    This is what gets called when a user uploads a document.
    Returns a summary of what was indexed.
    """
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"INGESTING: {os.path.basename(pdf_path)}")
    print(f"{'='*60}\n")

    # ── Step 1: Chunk ──────────────────────────────────────────
    parents, children = process_document(pdf_path)

    # ── Step 2: Embed ──────────────────────────────────────────
    print("[Pipeline] Embedding child chunks (used for retrieval)...")
    embedded_children = embed_chunks(children, chunk_type="child")

    print("[Pipeline] Embedding parent chunks (used for context)...")
    embedded_parents = embed_chunks(parents, chunk_type="parent")

    # ── Step 3: Index ──────────────────────────────────────────
    client            = get_chroma_client()
    child_collection  = get_or_create_collection(client, CHILD_COLLECTION)
    parent_collection = get_or_create_collection(client, PARENT_COLLECTION)

    print("\n[Pipeline] Indexing into ChromaDB...")
    indexed_children = index_chunks(embedded_children, child_collection)
    indexed_parents  = index_chunks(embedded_parents,  parent_collection)

    elapsed = round(time.time() - start_time, 2)

    summary = {
        "document":        os.path.basename(pdf_path),
        "pages":           len(set(c.page_number for c in children)),
        "parent_chunks":   indexed_parents,
        "child_chunks":    indexed_children,
        "time_seconds":    elapsed,
        "child_collection": get_collection_stats(child_collection),
        "parent_collection": get_collection_stats(parent_collection)
    }

    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE")
    print(f"  Document      : {summary['document']}")
    print(f"  Parent chunks : {summary['parent_chunks']}")
    print(f"  Child chunks  : {summary['child_chunks']}")
    print(f"  Time taken    : {summary['time_seconds']}s")
    print(f"{'='*60}\n")

    return summary


def ingest_all_documents(raw_dir: str) -> list[dict]:
    """
    Ingest every PDF in the raw directory.
    This is what you run once to build the full index.
    """
    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"[Pipeline] No PDFs found in {raw_dir}")
        return []

    print(f"[Pipeline] Found {len(pdf_files)} PDF(s) to ingest:")
    for f in pdf_files:
        print(f"  - {f}")

    summaries = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(raw_dir, pdf_file)
        summary  = ingest_document(pdf_path)
        summaries.append(summary)

    print(f"\n[Pipeline] All documents ingested successfully.")
    return summaries


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw"
    )
    ingest_all_documents(raw_dir)