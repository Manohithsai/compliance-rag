# backend/ingestion/chunker.py

import os
import hashlib
from dataclasses import dataclass
from typing import Optional
from pypdf import PdfReader


@dataclass
class Chunk:
    """
    A single chunk of text extracted from a document.
    This is the fundamental unit that flows through the entire pipeline.
    """
    chunk_id:      str        # unique ID for this chunk
    text:          str        # the actual text content
    doc_name:      str        # which document this came from
    page_number:   int        # which page
    chunk_index:   int        # position of chunk within the document
    parent_id:     Optional[str]  # ID of parent chunk (for hierarchical chunking)
    metadata:      dict       # any extra info (section heading, doc type etc.)


def generate_chunk_id(doc_name: str, chunk_index: int, text: str) -> str:
    """
    Generate a unique, reproducible ID for each chunk.
    Uses MD5 hash of content so the same chunk always gets the same ID.
    This matters for deduplication and integrity checking later.
    """
    raw = f"{doc_name}_{chunk_index}_{text[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract raw text from each page of a PDF.
    Returns a list of dicts with page number and text.
    """
    pages = []
    reader = PdfReader(pdf_path)

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        # skip empty pages
        if not text or len(text.strip()) < 20:
            continue

        pages.append({
            "page_number": page_num + 1,   # human readable (starts at 1)
            "text":        text.strip()
        })

    print(f"[Chunker] Extracted {len(pages)} pages from {os.path.basename(pdf_path)}")
    return pages


def chunk_text_fixed(
    text:       str,
    chunk_size: int = 500,
    overlap:    int = 50
) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.

    Why overlap? So that a sentence split across two chunks
    still appears fully in at least one of them.
    This prevents losing context at chunk boundaries.

    chunk_size = 500 words is a good balance:
    - too small = loses context
    - too large = retrieval becomes imprecise
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end        = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)
        start = end - overlap   # move back by overlap to maintain continuity

    return chunks


def create_parent_child_chunks(
    pages:          list[dict],
    doc_name:       str,
    parent_size:    int = 512,
    child_size:     int = 128,
    overlap:        int = 20
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Hierarchical chunking — the most important function in this file.

    WHY HIERARCHICAL?
    - Child chunks (128 words) = small, precise, great for retrieval
    - Parent chunks (512 words) = large, contextual, great for answering

    At query time:
    1. We retrieve small child chunks (precise match)
    2. We return their parent chunks to the LLM (full context)

    This gives us precision of retrieval + quality of answer.
    Without this, you either get precise but context-poor chunks
    or context-rich but imprecise retrieval. Hierarchical solves both.
    """
    parent_chunks = []
    child_chunks  = []
    chunk_counter = 0

    for page in pages:
        page_text   = page["text"]
        page_number = page["page_number"]

        # --- Create PARENT chunks from this page ---
        parent_texts = chunk_text_fixed(page_text, chunk_size=parent_size, overlap=overlap)

        for parent_text in parent_texts:
            if len(parent_text.strip()) < 30:
                continue  # skip tiny fragments

            parent_id = generate_chunk_id(doc_name, chunk_counter, parent_text)

            parent_chunk = Chunk(
                chunk_id    = parent_id,
                text        = parent_text,
                doc_name    = doc_name,
                page_number = page_number,
                chunk_index = chunk_counter,
                parent_id   = None,   # parents have no parent
                metadata    = {
                    "type":      "parent",
                    "doc_name":  doc_name,
                    "page":      page_number,
                    "char_count": len(parent_text)
                }
            )
            parent_chunks.append(parent_chunk)

            # --- Create CHILD chunks from this parent ---
            child_texts = chunk_text_fixed(parent_text, chunk_size=child_size, overlap=overlap)

            for child_text in child_texts:
                if len(child_text.strip()) < 20:
                    continue

                child_id = generate_chunk_id(doc_name, chunk_counter + hash(child_text), child_text)

                child_chunk = Chunk(
                    chunk_id    = child_id,
                    text        = child_text,
                    doc_name    = doc_name,
                    page_number = page_number,
                    chunk_index = chunk_counter,
                    parent_id   = parent_id,  # linked to parent
                    metadata    = {
                        "type":      "child",
                        "doc_name":  doc_name,
                        "page":      page_number,
                        "parent_id": parent_id,
                        "char_count": len(child_text)
                    }
                )
                child_chunks.append(child_chunk)
                chunk_counter += 1

    return parent_chunks, child_chunks


def process_document(pdf_path: str) -> tuple[list[Chunk], list[Chunk]]:
    """
    Main entry point.
    Give it a PDF path, get back parent and child chunks.
    This is what the rest of the pipeline calls.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_name = os.path.basename(pdf_path).replace(".pdf", "")

    print(f"[Chunker] Processing: {doc_name}")

    # Step 1 — extract text page by page
    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        raise ValueError(f"No text extracted from {pdf_path}. Is it a scanned PDF?")

    # Step 2 — create hierarchical chunks
    parent_chunks, child_chunks = create_parent_child_chunks(pages, doc_name)

    print(f"[Chunker] Created {len(parent_chunks)} parent chunks")
    print(f"[Chunker] Created {len(child_chunks)} child chunks")
    print(f"[Chunker] Done.\n")

    return parent_chunks, child_chunks


# ── Quick test ──────────────────────────────────────────────────────────────
# Run this file directly to test: python backend/ingestion/chunker.py

if __name__ == "__main__":
    import sys

    # find first PDF in data/raw/
    raw_dir  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found in data/raw/ — add one first")
        sys.exit(1)

    pdf_path = os.path.join(raw_dir, pdf_files[0])

    parents, children = process_document(pdf_path)

    # print sample output
    print("=" * 60)
    print("SAMPLE PARENT CHUNK:")
    print(f"  ID:   {parents[0].chunk_id}")
    print(f"  Page: {parents[0].page_number}")
    print(f"  Text: {parents[0].text[:200]}...")
    print()
    print("SAMPLE CHILD CHUNK:")
    print(f"  ID:        {children[0].chunk_id}")
    print(f"  Page:      {children[0].page_number}")
    print(f"  Parent ID: {children[0].parent_id}")
    print(f"  Text:      {children[0].text[:150]}...")
    print("=" * 60)