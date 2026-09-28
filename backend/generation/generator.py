# backend/generation/generator.py

import os
import sys
import ollama
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the prompt that goes to Llama.
    Chunks are already deduplicated before this is called.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk["metadata"].get("page_number", "?")
        text = chunk["text"]
        context_parts.append(f"[Source {i} — Page {page}]\n{text}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a compliance assistant. 
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find 
this information in the provided documents."
Always mention which source you used.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    return prompt

def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate an answer using Llama + retrieved chunks.
    Returns answer text + sources used.
    """
    if not chunks:
        return {
            "answer":  "No relevant documents found for your query.",
            "sources": [],
            "query":   query
        }

    # deduplicate by page number FIRST
    seen_pages    = set()
    unique_chunks = []

    for chunk in chunks:
        page = chunk["metadata"].get("page_number", "?")
        if page not in seen_pages:
            seen_pages.add(page)
            unique_chunks.append(chunk)

    print(f"[Generator] Generating answer for: '{query[:60]}...'")
    print(f"[Generator] Using {len(unique_chunks)} unique chunks (from {len(chunks)} retrieved)...")

    prompt   = build_prompt(query, unique_chunks)

    response = ollama.chat(
        model    = OLLAMA_MODEL,
        messages = [{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    # build sources from unique chunks only
    sources = []
    for i, chunk in enumerate(unique_chunks, start=1):
        sources.append({
            "source_num": i,
            "page":       chunk["metadata"].get("page_number", "?"),
            "doc":        chunk["metadata"].get("doc_name", "unknown"),
            "preview":    chunk["text"][:100] + "..."
        })

    print(f"[Generator] Done.\n")

    return {
        "answer":  answer,
        "sources": sources,
        "query":   query
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from backend.retrieval.hybrid_fusion import HybridRetriever

    retriever = HybridRetriever()

    queries = [
        "What is the right to erasure under GDPR?",
        "What is the maximum fine for a GDPR violation?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        # retrieve relevant chunks
        chunks = retriever.retrieve(query, top_k=5)

        # generate answer
        result = generate_answer(query, chunks)

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES USED:")
        for s in result["sources"]:
            print(f"  [{s['source_num']}] Page {s['page']} — {s['preview']}")