# backend/retrieval/sparse_retriever.py

import os
import sys
import pickle
from rank_bm25 import BM25Okapi

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.ingestion.chunker import process_document

# Where to save the BM25 index so we don't rebuild every time
BM25_INDEX_PATH = "./data/processed/bm25_index.pkl"


class SparseRetriever:
    """
    Sparse retrieval using BM25 keyword matching.

    HOW IT WORKS:
    BM25 scores each document based on:
    - Term frequency (TF) — how often the query word appears
    - Inverse document frequency (IDF) — how rare the word is
    - Document length normalization — penalizes very long chunks

    WHY BM25 FOR COMPLIANCE?
    Legal text is full of exact terms that MUST match precisely:
    - "Article 17(3)(b)" — a vector model might miss this
    - "data processor" vs "data controller" — legally distinct
    - "GDPR" — an acronym that embeddings handle poorly
    - Clause numbers, section references, defined terms

    BM25 catches ALL of these perfectly because it's
    exact token matching, not semantic similarity.
    """

    def __init__(self):
        self.bm25       = None
        self.chunks     = []   # stores all chunk dicts
        self.corpus     = []   # tokenized corpus for BM25

        # load existing index if available
        if os.path.exists(BM25_INDEX_PATH):
            self._load_index()
        else:
            print("[SparseRetriever] No index found. Call build_index() first.")

    def build_index(self, pdf_paths: list[str]) -> None:
        """
        Build BM25 index from a list of PDF paths.
        Run this once after ingesting documents.
        """
        print("[SparseRetriever] Building BM25 index...")

        self.chunks = []
        self.corpus = []

        for pdf_path in pdf_paths:
            _, children = process_document(pdf_path)

            for chunk in children:
                # tokenize by splitting on whitespace + lowercase
                # simple but effective for BM25
                tokens = chunk.text.lower().split()

                self.chunks.append({
                    "text":     chunk.text,
                    "metadata": {
                        "doc_name":    chunk.doc_name,
                        "page_number": chunk.page_number,
                        "chunk_id":    chunk.chunk_id,
                        "parent_id":   chunk.parent_id or "",
                        "type":        "child"
                    }
                })
                self.corpus.append(tokens)

        # build BM25 model
        self.bm25 = BM25Okapi(self.corpus)

        # save to disk
        self._save_index()

        print(f"[SparseRetriever] Index built with {len(self.chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve top_k chunks using BM25 keyword matching.

        Returns results in same format as DenseRetriever
        so they can be merged easily by the hybrid fusion layer.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        # tokenize query same way as corpus
        query_tokens = query.lower().split()

        # get BM25 scores for all chunks
        scores = self.bm25.get_scores(query_tokens)

        # get top_k indices sorted by score
        top_indices = sorted(
            range(len(scores)),
            key    = lambda i: scores[i],
            reverse= True
        )[:top_k]

        results = []
        for idx in top_indices:
            score = scores[idx]

            # skip zero-score results
            if score == 0:
                continue

            results.append({
                "text":     self.chunks[idx]["text"],
                "metadata": self.chunks[idx]["metadata"],
                "score":    float(score),
                "source":   "sparse"
            })

        return results

    def _save_index(self) -> None:
        """Save BM25 index to disk."""
        os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({
                "bm25":   self.bm25,
                "chunks": self.chunks,
                "corpus": self.corpus
            }, f)
        print(f"[SparseRetriever] Index saved to {BM25_INDEX_PATH}")

    def _load_index(self) -> None:
        """Load BM25 index from disk."""
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25   = data["bm25"]
        self.chunks = data["chunks"]
        self.corpus = data["corpus"]
        print(f"[SparseRetriever] Loaded BM25 index ({len(self.chunks)} chunks)")


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    raw_dir   = "./data/raw"
    pdf_files = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.endswith(".pdf")
    ]

    retriever = SparseRetriever()

    # build index if not exists
    if retriever.bm25 is None:
        retriever.build_index(pdf_files)

    queries = [
        "What is the right to erasure under GDPR?",
        "Article 17 deletion of personal data",
        "maximum fine data breach penalty"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        results = retriever.retrieve(query, top_k=3)

        for i, r in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  BM25 Score : {r['score']:.4f}")
            print(f"  Page       : {r['metadata'].get('page_number', '?')}")
            print(f"  Text       : {r['text'][:200]}...")