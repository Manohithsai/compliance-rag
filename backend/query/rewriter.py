# backend/query/rewriter.py

import os
import sys
import ollama
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def rewrite_with_stepback(query: str) -> str:
    """
    Step-back prompting.

    Takes a specific question and generates a more
    abstract version that retrieves broader context.

    Example:
    Specific: "Can we store EU customer emails on US servers?"
    Abstract: "What are the GDPR requirements for 
               cross-border data transfers?"

    WHY THIS HELPS:
    The abstract question retrieves the general regulation
    context. The specific question retrieves the exact clause.
    Together they give the LLM full context to answer well.
    """
    prompt = f"""You are a legal research assistant.
Your job is to take a specific compliance question and 
rewrite it as a more general, abstract question that 
would help retrieve broader regulatory context.

Original question: {query}

Write ONLY the rewritten abstract question. 
Nothing else. No explanation.

Rewritten question:"""

    response = ollama.chat(
        model    = OLLAMA_MODEL,
        messages = [{"role": "user", "content": prompt}]
    )

    rewritten = response["message"]["content"].strip()
    print(f"[Rewriter] Step-back: {rewritten}")
    return rewritten


def rewrite_with_hyde(query: str) -> str:
    """
    HyDE — Hypothetical Document Embeddings.

    Instead of embedding the query directly,
    we ask Llama to generate what a perfect
    answer would look like, then embed THAT.

    WHY THIS WORKS:
    A hypothetical answer is in the same semantic
    space as actual document text.
    A question is in a different semantic space.

    Example:
    Query: "What is the right to erasure?"

    HyDE generates:
    "Under GDPR Article 17, data subjects have the right
    to obtain erasure of personal data without undue delay
    when the data is no longer necessary..."

    This hypothetical answer is much closer to the actual
    Article 17 text than the original question was.
    """
    prompt = f"""You are a GDPR compliance expert.
Write a short, factual paragraph (3-4 sentences) that 
would be the ideal answer to this compliance question.
Write it as if it came from an official regulatory document.

Question: {query}

Write ONLY the answer paragraph. No preamble.

Answer:"""

    response = ollama.chat(
        model    = OLLAMA_MODEL,
        messages = [{"role": "user", "content": prompt}]
    )

    hyde_doc = response["message"]["content"].strip()
    print(f"[Rewriter] HyDE doc: {hyde_doc[:100]}...")
    return hyde_doc


def generate_multi_queries(query: str, n: int = 3) -> list[str]:
    """
    Multi-query expansion.

    Generates n different versions of the same question.
    Each version is phrased differently to catch
    chunks that any single phrasing might miss.

    WHY THIS HELPS:
    If Article 17 uses the phrase "erasure of personal data"
    but the user asked "delete my account data" —
    a single query might miss it.
    Multiple phrasings increase the chance of a match.
    """
    prompt = f"""You are a compliance research assistant.
Generate {n} different ways to ask the following question.
Each version should use different words but mean the same thing.
Focus on legal and regulatory terminology.

Original question: {query}

Write ONLY the {n} questions, one per line.
No numbering. No explanation.

Questions:"""

    response = ollama.chat(
        model    = OLLAMA_MODEL,
        messages = [{"role": "user", "content": prompt}]
    )

    raw       = response["message"]["content"].strip()
    questions = [
        q.strip()
        for q in raw.split("\n")
        if q.strip() and len(q.strip()) > 10
    ][:n]

    print(f"[Rewriter] Generated {len(questions)} query variants")
    return questions


def rewrite_query(query: str) -> dict:
    """
    Main function — runs all three rewriting strategies.

    Returns a dict with:
    - original:     the user's original question
    - stepback:     abstract version for broader context
    - hyde:         hypothetical answer for better embedding
    - multi_queries: multiple phrasings for better recall
    - all_queries:  everything combined for retrieval
    """
    print(f"\n[Rewriter] Processing: '{query}'")
    print(f"[Rewriter] Running step-back prompting...")
    stepback = rewrite_with_stepback(query)

    print(f"[Rewriter] Running HyDE...")
    hyde = rewrite_with_hyde(query)

    print(f"[Rewriter] Running multi-query expansion...")
    multi = generate_multi_queries(query, n=3)

    # combine everything for retrieval
    all_queries = [query, stepback, hyde] + multi

    # deduplicate
    seen       = set()
    unique     = []
    for q in all_queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    print(f"[Rewriter] Total unique queries: {len(unique)}")

    return {
        "original":     query,
        "stepback":     stepback,
        "hyde":         hyde,
        "multi_queries": multi,
        "all_queries":  unique
    }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":

    queries = [
        "What is the right to erasure under GDPR?",
        "Can we share customer data with third parties?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"ORIGINAL: {query}")
        print(f"{'='*60}")

        result = rewrite_query(query)

        print(f"\nSTEP-BACK:")
        print(f"  {result['stepback']}")

        print(f"\nHYDE DOCUMENT:")
        print(f"  {result['hyde'][:200]}...")

        print(f"\nMULTI-QUERY VARIANTS:")
        for i, q in enumerate(result["multi_queries"], 1):
            print(f"  {i}. {q}")

        print(f"\nALL QUERIES FOR RETRIEVAL ({len(result['all_queries'])}):")
        for i, q in enumerate(result["all_queries"], 1):
            print(f"  {i}. {q}")