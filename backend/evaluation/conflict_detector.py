# backend/evaluation/conflict_detector.py

import os
import sys
import ollama
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


@dataclass
class ConflictResult:
    """
    Result of comparing two policy chunks.
    """
    conflict_type:  str    # DIRECT / CONDITIONAL / SCOPE / JURISDICTION / NONE
    severity:       str    # HIGH / MEDIUM / LOW / NONE
    explanation:    str    # human readable explanation
    chunk_a_text:   str    # first chunk
    chunk_b_text:   str    # second chunk
    chunk_a_page:   int    # page of first chunk
    chunk_b_page:   int    # page of second chunk
    chunk_a_doc:    str    # document of first chunk
    chunk_b_doc:    str    # document of second chunk
    recommendation: str    # what to do about it


def detect_conflict_between_chunks(
    chunk_a: dict,
    chunk_b: dict
) -> ConflictResult:
    """
    Compare two chunks and detect if they conflict.

    CONFLICT TYPES (as the senior reviewer suggested):

    1. DIRECT — clear contradiction
       "Delete immediately" vs "Never delete"

    2. CONDITIONAL — compatible with conditions
       "Delete within 30 days unless legally required"
       vs "Retain financial records 7 years"
       These may actually be compatible.

    3. SCOPE — different domains, not a conflict
       Policy A covers marketing data.
       Policy B covers financial records.

    4. JURISDICTION — different legal frameworks
       EU GDPR vs US CCPA
       Different requirements, not contradiction.

    5. NONE — no conflict detected
    """
    text_a = chunk_a["text"]
    text_b = chunk_b["text"]
    page_a = chunk_a["metadata"].get("page_number", "?")
    page_b = chunk_b["metadata"].get("page_number", "?")
    doc_a  = chunk_a["metadata"].get("doc_name", "Document A")
    doc_b  = chunk_b["metadata"].get("doc_name", "Document B")

    # skip if same document and same page
    if doc_a == doc_b and page_a == page_b:
        return ConflictResult(
            conflict_type  = "NONE",
            severity       = "NONE",
            explanation    = "Same source — no conflict possible",
            chunk_a_text   = text_a,
            chunk_b_text   = text_b,
            chunk_a_page   = page_a,
            chunk_b_page   = page_b,
            chunk_a_doc    = doc_a,
            chunk_b_doc    = doc_b,
            recommendation = "N/A"
        )

    prompt = f"""You are a legal compliance expert analyzing 
two policy clauses for conflicts.

Analyze these two clauses carefully:

CLAUSE A (from {doc_a}, Page {page_a}):
{text_a[:500]}

CLAUSE B (from {doc_b}, Page {page_b}):
{text_b[:500]}

Determine the relationship between these clauses.
Respond in EXACTLY this format:

CONFLICT_TYPE: [DIRECT or CONDITIONAL or SCOPE or JURISDICTION or NONE]
SEVERITY: [HIGH or MEDIUM or LOW or NONE]
EXPLANATION: [One sentence explaining the relationship]
RECOMMENDATION: [One sentence on what action to take]

Definitions:
- DIRECT: One clause clearly contradicts the other
- CONDITIONAL: Clauses differ but may be compatible depending on context
- SCOPE: Clauses cover different domains so no real conflict
- JURISDICTION: Clauses apply to different legal jurisdictions
- NONE: No conflict, clauses are compatible or unrelated

Respond with ONLY the four lines above. Nothing else."""

    response = ollama.chat(
        model    = OLLAMA_MODEL,
        messages = [{"role": "user", "content": prompt}]
    )

    raw = response["message"]["content"].strip()

    # parse the response
    conflict_type  = "NONE"
    severity       = "NONE"
    explanation    = "Could not parse response"
    recommendation = "Manual review recommended"

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("CONFLICT_TYPE:"):
            conflict_type = line.replace("CONFLICT_TYPE:", "").strip()
        elif line.startswith("SEVERITY:"):
            severity = line.replace("SEVERITY:", "").strip()
        elif line.startswith("EXPLANATION:"):
            explanation = line.replace("EXPLANATION:", "").strip()
        elif line.startswith("RECOMMENDATION:"):
            recommendation = line.replace("RECOMMENDATION:", "").strip()

    return ConflictResult(
        conflict_type  = conflict_type,
        severity       = severity,
        explanation    = explanation,
        chunk_a_text   = text_a,
        chunk_b_text   = text_b,
        chunk_a_page   = page_a,
        chunk_b_page   = page_b,
        chunk_a_doc    = doc_a,
        chunk_b_doc    = doc_b,
        recommendation = recommendation
    )


def detect_conflicts_in_results(
    chunks: list[dict],
    max_comparisons: int = 10
) -> list[ConflictResult]:
    """
    Check all retrieved chunks against each other for conflicts.

    We compare every pair of chunks from DIFFERENT documents.
    Same-document chunks are skipped — we only care about
    conflicts between different policies/regulations.

    max_comparisons limits how many pairs we check to
    avoid too many LLM calls.

    Returns only actual conflicts (not NONE results).
    """
    conflicts = []
    comparisons_done = 0

    print(f"[ConflictDetector] Checking {len(chunks)} chunks for conflicts...")

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if comparisons_done >= max_comparisons:
                break

            chunk_a = chunks[i]
            chunk_b = chunks[j]

            doc_a = chunk_a["metadata"].get("doc_name", "")
            doc_b = chunk_b["metadata"].get("doc_name", "")

            # only compare chunks from different documents
            if doc_a == doc_b:
                continue

            result = detect_conflict_between_chunks(chunk_a, chunk_b)
            comparisons_done += 1

            if result.conflict_type != "NONE":
                conflicts.append(result)
                print(f"[ConflictDetector] ⚠️  {result.conflict_type} conflict found "
                      f"between {doc_a} p{result.chunk_a_page} "
                      f"and {doc_b} p{result.chunk_b_page}")

    print(f"[ConflictDetector] Done. "
          f"{len(conflicts)} conflicts found "
          f"in {comparisons_done} comparisons.\n")

    return conflicts


def format_conflicts_for_response(
    conflicts: list[ConflictResult]
) -> list[dict]:
    """
    Format conflict results for API response.
    """
    formatted = []
    for c in conflicts:
        formatted.append({
            "conflict_type":  c.conflict_type,
            "severity":       c.severity,
            "explanation":    c.explanation,
            "recommendation": c.recommendation,
            "source_a": {
                "doc":  c.chunk_a_doc,
                "page": c.chunk_a_page,
                "text": c.chunk_a_text[:200] + "..."
            },
            "source_b": {
                "doc":  c.chunk_b_doc,
                "page": c.chunk_b_page,
                "text": c.chunk_b_text[:200] + "..."
            }
        })
    return formatted


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":

    # simulate two chunks from different documents
    # with a planted conflict for testing

    chunk_a = {
        "text": """Article 17 — Right to erasure.
        The data subject shall have the right to obtain 
        from the controller the erasure of personal data 
        concerning him or her without undue delay and the 
        controller shall have the obligation to erase 
        personal data without undue delay.""",
        "metadata": {
            "doc_name":    "GDPR_Official",
            "page_number": 44
        }
    }

    chunk_b = {
        "text": """Section 4.2 — Data Retention Policy.
        Customer transaction data and personal information 
        shall be retained for a minimum period of 90 days 
        after account closure for fraud prevention, 
        audit, and compliance purposes. 
        No deletion requests will be processed during 
        this retention period.""",
        "metadata": {
            "doc_name":    "Internal_Data_Policy",
            "page_number": 5
        }
    }

    print("="*60)
    print("CONFLICT DETECTION TEST")
    print("="*60)
    print(f"\nChunk A: GDPR Article 17 — right to erasure")
    print(f"Chunk B: Internal policy — 90 day retention")
    print(f"\nRunning conflict detection...")

    result = detect_conflict_between_chunks(chunk_a, chunk_b)

    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"{'='*60}")
    print(f"  Conflict Type  : {result.conflict_type}")
    print(f"  Severity       : {result.severity}")
    print(f"  Explanation    : {result.explanation}")
    print(f"  Recommendation : {result.recommendation}")

    if result.conflict_type != "NONE":
        print(f"\n⚠️  CONFLICT DETECTED between:")
        print(f"  {result.chunk_a_doc} (Page {result.chunk_a_page})")
        print(f"  {result.chunk_b_doc} (Page {result.chunk_b_page})")
    else:
        print(f"\n✅ No conflict detected")