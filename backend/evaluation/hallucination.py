# backend/evaluation/hallucination.py

import os
import sys
import ollama
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


@dataclass
class HallucinationResult:
    """
    Result of checking one answer against its source chunks.
    """
    is_hallucination:   bool    # True if answer contains unsupported claims
    confidence:         float   # 0.0 to 1.0 — how grounded the answer is
    supported_claims:   list    # claims found in source text
    unsupported_claims: list    # claims NOT found in source text
    verdict:            str     # GROUNDED / PARTIAL / HALLUCINATED
    recommendation:     str     # what to do


def split_into_claims(answer: str) -> list[str]:
    """
    Split an answer into individual factual claims.

    WHY:
    We can't check a whole paragraph at once.
    We split it into atomic claims and check each one.

    Example:
    Answer: "The maximum fine is 20M EUR. 
             This applies to all companies worldwide."

    Claims:
    1. "The maximum fine is 20M EUR"
    2. "This applies to all companies worldwide"
    """
    prompt = f"""Split this text into individual factual claims.
Each claim should be one sentence that can be verified independently.

Text: {answer}

Write each claim on a new line.
Write ONLY the claims. No numbering. No explanation."""

    response = ollama.chat(
        model    = OLLAMA_MODEL,
        messages = [{"role": "user", "content": prompt}]
    )

    raw    = response["message"]["content"].strip()
    claims = [
        c.strip()
        for c in raw.split("\n")
        if c.strip() and len(c.strip()) > 10
    ]

    return claims


def check_claim_against_context(
    claim:   str,
    context: str
) -> dict:
    """
    Check if a single claim is supported by the context.

    Uses Llama as an NLI (Natural Language Inference) model.
    Asks: does the context ENTAIL, CONTRADICT, or is NEUTRAL
    to this claim?

    WHY NLI FOR HALLUCINATION:
    A hallucination is a claim the LLM made that isn't
    supported by the retrieved context.
    NLI detects exactly this — unsupported claims.
    """
    prompt = f"""You are a fact-checking assistant.
Determine if the CLAIM is supported by the CONTEXT.

CONTEXT:
{context[:1000]}

CLAIM:
{claim}

Respond in EXACTLY this format:
VERDICT: [SUPPORTED or UNSUPPORTED or PARTIAL]
REASON: [One sentence explaining why]

SUPPORTED = the context clearly supports this claim
UNSUPPORTED = the context does not mention or contradicts this claim  
PARTIAL = the context partially supports this claim

Respond with ONLY the two lines above."""

    response = ollama.chat(
        model    = OLLAMA_MODEL,
        messages = [{"role": "user", "content": prompt}]
    )

    raw     = response["message"]["content"].strip()
    verdict = "UNSUPPORTED"
    reason  = "Could not parse response"

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return {
        "claim":   claim,
        "verdict": verdict,
        "reason":  reason
    }


def check_hallucination(
    answer: str,
    chunks: list[dict]
) -> HallucinationResult:
    """
    Main hallucination detection function.

    Steps:
    1. Split answer into atomic claims
    2. Build context from retrieved chunks
    3. Check each claim against context
    4. Calculate grounding score
    5. Return verdict

    GROUNDING SCORE:
    supported_claims / total_claims

    1.0 = fully grounded (no hallucination)
    0.7+ = mostly grounded (acceptable)
    0.5-0.7 = partial (warning)
    <0.5 = likely hallucinated (reject/retry)
    """
    if not answer or not chunks:
        return HallucinationResult(
            is_hallucination   = True,
            confidence         = 0.0,
            supported_claims   = [],
            unsupported_claims = [],
            verdict            = "HALLUCINATED",
            recommendation     = "No answer or context provided"
        )

    print(f"[Hallucination] Checking answer for hallucinations...")

    # build full context from all chunks
    context = "\n\n".join([
        f"[Page {c['metadata'].get('page_number','?')}]\n{c['text']}"
        for c in chunks
    ])

    # Step 1 — split into claims
    print(f"[Hallucination] Splitting answer into claims...")
    claims = split_into_claims(answer)
    print(f"[Hallucination] Found {len(claims)} claims to verify")

    if not claims:
        return HallucinationResult(
            is_hallucination   = False,
            confidence         = 1.0,
            supported_claims   = [],
            unsupported_claims = [],
            verdict            = "GROUNDED",
            recommendation     = "Answer is too short to verify"
        )

    # Step 2 — check each claim
    supported   = []
    unsupported = []
    partial     = []

    for i, claim in enumerate(claims, 1):
        print(f"[Hallucination] Checking claim {i}/{len(claims)}...")
        result = check_claim_against_context(claim, context)

        if result["verdict"] == "SUPPORTED":
            supported.append(result)
        elif result["verdict"] == "UNSUPPORTED":
            unsupported.append(result)
        else:
            partial.append(result)

    # Step 3 — calculate confidence score
    total      = len(claims)
    supported_count = len(supported) + (len(partial) * 0.5)
    confidence = round(supported_count / total, 2) if total > 0 else 0.0

    # Step 4 — determine verdict
    if confidence >= 0.6:
        verdict          = "GROUNDED"
        is_hallucination = False
        recommendation   = "Answer is well grounded in source documents."
    elif confidence >= 0.3:
        verdict          = "PARTIAL"
        is_hallucination = False
        recommendation   = (
            "Answer is partially grounded. "
            "Review unsupported claims before using."
        )
    else:
        verdict          = "HALLUCINATED"
        is_hallucination = True
        recommendation   = (
            "Answer contains unsupported claims. "
            "Do not use without manual verification."
        )

    print(f"[Hallucination] Verdict: {verdict} "
          f"(confidence: {confidence})")

    return HallucinationResult(
        is_hallucination   = is_hallucination,
        confidence         = confidence,
        supported_claims   = supported,
        unsupported_claims = unsupported,
        verdict            = verdict,
        recommendation     = recommendation
    )


def format_hallucination_for_response(
    result: HallucinationResult
) -> dict:
    """Format for API response."""
    return {
        "verdict":            result.verdict,
        "confidence":         result.confidence,
        "is_hallucination":   result.is_hallucination,
        "supported_claims":   len(result.supported_claims),
        "unsupported_claims": len(result.unsupported_claims),
        "recommendation":     result.recommendation,
        "details": {
            "supported": [
                c["claim"] for c in result.supported_claims
            ],
            "unsupported": [
                c["claim"] for c in result.unsupported_claims
            ]
        }
    }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":

    # test with a grounded answer
    answer_grounded = """The maximum fine for a GDPR violation 
    is 20,000,000 EUR, or up to 4% of the total worldwide 
    annual turnover of the preceding financial year, 
    whichever is higher. This applies to serious infringements 
    such as violations of basic principles for processing 
    and data subjects rights."""

    # test with a hallucinated answer
    answer_hallucinated = """The maximum fine for a GDPR violation 
    is 50,000,000 EUR. Companies must pay this fine within 
    24 hours of receiving notice. The fine doubles for 
    repeat offenders within 6 months."""

    chunks = [
        {
            "text": """5. Infringements of the following provisions 
            shall be subject to administrative fines up to 
            20 000 000 EUR, or in the case of an undertaking, 
            up to 4% of the total worldwide annual turnover 
            of the preceding financial year, whichever is higher.""",
            "metadata": {
                "page_number": 83,
                "doc_name": "GDPR"
            }
        }
    ]

    print("=" * 60)
    print("TEST 1 — GROUNDED ANSWER")
    print("=" * 60)
    result1 = check_hallucination(answer_grounded, chunks)
    formatted1 = format_hallucination_for_response(result1)
    print(f"\nVerdict    : {formatted1['verdict']}")
    print(f"Confidence : {formatted1['confidence']}")
    print(f"Supported  : {formatted1['supported_claims']} claims")
    print(f"Unsupported: {formatted1['unsupported_claims']} claims")
    print(f"Recommendation: {formatted1['recommendation']}")

    print("\n" + "=" * 60)
    print("TEST 2 — HALLUCINATED ANSWER")
    print("=" * 60)
    result2 = check_hallucination(answer_hallucinated, chunks)
    formatted2 = format_hallucination_for_response(result2)
    print(f"\nVerdict    : {formatted2['verdict']}")
    print(f"Confidence : {formatted2['confidence']}")
    print(f"Supported  : {formatted2['supported_claims']} claims")
    print(f"Unsupported: {formatted2['unsupported_claims']} claims")
    print(f"Recommendation: {formatted2['recommendation']}")