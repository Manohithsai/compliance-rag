# backend/main.py

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil

from backend.retrieval.hybrid_fusion        import HybridRetriever
from backend.query.multi_query_retriever    import multi_query_retrieve
from backend.reranker.cross_encoder         import CrossEncoderReranker
from backend.generation.generator           import generate_answer
from backend.ingestion.pipeline             import ingest_document
from backend.evaluation.conflict_detector   import (
    detect_conflicts_in_results,
    format_conflicts_for_response
)
from backend.evaluation.hallucination import (
    check_hallucination,
    format_hallucination_for_response
)
# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title       = "ComplianceRAG API",
    description = "Advanced AI-powered compliance document Q&A system",
    version     = "2.0.0"
)

# initialise components once at startup
print("[Main] Initializing pipeline components...")
retriever = HybridRetriever()
reranker  = CrossEncoderReranker()
print("[Main] All components ready.\n")

# ── Request/Response models ────────────────────────────────────
class QueryRequest(BaseModel):
    question:       str
    top_k:          int  = 5
    use_multi_query: bool = True
    use_reranker:   bool = True

class QueryResponse(BaseModel):
    question:        str
    answer:          str
    sources:         list[dict]
    conflicts:       list[dict]
    hallucination:   dict
    pipeline_used:   dict

# ── Routes ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   "llama3.2",
        "version": "2.0.0"
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Advanced RAG pipeline:
    Query → Rewrite → Multi-Query → Hybrid Search
    → Rerank → Generate → Answer
    """
    if not request.question.strip():
        raise HTTPException(
            status_code = 400,
            detail      = "Question cannot be empty"
        )

    pipeline_info = {
        "multi_query": request.use_multi_query,
        "reranker":    request.use_reranker,
        "queries_used": 1
    }

    # ── Step 1: Retrieval ─────────────────────────────────────
    if request.use_multi_query:
        # advanced: rewrite + multi-query + hybrid search
        retrieval_result = multi_query_retrieve(
            query     = request.question,
            retriever = retriever,
            top_k     = 20
        )
        chunks = retrieval_result["final_chunks"]
        pipeline_info["queries_used"]    = len(
            retrieval_result["queries_used"]
        )
        pipeline_info["total_retrieved"] = retrieval_result["total_retrieved"]
        pipeline_info["unique_chunks"]   = retrieval_result["unique_chunks"]

    else:
        # basic: single query hybrid search
        chunks = retriever.retrieve(
            query = request.question,
            top_k = 20
        )

    # ── Step 2: Reranking ─────────────────────────────────────
    if request.use_reranker and chunks:
        chunks = reranker.rerank(
            query  = request.question,
            chunks = chunks,
            top_k  = request.top_k
        )
        pipeline_info["reranked_to"] = len(chunks)
    else:
        chunks = chunks[:request.top_k]

    # ── Step 3: Conflict Detection ────────────────────────────
    conflicts = []
    if len(chunks) > 1:
        raw_conflicts = detect_conflicts_in_results(chunks)
        conflicts     = format_conflicts_for_response(raw_conflicts)

    # ── Step 4: Generation ────────────────────────────────────
        # ── Step 4: Generation ────────────────────────────────────
    result = generate_answer(request.question, chunks)

    # ── Step 5: Hallucination Check ───────────────────────────
    hallucination_result = check_hallucination(
        answer = result["answer"],
        chunks = chunks
    )
    hallucination = format_hallucination_for_response(
        hallucination_result
    )

    return QueryResponse(
        question      = request.question,
        answer        = result["answer"],
        sources       = result["sources"],
        conflicts     = conflicts,
        hallucination = hallucination,
        pipeline_used = pipeline_info
    )


@app.post("/query/basic")
def query_basic(request: QueryRequest):
    """
    Basic pipeline without rewriting or reranking.
    Useful for A/B comparison with advanced pipeline.
    """
    chunks = retriever.retrieve(
        query = request.question,
        top_k = request.top_k
    )
    result = generate_answer(request.question, chunks)

    return {
        "question":     request.question,
        "answer":       result["answer"],
        "sources":      result["sources"],
        "pipeline_used": {
            "multi_query": False,
            "reranker":    False,
            "queries_used": 1
        }
    }


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    """Upload and ingest a new PDF document."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code = 400,
            detail      = "Only PDF files accepted"
        )

    upload_path = f"./data/raw/{file.filename}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    summary = ingest_document(upload_path)

    return {
        "message": f"Successfully ingested {file.filename}",
        "summary": summary
    }


@app.get("/status")
def status():
    """Returns indexed document stats."""
    from backend.ingestion.indexer import (
        get_chroma_client,
        get_or_create_collection,
        get_collection_stats,
        CHILD_COLLECTION,
        PARENT_COLLECTION
    )
    client = get_chroma_client()
    child  = get_or_create_collection(client, CHILD_COLLECTION)
    parent = get_or_create_collection(client, PARENT_COLLECTION)

    return {
        "child_chunks":  get_collection_stats(child),
        "parent_chunks": get_collection_stats(parent),
        "pipeline":      "advanced"
    }