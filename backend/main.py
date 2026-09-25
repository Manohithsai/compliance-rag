# backend/main.py

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil

from backend.retrieval.hybrid_fusion import HybridRetriever
from backend.generation.generator   import generate_answer
from backend.ingestion.pipeline     import ingest_document

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title       = "ComplianceRAG API",
    description = "AI-powered compliance document Q&A system",
    version     = "1.0.0"
)

# initialise retriever once at startup — not on every request
retriever = HybridRetriever()

# ── Request/Response models ───────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k:    int = 5

class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[dict]

# ── Routes ───────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "llama3.2"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main endpoint. Takes a question, returns an answer with sources.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Step 1 — retrieve
    chunks = retriever.retrieve(
        query = request.question,
        top_k = request.top_k
    )

    # Step 2 — generate
    result = generate_answer(request.question, chunks)

    return QueryResponse(
        question = request.question,
        answer   = result["answer"],
        sources  = result["sources"]
    )


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF and ingest it into the system.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    # save uploaded file temporarily
    upload_path = f"./data/raw/{file.filename}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ingest it
    summary = ingest_document(upload_path)

    return {
        "message": f"Successfully ingested {file.filename}",
        "summary": summary
    }


@app.get("/status")
def status():
    """
    Returns how many chunks are indexed.
    """
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
        "parent_chunks": get_collection_stats(parent)
    }