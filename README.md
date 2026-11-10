# ComplianceRAG

AI-powered compliance and policy question-answering system using Retrieval-Augmented Generation (RAG).

Ask natural-language questions about regulations and policies (GDPR, HIPAA, SOC 2, internal HR/finance documents) and get answers that are cited, conflict-checked, hallucination-checked and audit-logged.

Everything runs locally: Llama 3.2 for generation, nomic-embed-text for embeddings, both through Ollama.

## What it does

For every question, the pipeline runs:
1. Query rewriting (step-back question) + HyDE (hypothetical answer) + multi-query expansion
2. Hybrid retrieval: dense search (ChromaDB) + keyword search (BM25), fused with Reciprocal Rank Fusion
3. Cross-encoder reranking (FlashRank, ms-marco-MiniLM-L-12-v2)
4. Answer generation with Llama 3.2
5. Cross-document conflict detection (LLM-based)
6. Claim-by-claim hallucination check
7. Audit logging of every query (SQLite), viewable through the /audit endpoint

Documents are ingested with hierarchical parent/child chunking (512-word parent chunks for context, 128-word child chunks for precise retrieval, 20-word overlap) plus metadata.

## Requirements

- Python 3.10 or newer
- Ollama installed (https://ollama.com)

## Setup

1. Pull the two Ollama models:

```
ollama pull llama3.2
ollama pull nomic-embed-text
```

2. Clone the repo and install dependencies.

Windows (PowerShell):
```
git clone https://github.com/Manohithsai/compliance-rag.git
cd compliance-rag
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Mac/Linux:
```
git clone https://github.com/Manohithsai/compliance-rag.git
cd compliance-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

Terminal 1 - start the backend API:
```
uvicorn backend.main:app --reload
```

Terminal 2 - start the UI:
```
streamlit run frontend/app.py
```

Then open http://localhost:8501 in your browser.

## Adding documents

Put your PDF files in the `data/raw` folder, then either upload them through the Streamlit UI or run:

```
python backend/ingestion/pipeline.py
```

## API endpoints

- `GET /health` - health check
- `POST /query` - full pipeline (rewriting, hybrid search, reranking, conflicts, hallucination check)
- `POST /query/basic` - simple single-query hybrid search
- `POST /ingest` - ingest a PDF
- `GET /status` - index status
- `GET /audit` - audit logs

## Evaluation

`evaluation/quick_eval.py` runs 10 GDPR questions with known answer pages and prints Recall@5 for dense, BM25 and hybrid retrieval:

```
python evaluation/quick_eval.py
```

(Requires the GDPR PDF in `data/raw`.)

## Project structure

- `backend/ingestion` - PDF loading, parent/child chunking, embedding, ChromaDB + BM25 indexing
- `backend/query` - query rewriting, HyDE, multi-query retrieval
- `backend/retrieval` - dense retriever, BM25 retriever, hybrid fusion (RRF)
- `backend/reranker` - FlashRank cross-encoder reranking
- `backend/generation` - answer generation with Ollama
- `backend/evaluation` - conflict detection, hallucination check
- `backend/mcp` - audit logging (access control and integrity checks planned)
- `frontend` - Streamlit UI
- `evaluation` - quick_eval.py benchmark script

## Current limitations

- The ReAct agent (`backend/query/agent.py`), role-based access control, and the RAGAS 200-question benchmark are planned but not implemented yet.
- Conflict detection is LLM-based (Llama 3.2), not a trained NLI model.
- The `mcp` folder currently holds the audit logger; it is not a full MCP server yet.
- The quick evaluation covers 10 questions; a larger benchmark is planned.
