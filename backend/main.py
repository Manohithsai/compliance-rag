from fastapi import FastAPI

app = FastAPI(title="ComplianceRAG")

@app.get("/health")
def health():
    return {"status": "ok"}
