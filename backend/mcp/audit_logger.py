# backend/mcp/audit_logger.py

import os
import sys
import sqlite3
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# store audit log in data folder
AUDIT_DB_PATH = "./data/processed/audit_log.db"


def init_audit_db() -> None:
    """
    Create audit log table if it doesn't exist.
    Called once when the app starts.

    WHY SQLITE:
    Simple, file-based, no server needed.
    Perfect for single-instance deployment.
    In production you'd use PostgreSQL.
    """
    os.makedirs(os.path.dirname(AUDIT_DB_PATH), exist_ok=True)

    conn = sqlite3.connect(AUDIT_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT NOT NULL,
            question          TEXT NOT NULL,
            answer            TEXT NOT NULL,
            sources           TEXT NOT NULL,
            conflicts_found   INTEGER DEFAULT 0,
            hallucination     TEXT,
            confidence        REAL,
            verdict           TEXT,
            queries_used      INTEGER,
            chunks_retrieved  INTEGER,
            pipeline          TEXT,
            response_time_ms  REAL
        )
    """)

    conn.commit()
    conn.close()
    print(f"[AuditLogger] Database ready at {AUDIT_DB_PATH}")


def log_query(
    question:         str,
    answer:           str,
    sources:          list[dict],
    conflicts:        list[dict],
    hallucination:    dict,
    pipeline_used:    dict,
    response_time_ms: float = 0.0
) -> int:
    """
    Log a query and its response to the audit database.

    WHY AUDIT LOGGING MATTERS FOR COMPLIANCE:
    In regulated industries, organizations must prove
    what their AI system told users and when.

    If a compliance officer asks:
    "What did the system tell our team about GDPR 
     Article 17 on March 15th?"

    You can answer exactly — query, answer, sources,
    confidence score, timestamp. All in the log.

    Returns the log entry ID.
    """
    conn   = sqlite3.connect(AUDIT_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO audit_log (
            timestamp,
            question,
            answer,
            sources,
            conflicts_found,
            hallucination,
            confidence,
            verdict,
            queries_used,
            chunks_retrieved,
            pipeline,
            response_time_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        question,
        answer,
        json.dumps(sources),
        len(conflicts),
        json.dumps(hallucination),
        hallucination.get("confidence", 0.0),
        hallucination.get("verdict", "UNKNOWN"),
        pipeline_used.get("queries_used", 1),
        pipeline_used.get("unique_chunks", 0),
        json.dumps(pipeline_used),
        response_time_ms
    ))

    conn.commit()
    log_id = cursor.lastrowid
    conn.close()

    print(f"[AuditLogger] Logged query #{log_id} "
          f"— verdict: {hallucination.get('verdict','?')} "
          f"— conflicts: {len(conflicts)}")

    return log_id


def get_audit_logs(limit: int = 50) -> list[dict]:
    """
    Retrieve recent audit logs.
    Used by the /audit endpoint and dashboard.
    """
    conn   = sqlite3.connect(AUDIT_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            id, timestamp, question, answer,
            conflicts_found, confidence, verdict,
            queries_used, chunks_retrieved, response_time_ms
        FROM audit_log
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    logs = []
    for row in rows:
        logs.append({
            "id":               row[0],
            "timestamp":        row[1],
            "question":         row[2],
            "answer":           row[3][:200] + "...",
            "conflicts_found":  row[4],
            "confidence":       row[5],
            "verdict":          row[6],
            "queries_used":     row[7],
            "chunks_retrieved": row[8],
            "response_time_ms": row[9]
        })

    return logs


def get_audit_stats() -> dict:
    """
    Get summary statistics from audit log.
    Used for the dashboard.
    """
    conn   = sqlite3.connect(AUDIT_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM audit_log")
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM audit_log 
        WHERE verdict = 'HALLUCINATED'
    """)
    hallucinated = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM audit_log 
        WHERE conflicts_found > 0
    """)
    with_conflicts = cursor.fetchone()[0]

    cursor.execute("""
        SELECT AVG(confidence) FROM audit_log
    """)
    avg_confidence = cursor.fetchone()[0] or 0.0

    cursor.execute("""
        SELECT AVG(response_time_ms) FROM audit_log
    """)
    avg_response_time = cursor.fetchone()[0] or 0.0

    conn.close()

    return {
        "total_queries":      total,
        "hallucinated":       hallucinated,
        "with_conflicts":     with_conflicts,
        "avg_confidence":     round(avg_confidence, 2),
        "avg_response_time":  round(avg_response_time, 2)
    }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":

    # initialize DB
    init_audit_db()

    # log a test entry
    log_id = log_query(
        question      = "What is the maximum fine for GDPR?",
        answer        = "The maximum fine is 20,000,000 EUR...",
        sources       = [{"page": 83, "doc": "GDPR"}],
        conflicts     = [],
        hallucination = {
            "verdict":    "GROUNDED",
            "confidence": 0.85
        },
        pipeline_used = {
            "queries_used":   6,
            "unique_chunks":  15,
            "multi_query":    True,
            "reranker":       True
        },
        response_time_ms = 4230.5
    )

    print(f"\n[Test] Logged entry ID: {log_id}")

    # retrieve logs
    logs = get_audit_logs(limit=5)
    print(f"\n[Test] Recent logs:")
    for log in logs:
        print(f"  #{log['id']} | {log['timestamp'][:19]} | "
              f"verdict={log['verdict']} | "
              f"confidence={log['confidence']}")

    # stats
    stats = get_audit_stats()
    print(f"\n[Test] Audit stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")