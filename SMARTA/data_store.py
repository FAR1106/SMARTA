"""Local SQLite corpus used by SMARTA for durable records and full-text search."""

from __future__ import annotations

import json
import hashlib
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from document_loader import DATA_DIR, get_document_bundle, mysql_credentials_available

DB_PATH = DATA_DIR / "smarta.sqlite3"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_corpus(force: bool = False, database_url_override: Any = None) -> dict[str, int]:
    """Ingest all local source files and configured SQL records into SQLite."""
    with _connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY, source TEXT NOT NULL, locator TEXT NOT NULL,
            text TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}'
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS maintenance_logs (
            id INTEGER PRIMARY KEY, machine TEXT, fault TEXT, symptoms TEXT,
            cause TEXT, action TEXT, engineer_notes TEXT, source TEXT
        )""")
        db.execute("CREATE TABLE IF NOT EXISTS ingestion_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(text, source, locator)")
        files = sorted(
            path for path in DATA_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".xlsx", ".xls"}
        )
        signature_source = "\n".join(f"{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}" for path in files)
        signature_source += "\nSQL configured: " + ("yes" if mysql_credentials_available() else "no")
        signature = hashlib.sha256(signature_source.encode()).hexdigest()
        existing = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        old_signature = db.execute("SELECT value FROM ingestion_state WHERE key='signature'").fetchone()
        if existing and not force and old_signature and old_signature[0] == signature:
            return corpus_stats(db)

        documents, maintenance_rows = get_document_bundle(database_url_override)
        db.execute("DELETE FROM documents_fts")
        db.execute("DELETE FROM documents")
        db.execute("DELETE FROM maintenance_logs")
        db.executemany(
            "INSERT INTO documents(source, locator, text, metadata) VALUES (?, ?, ?, ?)",
            [(
                str(doc.get("source", "unknown")),
                str(doc.get("locator", "")),
                str(doc.get("text", "")),
                json.dumps(doc.get("metadata", {}), ensure_ascii=False, default=str),
            ) for doc in documents],
        )
        db.executemany(
            "INSERT INTO maintenance_logs(machine, fault, symptoms, cause, action, engineer_notes, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(r["machine"], r["fault"], r["symptoms"], r["cause"], r["action"], r["engineer_notes"], r["source"]) for r in maintenance_rows],
        )
        db.executemany(
            "INSERT INTO documents_fts(rowid, text, source, locator) VALUES (?, ?, ?, ?)",
            [(doc_id, source, locator, text) for doc_id, source, locator, text in db.execute(
                "SELECT id, source, locator, text FROM documents ORDER BY id"
            ).fetchall()],
        )
        db.execute("INSERT OR REPLACE INTO ingestion_state(key, value) VALUES ('signature', ?)", (signature,))
        return corpus_stats(db)


def corpus_stats(db: sqlite3.Connection | None = None) -> dict[str, int]:
    own_connection = db is None
    db = db or _connect()
    try:
        return {
            "documents": int(db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]),
            "maintenance_records": int(db.execute("SELECT COUNT(*) FROM maintenance_logs").fetchone()[0]),
            "sources": int(db.execute("SELECT COUNT(DISTINCT source) FROM documents").fetchone()[0]),
        }
    finally:
        if own_connection:
            db.close()


def list_maintenance_records(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """Read paginated maintenance rows from SMARTA's local SQLite corpus."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    with _connect() as db:
        total = int(db.execute("SELECT COUNT(*) FROM maintenance_logs").fetchone()[0])
        records = db.execute(
            """SELECT id, machine, fault, symptoms, cause, action,
                      engineer_notes, source
               FROM maintenance_logs ORDER BY id DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [dict(record) for record in records],
        }


def _fts_expression(query: str) -> str:
    terms = re.findall(r"[\w-]+", query, flags=re.UNICODE)
    # OR makes the retriever useful for natural-language questions; BM25 ranks
    # records containing more of the user's terms first.
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms[:24])


def retrieve(query: str, limit: int = 6) -> list[dict[str, Any]]:
    raw_terms = re.findall(r"[\w-]+", query.lower(), flags=re.UNICODE)
    expansion_map = {
        "overheat": ["overheating", "over heated", "over-heating"],
        "overheating": ["overheat", "over heated", "over-heating"],
        "tripping": ["trip", "tripped", "trips"],
        "tripped": ["trip", "tripping", "trips"],
        "failed": ["failure", "fault"],
        "failure": ["failed", "fault"],
    }
    expanded_terms = list(dict.fromkeys(term for raw in raw_terms for term in [raw, *expansion_map.get(raw, [])]))
    expression = _fts_expression(" ".join(expanded_terms))
    if not expression:
        return []
    with _connect() as db:
        try:
            rows = db.execute(
                """SELECT d.id, d.source, d.locator, d.text, d.metadata,
                          bm25(documents_fts, 1.0, 0.15, 0.3) AS score
                   FROM documents_fts f JOIN documents d ON d.id = f.rowid
                   WHERE documents_fts MATCH ? ORDER BY score LIMIT ?""",
                (expression, 120),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        total_docs = max(1, db.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        query_terms = list(dict.fromkeys(expanded_terms))
        stop_words = {"what", "why", "how", "is", "are", "the", "a", "an", "for", "of", "to", "in", "on", "with", "and", "or", "my", "it", "this", "that", "could", "would", "should", "recorded", "cause", "action"}
        query_terms = [term for term in query_terms if term not in stop_words]
        idf: dict[str, float] = {}
        if query_terms:
            # Keep the same per-token FTS document-frequency calculation, but
            # submit it as one SQLite statement instead of one round trip per
            # token. No embedding is used by this lexical retrieval pipeline.
            df_query = " UNION ALL ".join(
                "SELECT ? AS term, (SELECT COUNT(*) FROM documents_fts WHERE documents_fts MATCH ?) AS df"
                for _ in query_terms
            )
            df_params = tuple(
                value for term in query_terms for value in (term, f'"{term}"')
            )
            try:
                df_rows = db.execute(df_query, df_params).fetchall()
            except sqlite3.OperationalError:
                df_rows = []
            document_frequency = {row["term"]: row["df"] for row in df_rows}
            for term in query_terms:
                df = document_frequency.get(term, 0)
                # Unseen identifier tokens add no lexical-search penalty.
                idf[term] = math.log((total_docs + 1) / (df + 1)) if df else 0.0

        ranked = []
        equipment_terms = {"compressor", "pump", "boiler", "lathe", "conveyor", "motor", "transformer", "refrigerator", "forklift"}
        requested_equipment = equipment_terms.intersection(query_terms)
        for row in rows:
            metadata = json.loads(row["metadata"] or "{}")
            text = row["text"]
            lowered = text.lower()
            lowered = re.sub(r"over[\W\s]*heating", "overheating", lowered)
            if requested_equipment and not any(re.search(rf"\b{re.escape(term)}s?\b", lowered) for term in requested_equipment):
                continue
            relevance = sum(
                weight * min(2, lowered.count(term)) * (2.5 if term in {"compressor", "pump", "motor", "machine"} else 1.0)
                for term, weight in idf.items()
            )
            relevance += sum(8.0 for term in query_terms if len(term) > 1 and re.fullmatch(r"[a-z]+\d+[a-z\d]*", term) and re.search(rf"(?<![\w]){re.escape(term)}(?![\w])", lowered))
            ranked.append((relevance, {
                "text": row["text"],
                "source": row["source"],
                "locator": row["locator"],
                "metadata": metadata,
                "score": float(row["score"]),
            }))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        if not ranked or ranked[0][0] <= 0:
            return []
        relevance_floor = ranked[0][0] * 0.42
        unique_results = []
        seen_locations = set()
        for relevance, item in ranked:
            location_key = (item["source"], item["locator"])
            if relevance < relevance_floor or location_key in seen_locations:
                continue
            seen_locations.add(location_key)
            unique_results.append(item)
            if len(unique_results) == limit:
                break
        return unique_results
