"""SQLite persistence for distillcore — optional storage extension."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from pathlib import Path

from ..models import ProcessingResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    source_filename TEXT NOT NULL,
    document_title  TEXT,
    document_type   TEXT DEFAULT 'unknown',
    page_count      INTEGER DEFAULT 0,
    full_text       TEXT NOT NULL,
    metadata_json   TEXT DEFAULT '{}',
    sections_json   TEXT DEFAULT '[]',
    validation_json TEXT DEFAULT '{}',
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id                TEXT PRIMARY KEY,
    document_id       TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index       INTEGER NOT NULL,
    text              TEXT NOT NULL,
    token_estimate    INTEGER NOT NULL,
    section_type      TEXT,
    section_heading   TEXT,
    page_start        INTEGER,
    page_end          INTEGER,
    speakers_json     TEXT,
    topic             TEXT,
    key_concepts_json TEXT DEFAULT '[]',
    relevance         TEXT,
    embedding_json    TEXT,
    created_at        TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

CREATE TABLE IF NOT EXISTS search_log (
    id              TEXT PRIMARY KEY,
    query           TEXT NOT NULL,
    result_count    INTEGER DEFAULT 0,
    top_chunk_ids   TEXT DEFAULT '[]',
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Store:
    """SQLite-backed document and chunk storage with cosine similarity search."""

    def __init__(self, path: Path | str = "~/.distillcore/store.db") -> None:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- Write -----------------------------------------------------------------

    def save(self, result: ProcessingResult) -> str:
        """Save a ProcessingResult (document + chunks). Returns document_id."""
        doc_id = str(uuid.uuid4())
        meta = result.document.metadata

        sections_data = [s.model_dump() for s in result.document.sections]
        validation_data = result.validation.model_dump()

        with self._lock:
            self._conn.execute(
                """INSERT INTO documents
                   (id, source_filename, document_title, document_type,
                    page_count, full_text, metadata_json, sections_json, validation_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    meta.source_filename,
                    meta.document_title,
                    meta.document_type,
                    meta.page_count,
                    result.document.full_text,
                    json.dumps(meta.extra),
                    json.dumps(sections_data),
                    json.dumps(validation_data),
                ),
            )

            for chunk in result.chunks:
                chunk_id = str(uuid.uuid4())
                self._conn.execute(
                    """INSERT INTO chunks
                       (id, document_id, chunk_index, text, token_estimate,
                        section_type, section_heading, page_start, page_end,
                        speakers_json, topic, key_concepts_json, relevance,
                        embedding_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk_id,
                        doc_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.token_estimate,
                        chunk.section_type,
                        chunk.section_heading,
                        chunk.page_start,
                        chunk.page_end,
                        json.dumps(chunk.speakers) if chunk.speakers else None,
                        chunk.topic,
                        json.dumps(chunk.key_concepts),
                        chunk.relevance,
                        json.dumps(chunk.embedding) if chunk.embedding else None,
                    ),
                )

            self._conn.commit()

        return doc_id

    # -- Read ------------------------------------------------------------------

    def get_document(self, document_id: str) -> dict | None:
        """Get a document by ID with metadata and validation."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        if not row:
            return None
        return _doc_row_to_dict(row)

    def list_documents(
        self,
        document_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List documents, optionally filtered by type."""
        if document_type:
            sql = "SELECT * FROM documents WHERE document_type = ? ORDER BY created_at DESC LIMIT ?"
            params: tuple = (document_type, limit)
        else:
            sql = "SELECT * FROM documents ORDER BY created_at DESC LIMIT ?"
            params = (limit,)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_doc_row_to_dict(r) for r in rows]

    def get_chunks(self, document_id: str) -> list[dict]:
        """Get all chunks for a document, ordered by chunk_index."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,),
            ).fetchall()
        return [_chunk_row_to_dict(r) for r in rows]

    # -- Search ----------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        document_type: str | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        """Cosine similarity search over stored chunks.

        Args:
            query_embedding: The query vector (same dimensionality as stored embeddings).
            top_k: Number of results to return.
            document_type: Optional filter by document type.
            document_id: Optional filter to a specific document.

        Returns:
            List of chunk dicts with a 'score' field (higher = more similar).
        """
        conditions = ["c.embedding_json IS NOT NULL"]
        params: list = []

        if document_type:
            conditions.append("d.document_type = ?")
            params.append(document_type)
        if document_id:
            conditions.append("c.document_id = ?")
            params.append(document_id)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT c.*, d.source_filename, d.document_type, d.document_title
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE {where}
        """

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        # Score and rank
        scored = []
        for row in rows:
            emb = json.loads(row["embedding_json"])
            score = _cosine_similarity(query_embedding, emb)
            result = _chunk_row_to_dict(row)
            result["score"] = score
            result["source_filename"] = row["source_filename"]
            result["document_type"] = row["document_type"]
            result["document_title"] = row["document_title"]
            scored.append(result)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def log_search(
        self,
        query: str,
        result_count: int,
        top_chunk_ids: list[str],
    ) -> None:
        """Log a search query for analytics."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO search_log (id, query, result_count, top_chunk_ids)"
                " VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), query, result_count, json.dumps(top_chunk_ids)),
            )
            self._conn.commit()

    # -- Delete ----------------------------------------------------------------

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks. Returns True if found."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM documents WHERE id = ?", (document_id,)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    # -- Stats -----------------------------------------------------------------

    def stats(self) -> dict:
        """Get aggregate stats about the store."""
        with self._lock:
            doc_count = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embedded_count = self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding_json IS NOT NULL"
            ).fetchone()[0]
            search_count = self._conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0]
            types = self._conn.execute(
                "SELECT document_type, COUNT(*) as cnt FROM documents GROUP BY document_type"
            ).fetchall()
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "chunks_with_embeddings": embedded_count,
            "searches": search_count,
            "document_types": {r["document_type"]: r["cnt"] for r in types},
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# -- Row helpers ---------------------------------------------------------------


def _doc_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "source_filename": row["source_filename"],
        "document_title": row["document_title"],
        "document_type": row["document_type"],
        "page_count": row["page_count"],
        "full_text": row["full_text"],
        "metadata": json.loads(row["metadata_json"]),
        "sections": json.loads(row["sections_json"]),
        "validation": json.loads(row["validation_json"]),
        "created_at": row["created_at"],
    }


def _chunk_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "document_id": row["document_id"],
        "chunk_index": row["chunk_index"],
        "text": row["text"],
        "token_estimate": row["token_estimate"],
        "section_type": row["section_type"],
        "section_heading": row["section_heading"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "speakers": json.loads(row["speakers_json"]) if row["speakers_json"] else None,
        "topic": row["topic"],
        "key_concepts": json.loads(row["key_concepts_json"]),
        "relevance": row["relevance"],
        "has_embedding": row["embedding_json"] is not None,
        "created_at": row["created_at"],
    }
