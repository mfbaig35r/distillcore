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
            # v0.1.1 migration: embedding_dim column
            try:
                self._conn.execute(
                    "ALTER TABLE documents ADD COLUMN embedding_dim INTEGER"
                )
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists
            # v0.3.1 migration: tenant_id column
            for ddl in [
                "ALTER TABLE documents ADD COLUMN tenant_id TEXT",
            ]:
                try:
                    self._conn.execute(ddl)
                    self._conn.commit()
                except sqlite3.OperationalError:
                    pass
            # Create index outside executescript (may already exist)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_tenant_id "
                "ON documents(tenant_id)"
            )
            self._conn.commit()

    # -- Write -----------------------------------------------------------------

    def save(self, result: ProcessingResult, tenant_id: str | None = None) -> str:
        """Save a ProcessingResult (document + chunks). Returns document_id."""
        doc_id = str(uuid.uuid4())
        meta = result.document.metadata

        sections_data = [s.model_dump() for s in result.document.sections]
        validation_data = result.validation.model_dump()

        with self._lock:
            with self._conn:  # transaction — auto COMMIT/ROLLBACK
                self._conn.execute(
                    """INSERT INTO documents
                       (id, source_filename, document_title, document_type,
                        page_count, full_text, metadata_json, sections_json,
                        validation_json, tenant_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        tenant_id,
                    ),
                )

                embedding_dim = None
                for chunk in result.chunks:
                    chunk_id = str(uuid.uuid4())
                    emb_json = json.dumps(chunk.embedding) if chunk.embedding else None
                    if chunk.embedding and embedding_dim is None:
                        embedding_dim = len(chunk.embedding)
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
                            emb_json,
                        ),
                    )

                if embedding_dim is not None:
                    self._conn.execute(
                        "UPDATE documents SET embedding_dim = ? WHERE id = ?",
                        (embedding_dim, doc_id),
                    )

        return doc_id

    # -- Read ------------------------------------------------------------------

    def get_document(
        self, document_id: str, tenant_id: str | None = None
    ) -> dict | None:
        """Get a document by ID with metadata and validation."""
        if tenant_id:
            sql = "SELECT * FROM documents WHERE id = ? AND tenant_id = ?"
            params_get: tuple = (document_id, tenant_id)
        else:
            sql = "SELECT * FROM documents WHERE id = ?"
            params_get = (document_id,)
        with self._lock:
            row = self._conn.execute(sql, params_get).fetchone()
        if not row:
            return None
        return _doc_row_to_dict(row)

    def list_documents(
        self,
        document_type: str | None = None,
        limit: int = 50,
        tenant_id: str | None = None,
    ) -> list[dict]:
        """List documents, optionally filtered by type and/or tenant."""
        conditions = []
        params_list: list = []
        if document_type:
            conditions.append("document_type = ?")
            params_list.append(document_type)
        if tenant_id:
            conditions.append("tenant_id = ?")
            params_list.append(tenant_id)
        where = f"WHERE {' AND '.join(conditions)} " if conditions else ""
        sql = f"SELECT * FROM documents {where}ORDER BY created_at DESC LIMIT ?"
        params_list.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params_list).fetchall()
        return [_doc_row_to_dict(r) for r in rows]

    def get_chunks(self, document_id: str, tenant_id: str | None = None) -> list[dict]:
        """Get all chunks for a document, ordered by chunk_index."""
        # Verify tenant ownership if tenant_id is provided
        if tenant_id:
            doc = self.get_document(document_id, tenant_id=tenant_id)
            if not doc:
                return []
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
        tenant_id: str | None = None,
    ) -> list[dict]:
        """Cosine similarity search over stored chunks.

        Args:
            query_embedding: The query vector (same dimensionality as stored embeddings).
            top_k: Number of results to return.
            document_type: Optional filter by document type.
            document_id: Optional filter to a specific document.
            tenant_id: Optional tenant isolation.

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
        if tenant_id:
            conditions.append("d.tenant_id = ?")
            params.append(tenant_id)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT c.*, d.source_filename, d.document_type, d.document_title
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE {where}
        """

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        if not rows:
            return []

        # Validate embedding dimension
        first_emb = json.loads(rows[0]["embedding_json"])
        if len(query_embedding) != len(first_emb):
            raise ValueError(
                f"Query embedding dimension ({len(query_embedding)}) doesn't match "
                f"stored dimension ({len(first_emb)})"
            )

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

    def delete_document(self, document_id: str, tenant_id: str | None = None) -> bool:
        """Delete a document and its chunks. Returns True if found."""
        if tenant_id:
            sql = "DELETE FROM documents WHERE id = ? AND tenant_id = ?"
            params_del: tuple = (document_id, tenant_id)
        else:
            sql = "DELETE FROM documents WHERE id = ?"
            params_del = (document_id,)
        with self._lock:
            cursor = self._conn.execute(sql, params_del)
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
