"""Tests for distillcore.storage."""

from pathlib import Path

import pytest

from distillcore.models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    ProcessingResult,
    Section,
    ValidationReport,
)
from distillcore.storage import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "test.db")


@pytest.fixture
def sample_result() -> ProcessingResult:
    return ProcessingResult(
        document=Document(
            metadata=DocumentMetadata(
                source_filename="report.pdf",
                document_title="Q4 Report",
                document_type="report",
                page_count=5,
                extra={"author": "Jane Doe"},
            ),
            sections=[Section(heading="Intro", content="Introduction text")],
            full_text="Introduction text. Details here.",
        ),
        chunks=[
            DocumentChunk(
                chunk_index=0,
                text="Introduction text.",
                token_estimate=3,
                section_type="header",
                section_heading="Intro",
                topic="Introduction",
                key_concepts=["overview"],
                relevance="high",
                embedding=[0.1, 0.2, 0.3],
            ),
            DocumentChunk(
                chunk_index=1,
                text="Details here.",
                token_estimate=2,
                section_type="body",
                topic="Details",
                key_concepts=["specifics"],
                relevance="medium",
                embedding=[0.4, 0.5, 0.6],
            ),
        ],
        validation=ValidationReport(
            structuring_coverage=0.95,
            chunking_coverage=1.0,
            end_to_end_coverage=1.0,
            passed=True,
        ),
    )


class TestSave:
    def test_save_returns_id(self, store: Store, sample_result: ProcessingResult) -> None:
        doc_id = store.save(sample_result)
        assert isinstance(doc_id, str)
        assert len(doc_id) == 36  # uuid4

    def test_save_multiple(self, store: Store, sample_result: ProcessingResult) -> None:
        id1 = store.save(sample_result)
        id2 = store.save(sample_result)
        assert id1 != id2


class TestGetDocument:
    def test_found(self, store: Store, sample_result: ProcessingResult) -> None:
        doc_id = store.save(sample_result)
        doc = store.get_document(doc_id)
        assert doc is not None
        assert doc["source_filename"] == "report.pdf"
        assert doc["document_title"] == "Q4 Report"
        assert doc["document_type"] == "report"
        assert doc["page_count"] == 5
        assert doc["metadata"]["author"] == "Jane Doe"
        assert len(doc["sections"]) == 1
        assert doc["validation"]["passed"] is True

    def test_not_found(self, store: Store) -> None:
        assert store.get_document("nonexistent") is None


class TestListDocuments:
    def test_list_all(self, store: Store, sample_result: ProcessingResult) -> None:
        store.save(sample_result)
        store.save(sample_result)
        docs = store.list_documents()
        assert len(docs) == 2

    def test_filter_by_type(self, store: Store, sample_result: ProcessingResult) -> None:
        store.save(sample_result)
        assert len(store.list_documents(document_type="report")) == 1
        assert len(store.list_documents(document_type="memo")) == 0

    def test_limit(self, store: Store, sample_result: ProcessingResult) -> None:
        for _ in range(5):
            store.save(sample_result)
        assert len(store.list_documents(limit=3)) == 3

    def test_empty(self, store: Store) -> None:
        assert store.list_documents() == []


class TestGetChunks:
    def test_returns_chunks(self, store: Store, sample_result: ProcessingResult) -> None:
        doc_id = store.save(sample_result)
        chunks = store.get_chunks(doc_id)
        assert len(chunks) == 2
        assert chunks[0]["chunk_index"] == 0
        assert chunks[1]["chunk_index"] == 1
        assert chunks[0]["topic"] == "Introduction"
        assert chunks[0]["key_concepts"] == ["overview"]
        assert chunks[0]["has_embedding"] is True

    def test_empty_for_unknown(self, store: Store) -> None:
        assert store.get_chunks("nonexistent") == []


class TestSearch:
    def test_basic_search(self, store: Store, sample_result: ProcessingResult) -> None:
        store.save(sample_result)
        results = store.search(query_embedding=[0.1, 0.2, 0.3], top_k=5)
        assert len(results) == 2
        # First result should be most similar to [0.1, 0.2, 0.3]
        assert results[0]["text"] == "Introduction text."
        assert results[0]["score"] > results[1]["score"]
        assert "source_filename" in results[0]

    def test_top_k(self, store: Store, sample_result: ProcessingResult) -> None:
        store.save(sample_result)
        results = store.search(query_embedding=[0.1, 0.2, 0.3], top_k=1)
        assert len(results) == 1

    def test_filter_by_type(self, store: Store, sample_result: ProcessingResult) -> None:
        store.save(sample_result)
        results = store.search(
            query_embedding=[0.1, 0.2, 0.3], document_type="report"
        )
        assert len(results) == 2
        results = store.search(
            query_embedding=[0.1, 0.2, 0.3], document_type="memo"
        )
        assert len(results) == 0

    def test_filter_by_document_id(self, store: Store, sample_result: ProcessingResult) -> None:
        doc_id = store.save(sample_result)
        store.save(sample_result)  # second doc
        results = store.search(
            query_embedding=[0.1, 0.2, 0.3], document_id=doc_id
        )
        assert all(r["document_id"] == doc_id for r in results)

    def test_dimension_mismatch_raises(self, store: Store, sample_result: ProcessingResult) -> None:
        store.save(sample_result)  # embeddings are 3d
        with pytest.raises(ValueError, match="dimension"):
            store.search(query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5])  # 5d query

    def test_no_embeddings(self, store: Store) -> None:
        result = ProcessingResult(
            document=Document(
                metadata=DocumentMetadata(source_filename="test.txt"),
                full_text="text",
            ),
            chunks=[DocumentChunk(chunk_index=0, text="text", token_estimate=1)],
            validation=ValidationReport(),
        )
        store.save(result)
        results = store.search(query_embedding=[0.1, 0.2, 0.3])
        assert len(results) == 0  # no embeddings to search


class TestDelete:
    def test_delete_existing(self, store: Store, sample_result: ProcessingResult) -> None:
        doc_id = store.save(sample_result)
        assert store.delete_document(doc_id) is True
        assert store.get_document(doc_id) is None
        assert store.get_chunks(doc_id) == []  # cascade delete

    def test_delete_nonexistent(self, store: Store) -> None:
        assert store.delete_document("nonexistent") is False


class TestTenantIsolation:
    def test_save_and_get_with_tenant(
        self, store: Store, sample_result: ProcessingResult
    ) -> None:
        doc_id = store.save(sample_result, tenant_id="tenant_a")
        assert store.get_document(doc_id, tenant_id="tenant_a") is not None
        assert store.get_document(doc_id, tenant_id="tenant_b") is None

    def test_list_scoped_to_tenant(
        self, store: Store, sample_result: ProcessingResult
    ) -> None:
        store.save(sample_result, tenant_id="tenant_a")
        store.save(sample_result, tenant_id="tenant_b")
        assert len(store.list_documents(tenant_id="tenant_a")) == 1
        assert len(store.list_documents(tenant_id="tenant_b")) == 1
        assert len(store.list_documents()) == 2  # no tenant filter = all

    def test_get_chunks_scoped(
        self, store: Store, sample_result: ProcessingResult
    ) -> None:
        doc_id = store.save(sample_result, tenant_id="tenant_a")
        assert len(store.get_chunks(doc_id, tenant_id="tenant_a")) == 2
        assert store.get_chunks(doc_id, tenant_id="tenant_b") == []

    def test_search_scoped(
        self, store: Store, sample_result: ProcessingResult
    ) -> None:
        store.save(sample_result, tenant_id="tenant_a")
        store.save(sample_result, tenant_id="tenant_b")
        results_a = store.search([0.1, 0.2, 0.3], tenant_id="tenant_a")
        results_b = store.search([0.1, 0.2, 0.3], tenant_id="tenant_b")
        assert len(results_a) == 2
        assert len(results_b) == 2

    def test_delete_scoped(
        self, store: Store, sample_result: ProcessingResult
    ) -> None:
        doc_id = store.save(sample_result, tenant_id="tenant_a")
        assert store.delete_document(doc_id, tenant_id="tenant_b") is False  # wrong tenant
        assert store.delete_document(doc_id, tenant_id="tenant_a") is True

    def test_no_tenant_backwards_compatible(
        self, store: Store, sample_result: ProcessingResult
    ) -> None:
        doc_id = store.save(sample_result)  # no tenant
        assert store.get_document(doc_id) is not None
        assert len(store.list_documents()) == 1


class TestStats:
    def test_empty(self, store: Store) -> None:
        s = store.stats()
        assert s["documents"] == 0
        assert s["chunks"] == 0
        assert s["searches"] == 0

    def test_with_data(self, store: Store, sample_result: ProcessingResult) -> None:
        store.save(sample_result)
        s = store.stats()
        assert s["documents"] == 1
        assert s["chunks"] == 2
        assert s["chunks_with_embeddings"] == 2
        assert s["document_types"] == {"report": 1}


class TestSearchLog:
    def test_log_search(self, store: Store) -> None:
        store.log_search("test query", result_count=5, top_chunk_ids=["a", "b"])
        s = store.stats()
        assert s["searches"] == 1


class TestSearchCacheInvalidation:
    """Cover the numpy matrix cache built in Store._rebuild_matrix_cache_locked."""

    def test_cache_built_on_first_search(
        self, store: Store, sample_result: ProcessingResult
    ) -> None:
        store.save(sample_result)
        assert store._matrix_cache is None  # not built yet
        results = store.search([0.1, 0.2, 0.3])
        assert len(results) == 2
        assert store._matrix_cache is not None
        # Cache holds 2 chunks (both have embeddings in sample_result)
        _, matrix, chunk_ids, _ = store._matrix_cache
        assert matrix is not None
        assert matrix.shape[0] == 2
        assert len(chunk_ids) == 2

    def test_save_invalidates_cache(
        self, store: Store, sample_result: ProcessingResult
    ) -> None:
        store.save(sample_result)
        store.search([0.1, 0.2, 0.3])  # build cache
        version_before = store._matrix_cache[0]
        store.save(sample_result)  # second save
        # Cache itself isn't rebuilt yet — lazy
        results = store.search([0.1, 0.2, 0.3])
        # ...but it WAS rebuilt during this search
        version_after = store._matrix_cache[0]
        assert version_after > version_before
        # 4 chunks now (2 docs × 2 chunks each)
        assert len(results) == 4

    def test_delete_invalidates_cache(
        self, store: Store, sample_result: ProcessingResult
    ) -> None:
        doc_id_1 = store.save(sample_result)
        store.save(sample_result)
        store.search([0.1, 0.2, 0.3])  # build cache covering 4 chunks
        version_before = store._matrix_cache[0]
        deleted = store.delete_document(doc_id_1)
        assert deleted is True
        results = store.search([0.1, 0.2, 0.3])
        version_after = store._matrix_cache[0]
        assert version_after > version_before
        assert len(results) == 2  # only the surviving doc's chunks

    def test_delete_nonexistent_does_not_invalidate(
        self, store: Store, sample_result: ProcessingResult
    ) -> None:
        store.save(sample_result)
        store.search([0.1, 0.2, 0.3])  # build cache
        version_before = store._matrix_cache[0]
        store.delete_document("does-not-exist")
        # search() rebuild check uses _matrix_version — that didn't change
        store.search([0.1, 0.2, 0.3])
        assert store._matrix_cache[0] == version_before

    def test_score_order_matches_python_fallback(
        self, store: Store, sample_result: ProcessingResult
    ) -> None:
        """numpy path should produce the same top-K order as the Python loop."""
        store.save(sample_result)
        query = [0.5, 0.5, 0.5]

        # numpy path (default)
        np_results = store.search(query, top_k=2)

        # Force Python fallback by stashing numpy
        original_np = store._np
        store._np = None
        try:
            py_results = store.search(query, top_k=2)
        finally:
            store._np = original_np

        assert [r["chunk_index"] for r in np_results] == [
            r["chunk_index"] for r in py_results
        ]
        # Scores within a small float tolerance
        for np_r, py_r in zip(np_results, py_results):
            assert abs(np_r["score"] - py_r["score"]) < 1e-5
