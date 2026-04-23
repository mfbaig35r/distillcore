# distillcore Roadmap

## v0.5.0 — Data Format Extractors

### CSV Extractor
- **File**: `extractors/csv.py`
- **Formats**: `csv`, `tsv`
- **Deps**: none (stdlib `csv`)
- **Approach**: `csv.Sniffer` for delimiter detection, rows joined with `\n`, cells tab-separated. Header row preserved.
- **Metadata**: column names, row count, detected delimiter
- **Effort**: small

### Excel Extractor
- **File**: `extractors/excel.py`
- **Formats**: `xlsx`, `xls`
- **Deps**: `openpyxl>=3.1` (new optional extra `[excel]`)
- **Approach**: iterate sheets, each sheet becomes a `PageText` (page_number = sheet index). Rows tab-separated, same as DOCX table pattern.
- **Metadata**: sheet names, row counts per sheet
- **Effort**: small

---

## v0.6.0 — Domain Presets

### Medical Preset
- **File**: `presets/medical.py`
- **Classification**: document_type (discharge_summary, lab_report, radiology, pathology, operative_note, progress_note, prescription, referral), patient_id (redacted), facility, provider, date_of_service, specialty
- **Enrichment**: medical_concepts (ICD-10 relevant), body_systems, medications_mentioned, procedures, relevance (critical/supporting/administrative)
- **Parser**: populates `metadata.extra` with facility, provider, specialty, date_of_service
- **Validation needed**: sample documents from each document type to tune prompts
- **Effort**: moderate (prompt engineering + validation)

### Financial Preset
- **File**: `presets/financial.py`
- **Classification**: document_type (10-K, 10-Q, 8-K, proxy, earnings_call, annual_report, balance_sheet, invoice, contract), company, ticker, fiscal_period, filing_date
- **Enrichment**: financial_concepts (revenue, EBITDA, margins, guidance), entities_mentioned, time_periods, monetary_values, relevance (material/supporting/boilerplate)
- **Parser**: populates `metadata.extra` with company, ticker, fiscal_period, filing_date
- **Validation needed**: sample SEC filings + earnings transcripts to tune prompts
- **Effort**: moderate (prompt engineering + validation)

---

## v0.7.0 — Search at Scale

### Phase 1: numpy batch scoring (quick win)
- **Where**: `storage/database.py`
- **What**: cache embeddings as a numpy matrix on first search, compute similarity via single matmul instead of per-row Python loop
- **Cache invalidation**: rebuild matrix after `save()` or `delete_document()`
- **Scale**: good to ~500k chunks
- **Deps**: numpy (already a transitive dep via openai)
- **Effort**: moderate

### Phase 2: sqlite-vec extension
- **Where**: `storage/database.py` or new `storage/vec.py`
- **What**: use [sqlite-vec](https://github.com/asg017/sqlite-vec) for native vector indexing in SQLite
- **Schema**: `CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[dim])`
- **Deps**: `sqlite-vec>=0.1` (new optional extra `[vec]`)
- **Scale**: good to millions of chunks
- **Migration**: needs to populate virtual table from existing `embedding_json` column
- **Effort**: moderate

### Phase 3: pluggable vector backend (future)
- **Where**: new `storage/vector.py` with `VectorStore` protocol
- **What**: abstract interface for vector storage/search. Built-in SQLite backend, optional ChromaDB/Pinecone/Weaviate backends.
- **Effort**: significant — only if demand warrants it

---

## Backlog

- **Wire Vector Lex to use distillcore** — replace Vector Lex's internal pipeline with `pip install distillcore[pdf]`
- **Streaming progress via SSE** — real-time pipeline progress for web UIs
- **DOCX section detection** — use heading styles in python-docx to auto-detect document structure without LLM
- **Markdown heading-aware chunking** — split on `#` / `##` headers before falling back to paragraph splitting
- **Token counting** — replace `len(text) // 4` estimate with actual tiktoken counts
- **Retry with backoff** — configurable retry policy for LLM calls (currently: single retry, no backoff)
