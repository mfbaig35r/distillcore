# distillcore Roadmap

> **Note on past version labels.** Earlier versions of this document used `v0.5.0` / `v0.6.0` / `v0.7.0` headers as wishlists; the actual releases shipped under those numbers covered different work (see `CHANGELOG.md`). This rewrite groups items by status — shipped, planned, deferred — and stops promising specific version numbers up front.

---

## Shipped in 0.8.0 (2026-06-10)

- **CSV / TSV extractor** (`extractors/csv.py`) — stdlib `csv`, `csv.Sniffer` for delimiter detection.
- **Excel (`.xlsx`) extractor** (`extractors/excel.py`) — `openpyxl` via new `[excel]` extra. Each sheet becomes one `PageText`. Legacy `.xls` not supported.
- **Search at Scale Phase 1** — numpy matrix cache in `Store.search()`. `~16x` speedup at 5K-50K chunks vs the pure-Python loop. New `[search-scale]` extra; falls back gracefully when numpy is missing.
- **Paragraph chunking is ~7x faster** — `re.split` → `str.split` in `split_paragraphs()`. distillcore is now within 12-18% of LangChain on paragraph throughput.
- **Quick-wins bundle** from the April code review — friendly `local_embedder` / `cohere_embedder` re-exports via `__getattr__`, `allowed_dirs: list[str | Path]`, typed `parse_classification`, `-> dict[str, Any]` cleanup in `server.py`, scaling-boundary docstring on `Store.search()`.
- **Sequential coverage metric** — `compute_coverage_sequential()` as a secondary signal alongside the bag-of-words gate.
- **Benchmark suite** (`benchmarks/`) — B1 chunking + B2 PDF extraction + B3 coverage (`--with-llm`) + B4 end-to-end (`--with-llm`) + B5 search. Head-to-head vs LangChain on B1.
- **Contributing guide + issue / PR templates.**

---

## Planned

### Domain presets

**Medical preset** (`presets/medical.py`)
- Classification: document_type (discharge_summary, lab_report, radiology, pathology, operative_note, progress_note, prescription, referral), facility, provider, date_of_service, specialty.
- Enrichment: medical_concepts (ICD-10 relevant), body_systems, medications_mentioned, procedures, relevance (critical / supporting / administrative).
- Parser populates `metadata.extra` with facility, provider, specialty, date_of_service.
- **Gate before shipping:** real sample documents from each document type to tune prompts. Currently waiting on a corpus.

**Financial preset** (`presets/financial.py`)
- Classification: document_type (10-K, 10-Q, 8-K, proxy, earnings_call, annual_report, balance_sheet, invoice, contract), company, ticker, fiscal_period, filing_date.
- Enrichment: financial_concepts (revenue, EBITDA, margins, guidance), entities_mentioned, time_periods, monetary_values, relevance (material / supporting / boilerplate).
- Parser populates `metadata.extra` with company, ticker, fiscal_period, filing_date.
- **Gate before shipping:** sample SEC filings + earnings transcripts.

### Search at Scale Phase 2 — sqlite-vec

- New optional extra `[vec]` adds `sqlite-vec>=0.1`.
- Either extend `storage/database.py` or split into `storage/vec.py`.
- Schema: `CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[dim])`.
- Migration: populate the virtual table from the existing `embedding_json` column.
- **Scale target:** millions of chunks.

### Sentence-strategy chunking rewrite

- Day 1 of the 0.8.0 plan found the `_chunk_sentence` hotspot: 70% of time in the lookaround regex `(?<=[.!?])\s+(?=[A-Z])`. Cannot be replaced with `str.split`.
- Fix is a hand-rolled forward-scanner over the string (~20-30 lines).
- Target similar parity-with-LangChain throughput as paragraph chunking already achieves.

### DOCX heading-aware section detection

- Use `python-docx` heading styles (`Heading 1`, `Heading 2`, ...) to auto-detect document structure without an LLM call.
- Cuts cost for structured DOCX inputs.

### Markdown heading-aware chunking

- Split on `#`, `##`, `###` boundaries before falling back to paragraph splitting.
- Same shape as DOCX heading detection — saves an LLM call when structure is explicit.

### tiktoken token counting

- Replace `len(text) // 4` estimate with actual `tiktoken` counts.
- Optional extra so the dep stays out of the core install. Configurable via existing `tokenizer` parameter.

### Retry with backoff for LLM calls

- Currently single retry, no backoff.
- Configurable retry policy on `DistillConfig`. Respect `Retry-After` headers.

---

## Deferred — only if demand warrants

### Search at Scale Phase 3 — pluggable vector backend

- New `storage/vector.py` with a `VectorStore` protocol.
- Abstract interface for vector storage / search. Built-in SQLite backend, optional ChromaDB / Pinecone / Weaviate backends.
- Significant scope. Only ship if users ask.

### Streaming progress via SSE

- Real-time pipeline progress for web UIs. The existing `on_progress` callback already covers programmatic use; SSE adds an HTTP transport.

### `.xls` (legacy binary Excel)

- Out of scope for `[excel]`. `openpyxl` doesn't support it. Would need `xlrd` and adds enough surface area that it's not worth shipping without a concrete request.

---

## External / ecosystem (not part of this repo)

- **Wire Vector Lex to use distillcore** — replace Vector Lex's internal pipeline with `pip install distillcore[pdf]`. Dogfood opportunity.
- **`distillcore-agents`** PyPI release — pydantic-ai agent layer; needs trusted-publisher setup.
- **`distillcore-docs`** site — Next.js docs + interactive playground (`REQUIREMENTS.md` exists in that repo).
