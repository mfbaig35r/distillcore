# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-06-10

### Added
- `compute_coverage_sequential()` — order-preserving word coverage metric (greedy O(n+m) forward walk). Complements the bag-of-words `compute_coverage()` by catching reordered chunks, duplicated content with tail dropped, and repetitive/tabular input where the same vocabulary repeats. Exported from the package root.
- `structuring_coverage_sequential`, `chunking_coverage_sequential`, and `end_to_end_coverage_sequential` fields on `ValidationReport`, populated automatically by all three validate functions. The bag-of-words metric remains the pass/fail gate to preserve existing user threshold behavior; the sequential metric is a secondary signal.
- Contributing guide (`CONTRIBUTING.md`), issue templates, and PR template.
- `benchmarks/` suite — chunking throughput (B1, with LangChain head-to-head), PDF extraction throughput (B2), coverage accuracy (B3, `--with-llm`), and end-to-end pipeline timing (B4, `--with-llm`). Synthetic fixtures, no external corpus required. Results pinned in `benchmarks/results.json` and rendered into `benchmarks/README.md`. New `benchmarks` optional dependency (`reportlab`, `langchain-text-splitters`).
- **CSV / TSV extractor** (`extractors/csv.py`) — stdlib `csv` only, no new optional extras. Registered for `.csv` and `.tsv`. Delimiter auto-detected via `csv.Sniffer` (supports `,`, `\t`, `|`, `;`); falls back to extension on single-column files. Output is tab-normalized regardless of source delimiter; `ExtractionResult.metadata` carries `columns`, `row_count`, and `delimiter`. Empty files extract cleanly with `page_count=0`.
- **Excel (`.xlsx`) extractor** (`extractors/excel.py`) via new `[excel]` optional extra (`openpyxl>=3.1`). Each non-empty worksheet becomes one `PageText` with `page_number = 1-based sheet index`; cells are tab-separated within rows, rows newline-separated within a sheet. Cell values coerce to strings (datetimes via ISO format, bools as `TRUE`/`FALSE`). Empty rows skipped; empty sheets dropped entirely (counts surfaced in metadata). `ExtractionResult.metadata` carries `sheet_names` and `row_counts`. Legacy `.xls` is not supported — use `.xlsx`.
- **Search at Scale Phase 1.** `Store.search()` now caches embeddings as an L2-normalized float32 numpy matrix on first call, turning cosine similarity into a single `matrix @ query` matmul with top-K via `argpartition`. Cache invalidates atomically on `save()` and `delete_document()` via a `_matrix_version` counter, with rebuild happening inside the same lock that owns the search SQL fetch. Benchmark B5: **~16x speedup** at 5K chunks (20ms vs 326ms per query), ~15x at 50K (216ms vs 3.4s). New `[search-scale]` optional extra adds `numpy>=1.26`; without it, `search()` falls back to the original per-row Python loop unchanged. See `benchmarks/README.md`.

### Fixed
- `distillcore.__version__` was stuck at `"0.7.0"` after the 0.7.1 bump; now matches `pyproject.toml`.
- **Paragraph chunking is ~7x faster.** `split_paragraphs()` was using `re.split(r"\n{2,}", ...)` when `text.split("\n\n")` produces the same result downstream (the existing `strip()` + empty-skip handles leftover newlines from `\n\n\n+` runs). 500K-char paragraph chunking dropped from 3.8ms → 0.52ms per call. distillcore is now within 12-18% of LangChain's `RecursiveCharacterTextSplitter` (was 4-5x slower). See `benchmarks/README.md`.
- `local_embedder` and `cohere_embedder` now resolve from the top-level `distillcore` package via lazy `__getattr__`. When the relevant extra is not installed, accessing the symbol raises a friendly `ImportError` with the install command instead of a bare `ImportError: cannot import name …`.
- `DistillConfig.allowed_dirs` is now `list[str | Path] | None` (was `list[str] | None`). Accepts `Path` objects without forcing callers to `str()`-stringify them. Backward-compatible: `list[str]` still works.
- `DomainConfig.parse_classification` is now typed `Callable[..., DocumentMetadata] | None` (was `... Any`). Every caller already expects `DocumentMetadata`; the annotation now reflects that.
- `_impl_distill_*` and other MCP impl functions in `server.py` are now annotated `-> dict[str, Any]` rather than bare `-> dict`. Type-only fix.
- `Store.search()` docstring now documents the ~50K-chunk scaling boundary and references the Search-at-Scale roadmap items.

## [0.7.1] - 2026-04-28

### Fixed
- **MCP response bloat** — embedding arrays are now stripped from `distill_file`, `distill_text`, and `distill_batch` responses, cutting payload size ~80% for embedded documents. A new `has_embedding` boolean is exposed on each chunk so callers can still tell whether embeddings were generated.
- **Silent structuring failures** — when structuring throws or no prompt is configured, the reason is now propagated through `ValidationReport.warnings` as `"Structuring failed: …"` instead of returning empty sections with no explanation.

### Changed
- `parse_structure_result()` now returns a 3-tuple `(sections, transcript_turns, error)`. Internal API, but callers outside the pipeline will need to unpack the new field.
- Added `has_embedding` computed field to `DocumentChunk`, consistent across the pipeline and DB retrieval paths.

## [0.7.0] - 2026-04-27

### Changed
- **BREAKING:** `openai` is now an optional dependency. Install with `pip install distillcore[openai]` for LLM features (classification, structuring, enrichment, OpenAI embeddings). Core features (chunking, extraction, validation, storage) work without it.
- **BREAKING:** Removed unused `store_path` field from `DistillConfig`. The server uses `DISTILLCORE_STORE` env var; library users construct `Store(path)` directly.
- Extracted shared sync/async pipeline helpers into `pipeline/_shared.py`, eliminating ~200 lines of duplicated code across classification, enrichment, structuring, and orchestrator pairs.

### Added
- 18 new unit tests for extracted shared helpers (265 total).

## [0.6.1] - 2026-04-27

### Fixed
- **Runtime crash in `distill_batch`** — `store` parameter shadowed the module-level `Store` instance, causing `AttributeError` when persisting batch results.
- **Event loop blocking in async pipeline** — `extract()` now runs in `asyncio.to_thread()` so PDF extraction doesn't block concurrent batch processing.
- **Prompt-injection defense in enrichment** — oversized prompts now drop chunks cleanly instead of slicing mid-JSON, preserving sentinel markers and the "ignore instructions" directive.
- **Bare `dict` types on Pydantic models** — `ExtractionResult.metadata` and `DocumentMetadata.extra` are now `dict[str, Any]`.

### Changed
- `openai` SDK is now lazy-imported, eliminating ~150ms cold-import cost for chunk-only users.
- `compute_coverage()` docstring now documents the bag-of-words matching limitation.

## [0.6.0] - 2026-04-27

### Added
- **Standalone chunking API** — `chunk()` and `achunk()` with 4 strategies:
  - `"paragraph"` — split on paragraph boundaries with cascading subsplit for oversized blocks.
  - `"sentence"` — split on sentence boundaries, greedily fill to target size.
  - `"fixed"` — pure sliding window at word boundaries with overlap.
  - `"llm"` — LLM-driven semantic chunking via GPT-4o (requires API key).
- `estimate_tokens()` function for token count estimation.
- `min_tokens` parameter on `chunk()` and `ChunkConfig` — merge small chunks into neighbors.
- `tokenizer` parameter for custom token counting functions.
- `strategy` field on `ChunkConfig` — named strategies (`"paragraph"`, `"sentence"`, `"fixed"`, `"llm"`) in addition to `"auto"`.
- LLM chunking handles large documents via 300-sentence overlapping windows.

## [0.5.0] - 2026-04-27

### Changed
- **Cascading subsplit** for oversized paragraphs (PDF pages with single-newline breaks): line breaks → sentence boundaries → hard cut at word boundary. No chunk exceeds `max_chars`.
- **Boundary-based structuring** — LLM returns `page_range` boundaries instead of content text. `_populate_section_content()` fills content by slicing original page text, reducing token usage and improving accuracy.

## [0.4.0] - 2026-04-26

### Added
- Async pipeline: `process_document_async()`, `process_text_async()`.
- Batch processing: `process_batch()`, `process_batch_sync()` with concurrency control.
- Progress callbacks via `DistillConfig.on_progress`.
- `config.validate()` for early API key warnings.

## [0.3.0] - 2026-04-25

### Added
- Embedding provider factories: `openai_embedder`, `ollama_embedder`, `local_embedder`, `cohere_embedder`.
- Tenant isolation on `Store` via `tenant_id` parameter.
- Path traversal protection via `allowed_dirs` config.
- LLM prompt hardening with untrusted content sentinels.

## [0.2.0] - 2026-04-24

### Added
- DOCX extractor (`python-docx`).
- HTML extractor (`beautifulsoup4`).
- CI/CD: tests on push, publish to PyPI on release.

## [0.1.0] - 2026-04-23

### Added
- Initial release.
- 7-stage pipeline: extract, classify, structure, chunk, enrich, embed, validate.
- PDF extraction via `pdfplumber` with vision OCR fallback.
- Text and Markdown extraction.
- SQLite storage with cosine similarity search.
- Generic and Legal domain presets.
- FastMCP server with 8 tools.
