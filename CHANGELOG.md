# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
