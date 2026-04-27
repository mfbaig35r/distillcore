# distillcore Changelog: v0.4.0 → v0.7.0

Use this document to update the companion project `distillcore-agents` to be compatible with distillcore v0.7.0.

---

## v0.5.0 — Cascading subsplit + boundary-based structuring

### Chunking overhaul
- **Cascading subsplit** for oversized paragraphs (e.g., PDF pages with only single-newline breaks):
  - Level 1: split on line breaks (`\n`)
  - Level 2: split on sentence boundaries (`.!?` + whitespace + capital)
  - Level 3: hard cut at word boundary
- No chunk exceeds `max_chars` (configurable, defaults to `target_chars * 2`)
- New internal helpers: `_subsplit()`, `_greedy_fill()`, `_hard_split()`

### Structuring improvements
- **Boundary-based structuring**: LLM returns `page_range: [start, end]` instead of content text
- `_populate_section_content()` fills content by slicing original `pages_text` — avoids LLM reproducing content verbatim
- Better accuracy and lower token usage for large documents

### No breaking changes from v0.4.0

---

## v0.6.0 — Standalone chunking API

### New public API: `chunk()` and `achunk()`

Four strategies available as a standalone API (no pipeline, no LLM required for first three):

```python
from distillcore import chunk, achunk, estimate_tokens

# Paragraph (default) — splits on \n\n, subsplits oversized blocks, supports overlap
chunks = chunk(text, strategy="paragraph", target_tokens=500, overlap_tokens=50)

# Sentence — splits on sentence boundaries, greedily fills to target size
chunks = chunk(text, strategy="sentence", target_tokens=500)

# Fixed — pure sliding window at word boundaries with overlap
chunks = chunk(text, strategy="fixed", target_tokens=500, overlap_tokens=50)

# LLM — sends numbered sentences to GPT-4o for semantic grouping (requires API key)
chunks = chunk(text, strategy="llm", api_key="sk-...", target_tokens=500)

# Async version (only LLM strategy is truly async)
chunks = await achunk(text, strategy="llm", api_key="sk-...")

# Token estimation
tokens = estimate_tokens("some text")  # len // 4 default, or pass custom tokenizer
```

### New parameters
- `chunk(strategy=)` — `"paragraph"`, `"sentence"`, `"fixed"`, `"llm"`
- `chunk(min_tokens=)` — merge chunks below this threshold into neighbors (0 = disabled)
- `chunk(tokenizer=)` — custom `Callable[[str], int]` for token counting
- `chunk(api_key=, model=)` — for LLM strategy only

### Pipeline integration
- `ChunkConfig(strategy=)` now accepts named strategies in addition to `"auto"`
- `"auto"` (default) still uses the pipeline's section/transcript/fallback logic
- Named strategies delegate to the public `chunk()` API

### LLM chunking details
- Large documents (>300 sentences) use overlapping 300-sentence windows with 30-sentence overlap
- `_resolve_overlaps()` deduplicates groups from windowed processing
- Falls back to paragraph strategy on any LLM error

### New exports from `distillcore`
- `chunk`, `achunk`, `estimate_tokens`

---

## v0.6.1 — Bug fixes from code review

### Bug fixes
1. **`distill_batch` runtime crash** — `store: bool` parameter shadowed the module-level `Store` instance, causing `AttributeError` when persisting batch results. Fixed by extracting `_impl_distill_batch(persist=)`.
2. **Async pipeline blocked event loop** — `extract()` (sync file I/O via pdfplumber) was called directly in async context. Now wrapped in `asyncio.to_thread()`.
3. **Enrichment truncation broke prompt-injection defense** — oversized prompts were sliced mid-JSON, chopping off `--- END UNTRUSTED CHUNK DATA ---` sentinel and the "ignore instructions" directive. Now drops chunks from the list to fit, preserving valid JSON and sentinels.
4. **Bare `dict` types on Pydantic models** — `ExtractionResult.metadata` and `DocumentMetadata.extra` changed from `dict` to `dict[str, Any]`.
5. **Eager `openai` import** — moved behind `TYPE_CHECKING` guard with lazy runtime import inside `get_client()`/`get_async_client()`. Eliminates ~150ms import cost for chunk-only users.
6. **Undocumented coverage limitation** — `compute_coverage()` docstring now notes the bag-of-words matching limitation.

### No breaking changes from v0.6.0

---

## v0.7.0 — Sync/async dedup, optional openai, config cleanup

### Breaking changes

#### 1. `openai` is now an optional dependency

**Before:** `pip install distillcore` installed openai automatically.
**After:** openai must be explicitly requested.

```bash
pip install distillcore[openai]    # for LLM features
pip install distillcore[all]       # for everything
pip install distillcore            # just pydantic — chunking, extraction, validation, storage work
```

If code calls `get_client()` or `get_async_client()` without openai installed, it raises:
```
ImportError: LLM features require the openai package. Install with: pip install distillcore[openai]
```

**What still works without openai:**
- `chunk()`, `achunk()`, `estimate_tokens()`
- All extractors (text, PDF, DOCX, HTML)
- `process_text()` / `process_document()` with `DomainConfig()` (no prompts → skips all LLM)
- `compute_coverage()`, `find_missing_segments()`
- `Store` (SQLite persistence and search)
- `ollama_embedder()`, `local_embedder()`, `cohere_embedder()`

#### 2. `DistillConfig.store_path` removed

This field was defined but never read by anything. The server reads `DISTILLCORE_STORE` env var. Library users construct `Store(path)` directly. If you were passing `store_path=` to `DistillConfig()`, remove it.

### Internal improvements (non-breaking)

#### Sync/async duplication collapsed

Created `pipeline/_shared.py` with pure helper functions extracted from duplicated sync/async pairs:

**Classification helpers:**
- `build_classification_user_msg(filename, pages_text) -> str`
- `sanitize_classification_output(result) -> dict`
- `fallback_metadata(filename, page_count) -> DocumentMetadata`
- `build_default_metadata(result, filename, page_count) -> DocumentMetadata`

**Enrichment helpers:**
- `build_chunk_summaries(chunks) -> list[dict]`
- `render_enrichment_msg(summaries, document_type, total_chunks) -> str`
- `truncate_enrichment_msg(chunk_summaries, document_type, total_chunks) -> str`
- `apply_enrichments(chunks, result) -> int`

**Structuring helpers (moved from `structuring.py`, re-exported for backward compatibility):**
- `parse_structure_result(result, pages_text) -> tuple[list[Section], list[TranscriptTurn]]`
- `_populate_section_content(section, pages_text) -> None`
- `_parse_section(data) -> Section`

**Orchestrator helpers:**
- `make_emitter(config) -> Callable`
- `build_combined_validation(struct_report, chunk_report, e2e_report) -> ValidationReport`

**Impact:** ~200 lines eliminated. A fix in any shared helper now applies to both sync and async paths automatically.

---

## Current public API (v0.7.0)

### Top-level exports (`from distillcore import ...`)

```python
# Standalone chunking
chunk(text, *, strategy, target_tokens, max_tokens, overlap_tokens, min_tokens, tokenizer, api_key, model) -> list[str]
achunk(...)  # async version
estimate_tokens(text, tokenizer=None) -> int

# Pipeline (sync)
process_document(source, *, config, format, embed) -> ProcessingResult
process_text(text, *, config, filename, embed) -> ProcessingResult

# Pipeline (async + batch)
process_document_async(source, *, config, format, embed) -> ProcessingResult
process_text_async(text, *, config, filename, embed) -> ProcessingResult
process_batch(sources, *, config, format, embed, max_concurrent, on_result) -> list[ProcessingResult]
process_batch_sync(sources, **kwargs) -> list[ProcessingResult]

# Config
DistillConfig(openai_api_key, openai_model, max_tokens, chunk, embedding, domain, enrich_chunks, enable_ocr, large_doc_char_threshold, llm_page_window_size, llm_page_window_overlap, structuring_coverage_threshold, chunking_coverage_threshold, end_to_end_coverage_threshold, allowed_dirs, on_progress)
ChunkConfig(target_tokens, overlap_chars, max_tokens, min_tokens, tokenizer, strategy)
EmbeddingConfig(model, embed_fn)
DomainConfig(name, classification_prompt, structuring_prompt, transcript_prompt, enrichment_prompt, parse_classification)

# Models (Pydantic v2)
PageText, ExtractionResult, Section, TranscriptTurn, DocumentMetadata, Document, DocumentChunk, ChunkedDocument, ValidationReport, ProcessingResult, BatchResult

# Extractors
extract(source, format=None, config=None) -> ExtractionResult
register_extractor(extractor) -> None

# Validation
compute_coverage(original, derived) -> float
find_missing_segments(original, derived, min_length=50) -> list[str]

# LLM utilities
safe_parse(raw) -> dict
try_fix_truncated_json(raw) -> str

# Presets
load_preset(name) -> DomainConfig  # "generic" or "legal"

# Embedding providers
openai_embedder(model, api_key) -> Callable  # requires distillcore[openai]
ollama_embedder(model, base_url) -> Callable
local_embedder(model, device) -> Callable    # requires distillcore[local]
cohere_embedder(model, api_key) -> Callable  # requires distillcore[cohere]

# Storage
Store(path) -> Store
Store.save(result, tenant_id=None) -> str
Store.get_document(doc_id, tenant_id=None) -> dict | None
Store.list_documents(document_type=None, limit=50, tenant_id=None) -> list[dict]
Store.get_chunks(doc_id, tenant_id=None) -> list[dict]
Store.search(query_embedding, top_k=10, document_type=None, document_id=None, tenant_id=None) -> list[dict]
Store.delete_document(doc_id, tenant_id=None) -> bool
Store.stats() -> dict
Store.log_search(query, result_count, top_chunk_ids) -> None
```

### MCP server tools (8 tools)

```
distill_file(file_path, format?, domain?, embed?, chunk_target_tokens?, enrich?, store?)
distill_text(text, domain?, embed?, chunk_target_tokens?, enrich?, store?)
distill_chunks_only(text, chunk_target_tokens?, overlap_tokens?, min_tokens?, strategy?)
distill_validate(original_text, chunk_texts)
distill_search(query, top_k?, document_type?)
distill_list_documents(document_type?, limit?)
distill_get_document(document_id)
distill_batch(file_paths, domain?, embed?, chunk_target_tokens?, enrich?, store?, max_concurrent?)
```

### Optional dependencies

```toml
[project.optional-dependencies]
openai = ["openai>=1.60.0"]       # LLM features (classification, structuring, enrichment, embeddings)
pdf = ["pdfplumber", "pdf2image", "Pillow"]
docx = ["python-docx"]
html = ["beautifulsoup4", "lxml"]
local = ["sentence-transformers"]  # local_embedder
cohere = ["cohere"]               # cohere_embedder
mcp = ["fastmcp>=2.0.0"]         # MCP server
all = ["distillcore[openai,pdf,docx,html]"]
```

### Key architectural details for agents

- **Pipeline stages:** extract → classify → structure → chunk → enrich → embed → validate
- **Graceful degradation:** Every LLM stage returns fallback values on failure. Pipeline always produces a result.
- **Validation:** Three-layer coverage checks (structuring 95%, chunking 98%, end-to-end 93%)
- **Prompt-injection defense:** All user content wrapped in `--- BEGIN/END UNTRUSTED ---` sentinels
- **Extraction is sync, offloaded to thread in async pipeline** via `asyncio.to_thread()`
- **Batch processing:** `asyncio.Semaphore(max_concurrent)` limits concurrency; failures don't crash the batch
- **Shared helpers in `pipeline/_shared.py`** — pure functions, no async, imported by both sync/async modules
