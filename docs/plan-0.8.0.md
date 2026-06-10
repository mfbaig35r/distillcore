# Plan: distillcore 0.8.0

**Theme:** Polish + capability bundle — perf investigation, format coverage (CSV/Excel), and Search-at-Scale Phase 1.

**Target window:** ~5-6 working days.

**Status:** Day 1 complete (Branch (a)). Days 2-6 pending.

## Day 1 outcome — Branch (a) confirmed

Profiled distillcore paragraph chunking on a 500K synthetic doc (`benchmarks/_fixtures.make_document`):

- **Hotspot:** `re.split(r"\n{2,}", text)` in `split_paragraphs()` — 81% of total chunking time.
- **Fix:** replace with `text.split("\n\n")`. Leftover `"\n"` or empty strings from `"\n\n\n+"` runs are already handled by the downstream `strip()` + empty-skip filter, so the regex was doing unnecessary work.
- **Result:** 500K paragraph chunking dropped from 3.8ms → 0.52ms per call (7x speedup). distillcore is now within 12-18% of LangChain's `RecursiveCharacterTextSplitter` across all doc sizes (was 4-5x slower).
- **Sentence strategy gap remains.** 70% of sentence-strategy time is in the lookaround regex `(?<=[.!?])\s+(?=[A-Z])`. Fix would require a hand-rolled sentence scanner (~20-30 lines, Branch (b) territory). Deferred to 0.8.1 — `paragraph` is the default strategy so paragraph is the headline number.

**Next:** Day 3 (CSV extractor). Day 2 quick-wins bundle is no longer required by Branch (c), but the items are still cheap and could be done opportunistically.

---

---

## Why a bundle (not a single-theme release)

The honest framing: 0.8.0 is backlog cleanup, not a marketable single-theme release. We carry three classes of debt worth closing together — measured perf gap, unmet ROADMAP commitments, and a real scaling cliff in `storage/database.py`. Each is small enough that splitting them across three patch releases is more ceremony than value.

If a marketable theme matters more than throughput, see "Alternative scoping" at the bottom.

---

## Day 1 — Chunking perf investigation (BLOCKING decision point)

**Why first:** the benchmark suite from Gap 5 made the gap public. distillcore is ~4-5x slower than LangChain's `RecursiveCharacterTextSplitter` on paragraph/sentence chunking. The root cause is unknown — the fix could be 10 lines or could be architectural.

**Time-box:** 4 hours of profiling, then decide and move on. No open-ended digs.

**What to profile:**
- 100K and 500K synthetic paragraph docs (from `benchmarks/_fixtures.py`)
- `cProfile` + `snakeviz` or `pyinstrument`
- Hotspot candidates worth checking first:
  - `_tokens_to_chars()` / `estimate_tokens()` per-chunk overhead
  - Cascading subsplit (line-break → sentence → hard cut) on already-fitting paragraphs
  - Sentence-boundary regex compilation vs cached pattern
  - Overlap handling string-copy cost
  - `min_tokens` merge pass when disabled (`min_tokens=0`)

**Branches after profiling:**
- **(a) 10-line fix found.** Land the fix Day 1 afternoon, verify via `uv run python -m benchmarks.run`, proceed to Day 3 (CSV).
- **(b) Real but bounded rewrite (1-2 days).** Use Day 2 for the rewrite. Push CSV/Excel into Day 4-5; defer Search-at-Scale Phase 1 to 0.8.1.
- **(c) Architectural gap.** Document honestly in `benchmarks/README.md` with a "Performance gap vs LangChain" section explaining why. Move quick-wins bundle into Day 2.

**Decision gets written into `docs/plan-0.8.0.md` (this doc) before continuing.**

---

## Day 2 — Perf fix OR quick-wins bundle

### Branch (a)/(b): land the perf fix
- Apply fix + targeted tests for whatever invariant the fix touches
- Re-run benchmarks, commit fresh `results.json` and `benchmarks/README.md`
- One commit with clear "fix: X" message

### Branch (c): quick-wins bundle
Six small items, all from `docs/code-review-2026-04-27.md`. One commit each, or one bundled commit:

1. **`local_embedder` + `cohere_embedder` re-exports** from `src/distillcore/__init__.py`. Either conditional like `embedding/__init__.py`, or lazy-raise with friendly install-hint. Smallest pure win.
2. **`config.py:84`** — `allowed_dirs: list[str] | None` → `list[Path] | None`. Saves repeated `Path()` conversion at consumers.
3. **`config.py:45`** — `parse_classification` callable return type: `Any` → `DocumentMetadata`.
4. **`server.py`** — `_impl_distill_*` returns annotated as `dict[str, Any]` (not bare `dict`).
5. **`src/distillcore/__main__.py`** — add `-> None` annotations so `mypy --disallow_untyped_defs` is clean.
6. **`storage/database.py` `search()`** — add a scaling-boundary docstring noting "~50K chunks" Python-loop limit; link forward to Search-at-Scale work.

---

## Day 3 — CSV extractor

**Files:**
- `src/distillcore/extractors/csv.py` — new
- `tests/test_extractors.py` — new test class
- `src/distillcore/extractors/__init__.py` — register

**Approach:**
- stdlib `csv` only — no new optional extras needed
- `csv.Sniffer` to detect delimiter on first 4KB
- Each row joined with `\n`, cells tab-separated. Header row preserved as first line.
- One `PageText` (page_number=1) since CSVs have no native pagination

**Metadata to populate:**
- `metadata.extra["columns"]` — list of header column names
- `metadata.extra["row_count"]`
- `metadata.extra["delimiter"]` — detected by Sniffer

**Test coverage:**
- Standard comma CSV
- Tab-separated (`.tsv`)
- Pipe-separated
- Quoted cells with embedded commas/newlines
- Empty file → empty ExtractionResult

---

## Day 4 — Excel extractor

**Files:**
- `src/distillcore/extractors/excel.py` — new
- `tests/test_extractors.py` — new test class
- `pyproject.toml` — add `excel = ["openpyxl>=3.1"]` extra
- `src/distillcore/extractors/__init__.py` — register

**Approach:**
- `openpyxl` only (no `xlrd` for legacy `.xls` — defer)
- Each worksheet → one `PageText` (page_number = sheet index, 1-based)
- Cells tab-separated within rows, rows newline-separated within a sheet
- Header row (row 1) preserved
- Empty sheets skipped

**Metadata:**
- `metadata.extra["sheet_names"]` — list of all sheet titles
- `metadata.extra["row_counts"]` — dict of sheet name → row count

**Type inference:** out of scope for v1 — cells coerced to `str(cell.value)`. Document in docstring; ASI's approach to dates/numerics can come later if needed.

**Test coverage:**
- Single-sheet workbook
- Multi-sheet workbook (verify page_number assignment)
- Mixed cell types (int, float, datetime, str)
- Empty rows interspersed
- Workbook with empty sheet

---

## Day 5 — Search at Scale Phase 1 (numpy batch matmul)

**Where:** `src/distillcore/storage/database.py:search()`

**Current:** per-row Python loop deserializing JSON embeddings, computing cosine similarity in pure Python.

**Target:** cache embeddings as a numpy `float32` matrix on first search call. Use a single `matrix @ query` matmul. Top-K via `np.argpartition`.

**Cache invalidation:**
- Invalidate on `save()`, `delete_document()`, and any path that mutates the embedding table
- Use a `_matrix_version: int` counter incremented on writes; `_cached_matrix: np.ndarray | None` checked against it
- Eager rebuild on the next `search()` call (lazy, not on every write)

**Numpy dep handling (per honest flag #2):**
- Make `search()` try `import numpy` lazily. If numpy is present, use the matrix path. If not, fall back to today's Python loop. Log once at info level if falling back.
- This keeps the dependency surface honest — users on `[openai]` already have numpy transitively; users on `chunking-only` paths don't pay
- Document in the search() docstring that "performance scales to ~500K chunks when numpy is installed"

**Benchmark addition:**
- Add `benchmarks/_search.py` as B5: build a Store with 5K and 50K chunks, embed with a tiny model (or random vectors of the right shape), measure search latency before vs after
- Wire into `benchmarks/run.py`

**Tests:**
- Existing `test_storage.py` should still pass unchanged
- Add a `test_search_matrix_cache_invalidation` test: save → search → save again → search returns updated results

---

## Day 6 — Release

**Pre-release:**
- Re-run full benchmark suite (`uv run python -m benchmarks.run`), commit fresh `results.json` + `README.md`
- Run `uv run python -m pytest tests/ -v --tb=short`
- Run `uv run ruff check src/ tests/ benchmarks/`

**Update `docs/ROADMAP.md`:**
- Acknowledge the v0.5/v0.6/v0.7 label drift. The roadmap headings are aspirational, not historical
- Move CSV/Excel from "v0.5.0" header into "Shipped in 0.8.0"
- Move Search-at-Scale Phase 1 from "v0.7.0" into "Shipped in 0.8.0"
- Push remaining items (Medical/Financial presets, Phase 2 sqlite-vec, Phase 3 pluggable backend) to "Future"

**Release ceremony:**
- Move `## [Unreleased]` content in `CHANGELOG.md` to `## [0.8.0] - YYYY-MM-DD`
- Bump `pyproject.toml` version to `0.8.0`
- Bump `src/distillcore/__init__.py:__version__` to `0.8.0`
- Commit: `chore: bump version to 0.8.0`
- Tag: `git tag v0.8.0`
- Push tag — PyPI auto-publishes via trusted publisher

---

## Three honest flags (kept here for the record)

1. **Day 1 fork is unavoidable.** Can't promise a week-shape today; the perf result determines whether 0.8.0 ships extractors + scaling (branches a/c) or perf-fix + extractors only (branch b). Pre-committed time-box: 4 hours profiling, then move.

2. **Numpy gets lazy-imported, not added as direct dep.** Keeps the dependency story honest for users who only use chunking. The cost is one `try: import numpy` per search call (~negligible) and a "fell back to Python loop" log line we hope to never see.

3. **Bundle vs single-theme.** This plan trades marketability for debt cleanup. The alternative scopings below would ship a cleaner narrative.

---

## Alternative scoping (if you want to revisit)

If "polish + capability" feels muddled and a single theme is worth more than throughput:

**Option B — "0.8.0: tabular data"**
- CSV + Excel extractors + Medical + Financial presets (the four format/domain items)
- Defers perf investigation and search-at-scale to 0.8.1
- Marketing line: "distillcore now handles real-world tabular and domain-specific documents"

**Option C — "0.8.0: scale"**
- Perf investigation + Phase 1 (numpy) + Phase 2 (sqlite-vec)
- Defers extractors and presets to 0.8.1
- Marketing line: "distillcore scales to millions of chunks with native vector search"

Both alternatives are ~1 week each, but each ships less surface area and one clearer message.

---

## What this plan does NOT include (and why)

- **Medical/Financial presets** — need real document corpora to validate prompts; without that, presets are guesses. Defer until either someone supplies sample docs or we want to use distillcore on something specific.
- **DOCX heading-aware section detection / MD heading-aware chunking** — both useful, both fit in 0.8.1
- **tiktoken token counting** — the `len(text)//4` estimate is good enough until users complain about wasted token budget
- **Retry with backoff** — useful but no acute pain reports
- **Vector Lex dogfooding** — separate project, separate planning
- **distillcore-agents PyPI release** — separate project; trusted publisher setup is a 1-2 hr task that can happen anytime
- **distillcore-docs Next.js site** — 1-2 week project on its own; not part of this lib release

These all stay in `docs/ROADMAP.md` Backlog section.
