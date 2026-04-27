# Plan: Remaining Gaps for Production-Grade PyPI Library

Status of each gap after today's work:

| # | Gap | Status |
|---|-----|--------|
| 1 | No CHANGELOG.md | **Done** — `CHANGELOG.md` at repo root, Keep a Changelog format |
| 2 | No py.typed marker | **Already existed** — `src/distillcore/py.typed` present |
| 3 | No contributing guide or issue templates | **Open** — see below |
| 4 | README stale | **Done** — rewritten for v0.7.0 |
| 5 | No benchmarks | **Open** — see below |
| 6 | Coverage metric weakness | **Open** — see below |

---

## Gap 3: Contributing Guide + Issue Templates

### Why
Signals that external contributions are welcome and reduces friction for first-time contributors. Also standardizes bug reports so you get reproducible issues.

### What to create

**`CONTRIBUTING.md`** (repo root):
- Prerequisites: Python 3.11+, uv
- Dev setup: `uv sync --extra dev --extra pdf --extra openai --extra docx --extra html`
- Running tests: `uv run python -m pytest tests/ -v --tb=short`
- Running lint: `uv run ruff check src/ tests/`
- Code style: ruff E/F/I, line length 100, mypy strict
- PR guidelines: one concern per PR, tests required for new behavior, async parity (if adding a sync function, add the async counterpart or justify why not)
- Commit message format: `type: description` (fix, feat, refactor, docs, test)

**`.github/ISSUE_TEMPLATE/bug_report.md`**:
- distillcore version
- Python version
- OS
- Minimal reproduction
- Expected vs actual behavior
- Full traceback

**`.github/ISSUE_TEMPLATE/feature_request.md`**:
- Problem statement (what are you trying to do?)
- Proposed solution
- Alternatives considered
- Which pipeline stage does this touch?

**`.github/PULL_REQUEST_TEMPLATE.md`**:
- What this PR does (one sentence)
- How to test
- Checklist: tests pass, ruff clean, async parity checked, CHANGELOG updated

### Effort
~1 hour. All boilerplate, no code changes.

---

## Gap 5: Benchmarks

### Why
Numbers build trust. A user evaluating distillcore vs. LangChain's text splitters or unstructured.io wants to know: how fast is chunking? How does coverage hold up on real documents? What's the token overhead of LLM structuring?

### What to build

**`benchmarks/` directory** with a runner script and a results table in the README.

**Benchmark 1: Chunking throughput**
- Input: 100K, 500K, 1M char synthetic documents
- Measure: chunks/sec for each strategy (paragraph, sentence, fixed)
- Compare: target_tokens=300 vs 500 vs 1000
- LLM strategy excluded (network-bound, not comparable)

**Benchmark 2: Extraction throughput**
- Input: 10-page PDF, 50-page PDF, 100-page PDF (use Federal Register PDFs from test-files/)
- Measure: pages/sec for pdfplumber extraction
- Include OCR path timing if scanned pages present

**Benchmark 3: Coverage accuracy**
- Input: 10 diverse documents (legal, reports, articles, transcripts)
- Measure: end-to-end coverage score, structuring coverage, chunking coverage
- Report: min, max, mean, std dev across documents
- Compare: bag-of-words metric vs a sequence-aware metric (see Gap 6)

**Benchmark 4: Pipeline end-to-end**
- Input: 10 real documents
- Measure: wall time per document (with and without LLM stages)
- Break down: extraction %, chunking %, LLM %, embedding %

### Output format
- `benchmarks/run.py` — CLI script that runs all benchmarks and writes results to `benchmarks/results.json`
- `benchmarks/README.md` — rendered table with latest numbers
- CI integration (optional): run on release, fail if chunking throughput regresses >20%

### Effort
~3-4 hours for initial benchmarks. Ongoing: update numbers on each release.

---

## Gap 6: Coverage Metric — Sequence-Aware Alternative

### Why
The current `compute_coverage()` uses bag-of-words matching. It counts "what fraction of words from the original appear anywhere in the derived text." This works for natural language but fails silently on:
- Repetitive documents (tabular data, form letters) — same words in different chunks looks like 100% coverage
- Reordered content — all words present but in wrong order
- Duplicated chunks — one chunk repeated, another dropped

### Proposed approach: `compute_coverage_sequential()`

Add a second coverage function that checks order-preserving word matches. Algorithm:

1. Normalize both texts (NFKC, collapse whitespace, lowercase)
2. Split into words
3. Find the longest common subsequence (LCS) of words between original and derived
4. Coverage = len(LCS) / len(original_words)

This catches reordering and duplication while still being tolerant of formatting changes.

### Performance concern
LCS on full documents is O(n*m) which is prohibitive for large texts. Mitigations:
- Use a windowed approach: split original into 500-word windows, compute LCS for each, average
- Or use a greedy sequential match (O(n+m)): walk both word lists forward-only, counting matches. Cheaper than LCS, catches dropped content but not reordering.

### Implementation plan

**Option A (recommended): Greedy sequential match**
```python
def compute_coverage_sequential(original: str, derived: str) -> float:
    orig_words = normalize_text(original).lower().split()
    derived_words = normalize_text(derived).lower().split()
    if not orig_words:
        return 1.0
    j = 0
    matched = 0
    for word in orig_words:
        while j < len(derived_words):
            if derived_words[j] == word:
                matched += 1
                j += 1
                break
            j += 1
        else:
            break  # derived exhausted
    return matched / len(orig_words)
```

This is O(n+m), catches dropped content and content reordering, and handles the repetitive-document case correctly.

**Integration:**
- Add to `validation/coverage.py` alongside existing `compute_coverage()`
- Don't replace the existing function — it's the established default and tests rely on it
- Add a `method` parameter to `validate_end_to_end()` and friends: `method="bag_of_words"` (default) or `method="sequential"`
- Or: add `sequential_coverage` field to `ValidationReport` as an optional second signal
- Export from `__init__.py`

**Tests:**
- Test that identical text returns 1.0
- Test that reordered text returns < 1.0 (bag-of-words would return 1.0)
- Test that dropped content returns < 1.0
- Test that duplicated chunks are detected (one chunk repeated, another dropped)
- Test performance: 1M char document completes in < 1s

### Effort
~2 hours for implementation + tests. The greedy sequential match is ~15 lines of code.

---

## Priority Order

1. **Contributing guide + issue templates** — low effort, immediate trust signal, do first
2. **Sequential coverage metric** — small code change, high correctness value, enables benchmarks
3. **Benchmarks** — depends on having real test documents, most effort, highest long-term value

## Version Strategy

- Contributing guide + templates: no version bump (no code changes)
- Sequential coverage + benchmarks: v0.7.1 (additive, non-breaking)
