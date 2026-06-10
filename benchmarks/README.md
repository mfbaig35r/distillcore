# Benchmarks

Reproducibility: `uv run python -m benchmarks.run`. 
Add `--with-llm` to include B3/B4 (requires `OPENAI_API_KEY`).

**Run:** 2026-06-10T05:39:42+00:00  
**Env:** distillcore 0.7.1, Python 3.11.13, Darwin arm64

## B1 — Chunking throughput

Synthetic paragraph-structured documents. distillcore vs LangChain `RecursiveCharacterTextSplitter` (`paragraph`, `sentence`) and `CharacterTextSplitter` (`fixed`).
`target_tokens=500` rows shown; full results in `results.json`.
Numbers are mean of 5 measured runs after 1 warmup.

| Doc | Strategy | distillcore chunks/s | LangChain chunks/s | distillcore chars/s | LangChain chars/s |
|---|---|---:|---:|---:|---:|
| 100K | paragraph | 513,043 | 572,816 | 869,082,437 | 975,138,973 |
| 100K | sentence | 67,974 | 345,455 | 131,132,177 | 606,680,209 |
| 100K | fixed | 2,454,545 | 16,602 | 9,248,846,966 | 29,752,948 |
| 500K | paragraph | 559,546 | 642,857 | 945,647,956 | 1,082,868,011 |
| 500K | sentence | 68,910 | 354,919 | 133,636,569 | 623,003,547 |
| 500K | fixed | 2,808,511 | 15,851 | 10,618,402,327 | 28,526,474 |
| 1000K | paragraph | 558,380 | 614,271 | 941,892,033 | 1,034,935,816 |
| 1000K | sentence | 66,307 | 347,535 | 128,583,004 | 608,954,614 |
| 1000K | fixed | 2,693,878 | 15,944 | 10,248,736,845 | 28,695,156 |

## B2 — PDF extraction throughput

pdfplumber extraction. Synthetic PDFs are reportlab-generated, 70-char-wide text-only pages.
Real-world docs (Federal Register notice) extract slower due to richer page layouts.

| Document | Pages | Elapsed | Pages/s | Chars/s |
|---|---:|---:|---:|---:|
| synthetic_10p | 10 | 0.29s | 34.8 | 106,236 |
| synthetic_50p | 50 | 1.49s | 33.6 | 102,680 |
| synthetic_100p | 100 | 3.07s | 32.5 | 99,305 |
| real_federal_register | 86 | 6.61s | 13.0 | 96,459 |

## B3 — Coverage accuracy

End-to-end pipeline coverage on the 86-page Federal Register PDF. 
Bag-of-words (gate) vs sequential (secondary signal).

_skipped: requires --with-llm (uses OpenAI API)_

## B4 — End-to-end pipeline

Wall time per document with full pipeline (classify + structure + chunk + enrich + embed). Network-bound stages dominate.

_skipped: requires --with-llm (uses OpenAI API)_

## B5 — Search throughput (numpy cache vs Python fallback)

Cosine-similarity search over a synthetic Store. `dim=384`, 10 results, 5 runs after warmup. Random vectors, no real LLM.
Numpy path uses a single matmul against a cached L2-normalized float32 matrix; Python fallback iterates row-by-row.

| Chunks | numpy ms / query | Python ms / query | speedup |
|---:|---:|---:|---:|
| 5,000 | 20.21 | 329.98 | 16.3x |
| 50,000 | 221.97 | 3417.10 | 15.4x |
