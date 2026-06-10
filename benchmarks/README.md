# Benchmarks

Reproducibility: `uv run python -m benchmarks.run`. 
Add `--with-llm` to include B3/B4 (requires `OPENAI_API_KEY`).

**Run:** 2026-06-10T15:31:07+00:00  
**Env:** distillcore 0.8.0, Python 3.11.13, Darwin arm64

## B1 — Chunking throughput

Synthetic paragraph-structured documents. distillcore vs LangChain `RecursiveCharacterTextSplitter` (`paragraph`, `sentence`) and `CharacterTextSplitter` (`fixed`).
`target_tokens=500` rows shown; full results in `results.json`.
Numbers are mean of 5 measured runs after 1 warmup.

| Doc | Strategy | distillcore chunks/s | LangChain chunks/s | distillcore chars/s | LangChain chars/s |
|---|---|---:|---:|---:|---:|
| 100K | paragraph | 551,402 | 662,921 | 941,590,408 | 1,126,368,840 |
| 100K | sentence | 152,941 | 363,057 | 295,132,655 | 641,009,243 |
| 100K | fixed | 2,700,000 | 16,939 | 10,205,011,067 | 30,355,279 |
| 500K | paragraph | 538,182 | 641,469 | 909,132,363 | 1,080,704,365 |
| 500K | sentence | 147,429 | 370,130 | 285,933,700 | 649,343,592 |
| 500K | fixed | 2,869,565 | 16,186 | 10,831,039,163 | 29,130,232 |
| 1000K | paragraph | 521,548 | 652,031 | 879,671,546 | 1,098,334,048 |
| 1000K | sentence | 145,352 | 352,905 | 281,902,894 | 618,499,653 |
| 1000K | fixed | 2,933,333 | 16,127 | 11,115,938,705 | 29,022,931 |

## B2 — PDF extraction throughput

pdfplumber extraction. Synthetic PDFs are reportlab-generated, 70-char-wide text-only pages.
Real-world docs (Federal Register notice) extract slower due to richer page layouts.

| Document | Pages | Elapsed | Pages/s | Chars/s |
|---|---:|---:|---:|---:|
| synthetic_10p | 10 | 0.28s | 35.5 | 108,321 |
| synthetic_50p | 50 | 1.44s | 34.6 | 105,725 |
| synthetic_100p | 100 | 2.99s | 33.4 | 102,013 |
| real_federal_register | 86 | 6.48s | 13.3 | 98,416 |

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
| 5,000 | 25.23 | 323.60 | 12.8x |
| 50,000 | 210.08 | 3342.72 | 15.9x |
