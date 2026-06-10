# Benchmarks

Reproducibility: `uv run python -m benchmarks.run`. 
Add `--with-llm` to include B3/B4 (requires `OPENAI_API_KEY`).

**Run:** 2026-06-10T05:03:23+00:00  
**Env:** distillcore 0.7.1, Python 3.11.13, Darwin arm64

## B1 — Chunking throughput

Synthetic paragraph-structured documents. distillcore vs LangChain `RecursiveCharacterTextSplitter` (`paragraph`, `sentence`) and `CharacterTextSplitter` (`fixed`).
`target_tokens=500` rows shown; full results in `results.json`.
Numbers are mean of 5 measured runs after 1 warmup.

| Doc | Strategy | distillcore chunks/s | LangChain chunks/s | distillcore chars/s | LangChain chars/s |
|---|---|---:|---:|---:|---:|
| 100K | paragraph | 556,604 | 678,161 | 948,711,921 | 1,149,265,978 |
| 100K | sentence | 68,966 | 345,455 | 133,093,153 | 607,047,937 |
| 100K | fixed | 2,700,000 | 16,558 | 10,239,790,931 | 29,668,428 |
| 500K | paragraph | 550,186 | 640,086 | 930,305,776 | 1,078,975,433 |
| 500K | sentence | 68,038 | 358,942 | 131,930,065 | 630,415,449 |
| 500K | fixed | 2,808,511 | 16,079 | 10,547,480,920 | 28,936,477 |
| 1000K | paragraph | 556,808 | 635,974 | 939,651,239 | 1,071,842,970 |
| 1000K | sentence | 68,236 | 332,557 | 132,328,277 | 582,638,741 |
| 1000K | fixed | 2,666,667 | 15,912 | 10,121,739,661 | 28,635,654 |

## B2 — PDF extraction throughput

pdfplumber extraction. Synthetic PDFs are reportlab-generated, 70-char-wide text-only pages.
Real-world docs (Federal Register notice) extract slower due to richer page layouts.

| Document | Pages | Elapsed | Pages/s | Chars/s |
|---|---:|---:|---:|---:|
| synthetic_10p | 10 | 0.28s | 35.6 | 108,687 |
| synthetic_50p | 50 | 1.46s | 34.1 | 104,207 |
| synthetic_100p | 100 | 3.05s | 32.8 | 100,104 |
| real_federal_register | 86 | 6.52s | 13.2 | 97,792 |

## B3 — Coverage accuracy

End-to-end pipeline coverage on the 86-page Federal Register PDF. 
Bag-of-words (gate) vs sequential (secondary signal).

_skipped: requires --with-llm (uses OpenAI API)_

## B4 — End-to-end pipeline

Wall time per document with full pipeline (classify + structure + chunk + enrich + embed). Network-bound stages dominate.

_skipped: requires --with-llm (uses OpenAI API)_
