# Benchmarks

Reproducibility: `uv run python -m benchmarks.run`. 
Add `--with-llm` to include B3/B4 (requires `OPENAI_API_KEY`).

**Run:** 2026-06-10T04:29:52+00:00  
**Env:** distillcore 0.7.1, Python 3.11.13, Darwin arm64

## B1 — Chunking throughput

Synthetic paragraph-structured documents. distillcore vs LangChain `RecursiveCharacterTextSplitter` (`paragraph`, `sentence`) and `CharacterTextSplitter` (`fixed`).
`target_tokens=500` rows shown; full results in `results.json`.
Numbers are mean of 5 measured runs after 1 warmup.

| Doc | Strategy | distillcore chunks/s | LangChain chunks/s | distillcore chars/s | LangChain chars/s |
|---|---|---:|---:|---:|---:|
| 100K | paragraph | 133,484 | 662,921 | 226,933,519 | 1,132,618,369 |
| 100K | sentence | 70,270 | 354,037 | 135,638,645 | 622,261,498 |
| 100K | fixed | 2,250,000 | 16,475 | 8,468,356,503 | 29,521,435 |
| 500K | paragraph | 130,569 | 630,573 | 220,673,221 | 1,061,500,928 |
| 500K | sentence | 68,617 | 350,123 | 133,069,926 | 614,270,741 |
| 500K | fixed | 2,588,235 | 15,695 | 9,869,545,104 | 28,245,719 |
| 1000K | paragraph | 127,062 | 622,642 | 214,397,686 | 1,048,399,825 |
| 1000K | sentence | 69,049 | 336,476 | 133,889,536 | 589,732,670 |
| 1000K | fixed | 2,666,667 | 15,553 | 10,136,278,253 | 27,990,661 |

## B2 — PDF extraction throughput

pdfplumber extraction. Synthetic PDFs are reportlab-generated, 70-char-wide text-only pages.
Real-world docs (Federal Register notice) extract slower due to richer page layouts.

| Document | Pages | Elapsed | Pages/s | Chars/s |
|---|---:|---:|---:|---:|
| synthetic_10p | 10 | 0.30s | 33.3 | 101,627 |
| synthetic_50p | 50 | 1.53s | 32.7 | 99,881 |
| synthetic_100p | 100 | 3.11s | 32.1 | 98,098 |
| real_federal_register | 86 | 6.68s | 12.9 | 95,403 |

## B3 — Coverage accuracy

End-to-end pipeline coverage on the 86-page Federal Register PDF. 
Bag-of-words (gate) vs sequential (secondary signal).

_skipped: requires --with-llm (uses OpenAI API)_

## B4 — End-to-end pipeline

Wall time per document with full pipeline (classify + structure + chunk + enrich + embed). Network-bound stages dominate.

_skipped: requires --with-llm (uses OpenAI API)_
