"""Run all benchmarks and write results to benchmarks/results.json.

Usage::

    uv run python -m benchmarks.run                # B1 + B2 only
    uv run python -m benchmarks.run --with-llm     # all four (requires OPENAI_API_KEY)
    uv run python -m benchmarks.run --render-only  # re-render README.md from existing results.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path
from typing import Any

from . import _chunking, _coverage, _end_to_end, _extraction, _search

BENCH_DIR = Path(__file__).parent
RESULTS_PATH = BENCH_DIR / "results.json"
README_PATH = BENCH_DIR / "README.md"


def _env_info() -> dict[str, str]:
    import distillcore

    return {
        "distillcore": getattr(distillcore, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
    }


def _render_readme(results: dict[str, Any]) -> str:
    env = results["env"]
    ts = results["timestamp"]
    lines: list[str] = [
        "# Benchmarks",
        "",
        "Reproducibility: `uv run python -m benchmarks.run`. ",
        "Add `--with-llm` to include B3/B4 (requires `OPENAI_API_KEY`).",
        "",
        f"**Run:** {ts}  ",
        f"**Env:** distillcore {env['distillcore']}, Python {env['python']}, {env['platform']}",
        "",
        "## B1 — Chunking throughput",
        "",
        "Synthetic paragraph-structured documents. distillcore vs LangChain"
        " `RecursiveCharacterTextSplitter` (`paragraph`, `sentence`) and"
        " `CharacterTextSplitter` (`fixed`).",
        "`target_tokens=500` rows shown; full results in `results.json`.",
        "Numbers are mean of 5 measured runs after 1 warmup.",
        "",
        "| Doc | Strategy | distillcore chunks/s | LangChain chunks/s |"
        " distillcore chars/s | LangChain chars/s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in results["benchmarks"]["B1"]["results"]:
        if row["target_tokens"] != 500:
            continue
        dc = row["distillcore"]
        lc = row["langchain"] or {}
        size_label = f"{row['doc_chars'] // 1000}K"
        def fmt(n: float | None) -> str:
            return f"{n:,.0f}" if isinstance(n, (int, float)) and n else "—"

        dc_chunks_s = dc["chunks"] / dc["elapsed_s"] if dc["elapsed_s"] else None
        lc_chunks_s = lc["chunks"] / lc["elapsed_s"] if lc.get("elapsed_s") else None
        lines.append(
            f"| {size_label} | {row['strategy']} | {fmt(dc_chunks_s)} | {fmt(lc_chunks_s)} | "
            f"{fmt(dc['chars_per_sec'])} | {fmt(lc.get('chars_per_sec'))} |"
        )

    lines += [
        "",
        "## B2 — PDF extraction throughput",
        "",
        "pdfplumber extraction. Synthetic PDFs are reportlab-generated,"
        " 70-char-wide text-only pages.",
        "Real-world docs (Federal Register notice) extract slower due to"
        " richer page layouts.",
        "",
        "| Document | Pages | Elapsed | Pages/s | Chars/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results["benchmarks"]["B2"]["results"]:
        if row.get("skipped"):
            lines.append(f"| {row['label']} | — | — | — | _skipped: {row['skipped']}_ |")
            continue
        lines.append(
            f"| {row['label']} | {row['pages']} | {row['elapsed_s']:.2f}s | "
            f"{row['pages_per_sec']:.1f} | {row['chars_per_sec']:,.0f} |"
        )

    lines += [
        "",
        "## B3 — Coverage accuracy",
        "",
        "End-to-end pipeline coverage on the 86-page Federal Register PDF. ",
        "Bag-of-words (gate) vs sequential (secondary signal).",
        "",
    ]
    b3 = results["benchmarks"]["B3"]
    if "skipped" in b3:
        lines.append(f"_skipped: {b3['skipped']}_")
    else:
        lines += [
            "| Stage | BoW coverage | Sequential coverage |",
            "|---|---:|---:|",
        ]
        for row in b3["results"]:
            lines.append(
                f"| Structuring | {row['structuring_coverage_bow']:.1%} | "
                f"{row['structuring_coverage_sequential']:.1%} |"
            )
            lines.append(
                f"| Chunking | {row['chunking_coverage_bow']:.1%} | "
                f"{row['chunking_coverage_sequential']:.1%} |"
            )
            lines.append(
                f"| End-to-end | {row['end_to_end_coverage_bow']:.1%} | "
                f"{row['end_to_end_coverage_sequential']:.1%} |"
            )

    lines += [
        "",
        "## B4 — End-to-end pipeline",
        "",
        "Wall time per document with full pipeline (classify + structure"
        " + chunk + enrich + embed). Network-bound stages dominate.",
        "",
    ]
    b4 = results["benchmarks"]["B4"]
    if "skipped" in b4:
        lines.append(f"_skipped: {b4['skipped']}_")
    else:
        for row in b4["results"]:
            lines.append(f"**{row['file']}** ({row['pages']} pages, {row['chunks']} chunks)")
            lines.append("")
            lines.append(f"Total: **{row['total_elapsed_s']:.1f}s**")
            lines.append("")
            lines += [
                "| Stage | Elapsed | % of total |",
                "|---|---:|---:|",
            ]
            for stage, d in row["stages"].items():
                if d.get("skipped"):
                    lines.append(f"| {stage} | — | _skipped_ |")
                else:
                    lines.append(f"| {stage} | {d['elapsed_s']:.2f}s | {d['pct']:.1f}% |")

    lines += [
        "",
        "## B5 — Search throughput (numpy cache vs Python fallback)",
        "",
        "Cosine-similarity search over a synthetic Store. `dim=384`,"
        " 10 results, 5 runs after warmup. Random vectors, no real LLM.",
        "Numpy path uses a single matmul against a cached L2-normalized"
        " float32 matrix; Python fallback iterates row-by-row.",
        "",
        "| Chunks | numpy ms / query | Python ms / query | speedup |",
        "|---:|---:|---:|---:|",
    ]
    b5 = results["benchmarks"].get("B5", {})
    for row in b5.get("results", []):
        numpy_ms = row["numpy"]["elapsed_ms"]
        py_ms = row["python_fallback"]["elapsed_ms"]
        sp = row["speedup"]
        lines.append(
            f"| {row['chunks']:,} | {numpy_ms:.2f} | {py_ms:.2f} | {sp:.1f}x |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run distillcore benchmarks.")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Include B3 + B4 (require OPENAI_API_KEY and spend tokens).",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Re-render README.md from existing results.json without re-running.",
    )
    args = parser.parse_args()

    if args.render_only:
        if not RESULTS_PATH.exists():
            print(f"error: {RESULTS_PATH} not found; run without --render-only first")
            return 1
        results = json.loads(RESULTS_PATH.read_text())
        README_PATH.write_text(_render_readme(results))
        print(f"rendered → {README_PATH}")
        return 0

    print("Running B1 (chunking throughput)...")
    b1 = _chunking.run()
    print("Running B2 (extraction throughput)...")
    b2 = _extraction.run()
    print(f"Running B3 (coverage accuracy, with_llm={args.with_llm})...")
    b3 = _coverage.run(with_llm=args.with_llm)
    print(f"Running B4 (end-to-end pipeline, with_llm={args.with_llm})...")
    b4 = _end_to_end.run(with_llm=args.with_llm)
    print("Running B5 (search throughput)...")
    b5 = _search.run()

    results = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "env": _env_info(),
        "with_llm": args.with_llm,
        "benchmarks": {"B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5},
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    README_PATH.write_text(_render_readme(results))
    print(f"\nwrote → {RESULTS_PATH}")
    print(f"wrote → {README_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
