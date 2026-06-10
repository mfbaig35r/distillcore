"""B2: PDF extraction throughput via pdfplumber."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from distillcore import extract

from ._fixtures import make_pdf

PAGE_COUNTS = [10, 50, 100]
REAL_PDF = Path(__file__).parent.parent / "test-files" / "2026-06963.pdf"


def _bench_pdf(path: Path, label: str) -> dict[str, Any]:
    start = time.perf_counter()
    result = extract(path)
    elapsed = time.perf_counter() - start
    pages = len(result.pages)
    return {
        "label": label,
        "file": str(path.name),
        "pages": pages,
        "elapsed_s": round(elapsed, 3),
        "pages_per_sec": round(pages / elapsed, 2) if elapsed > 0 else None,
        "chars_extracted": len(result.full_text),
        "chars_per_sec": round(len(result.full_text) / elapsed, 0) if elapsed > 0 else None,
    }


def run() -> dict[str, Any]:
    """Run the extraction benchmark suite."""
    results: list[dict[str, Any]] = []
    workdir = Path(tempfile.mkdtemp(prefix="distillcore-bench-pdf-"))
    try:
        for pages in PAGE_COUNTS:
            pdf_path = workdir / f"synthetic-{pages}p.pdf"
            make_pdf(pdf_path, pages=pages)
            results.append(_bench_pdf(pdf_path, f"synthetic_{pages}p"))

        if REAL_PDF.exists():
            results.append(_bench_pdf(REAL_PDF, "real_federal_register"))
        else:
            results.append({"label": "real_federal_register", "skipped": "fixture not found"})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {"benchmark": "B2_extraction_throughput", "results": results}
