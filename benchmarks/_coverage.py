"""B3: coverage accuracy on real document.

Requires OPENAI_API_KEY (full pipeline runs classification + structuring + embedding).
Skipped unless ``--with-llm`` is passed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from distillcore import DistillConfig, process_document
from distillcore.validation.coverage import (
    compute_coverage,
    compute_coverage_sequential,
)

REAL_PDF = Path(__file__).parent.parent / "test-files" / "2026-06963.pdf"


def run(with_llm: bool = False) -> dict[str, Any]:
    if not with_llm:
        return {
            "benchmark": "B3_coverage_accuracy",
            "skipped": "requires --with-llm (uses OpenAI API)",
        }
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "benchmark": "B3_coverage_accuracy",
            "skipped": "OPENAI_API_KEY not set",
        }
    if not REAL_PDF.exists():
        return {
            "benchmark": "B3_coverage_accuracy",
            "skipped": f"fixture missing: {REAL_PDF}",
        }

    config = DistillConfig()
    result = process_document(REAL_PDF, config=config, embed=False)

    chunk_text = "\n".join(c.text for c in result.chunks)
    bow_e2e = compute_coverage(result.document.full_text, chunk_text)
    seq_e2e = compute_coverage_sequential(result.document.full_text, chunk_text)

    return {
        "benchmark": "B3_coverage_accuracy",
        "results": [
            {
                "file": REAL_PDF.name,
                "pages": result.document.page_count,
                "chars": len(result.document.full_text),
                "chunks": len(result.chunks),
                "structuring_coverage_bow": round(result.validation.structuring_coverage, 4),
                "structuring_coverage_sequential": round(
                    result.validation.structuring_coverage_sequential, 4
                ),
                "chunking_coverage_bow": round(result.validation.chunking_coverage, 4),
                "chunking_coverage_sequential": round(
                    result.validation.chunking_coverage_sequential, 4
                ),
                "end_to_end_coverage_bow": round(bow_e2e, 4),
                "end_to_end_coverage_sequential": round(seq_e2e, 4),
                "passed": result.validation.passed,
                "warnings": result.validation.warnings[:5],
            }
        ],
    }
