"""B4: end-to-end pipeline timing with stage breakdown.

Requires OPENAI_API_KEY. Skipped unless ``--with-llm`` is passed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from distillcore import DistillConfig, process_document

REAL_PDF = Path(__file__).parent.parent / "test-files" / "2026-06963.pdf"

# Stage start → stage_done pairs we expect to see.
STAGES = ["extraction", "classification", "structuring", "chunking", "enrichment", "embedding"]


def run(with_llm: bool = False) -> dict[str, Any]:
    if not with_llm:
        return {
            "benchmark": "B4_end_to_end_pipeline",
            "skipped": "requires --with-llm (uses OpenAI API)",
        }
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "benchmark": "B4_end_to_end_pipeline",
            "skipped": "OPENAI_API_KEY not set",
        }
    if not REAL_PDF.exists():
        return {
            "benchmark": "B4_end_to_end_pipeline",
            "skipped": f"fixture missing: {REAL_PDF}",
        }

    stage_starts: dict[str, float] = {}
    stage_elapsed: dict[str, float] = {}

    def on_progress(stage: str, data: dict[str, Any]) -> None:
        now = time.perf_counter()
        if stage.endswith("_done"):
            base = stage[: -len("_done")]
            if base in stage_starts:
                stage_elapsed[base] = now - stage_starts[base]
        elif stage in STAGES:
            stage_starts[stage] = now

    config = DistillConfig(on_progress=on_progress)

    total_start = time.perf_counter()
    result = process_document(REAL_PDF, config=config, embed=True)
    total_elapsed = time.perf_counter() - total_start

    breakdown: dict[str, Any] = {}
    for stage in STAGES:
        if stage in stage_elapsed:
            elapsed = stage_elapsed[stage]
            breakdown[stage] = {
                "elapsed_s": round(elapsed, 3),
                "pct": round(100 * elapsed / total_elapsed, 1),
            }
        else:
            breakdown[stage] = {"elapsed_s": None, "pct": None, "skipped": True}

    return {
        "benchmark": "B4_end_to_end_pipeline",
        "results": [
            {
                "file": REAL_PDF.name,
                "pages": result.document.page_count,
                "chunks": len(result.chunks),
                "total_elapsed_s": round(total_elapsed, 3),
                "stages": breakdown,
            }
        ],
    }
