"""Deterministic synthetic fixtures for benchmarks.

Same seed → same output. No network, no real corpora needed for B1/B2.
"""

from __future__ import annotations

import random
from pathlib import Path

WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega document text section "
    "paragraph chunk extraction validation coverage pipeline embedding chunking "
    "structuring enrichment classification report analysis result metric "
    "throughput latency benchmark performance baseline regression"
).split()


def make_document(target_chars: int, seed: int = 42) -> str:
    """Generate a deterministic synthetic document of approximately ``target_chars``.

    Output is paragraph-structured: ~80-word paragraphs separated by blank lines.
    Sentences are 5-25 words ending in '.' '!' or '?'. Stable under a fixed seed.
    """
    rng = random.Random(seed)
    parts: list[str] = []
    total = 0
    while total < target_chars:
        sentences_in_para = rng.randint(4, 7)
        sentences = []
        for _ in range(sentences_in_para):
            n = rng.randint(5, 25)
            sent_words = [rng.choice(WORDS) for _ in range(n)]
            sent_words[0] = sent_words[0].capitalize()
            punct = rng.choice([".", ".", ".", "!", "?"])
            sentences.append(" ".join(sent_words) + punct)
        para = " ".join(sentences)
        parts.append(para)
        total += len(para) + 2  # blank-line separator
    return "\n\n".join(parts)


def make_pdf(path: Path, pages: int, seed: int = 42) -> Path:
    """Generate a deterministic synthetic PDF with ``pages`` pages of text."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen.canvas import Canvas

    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    line_height = 14
    margin = 72  # 1 inch
    max_lines = int((height - 2 * margin) / line_height)
    chars_per_line = 70

    for _ in range(pages):
        y = height - margin
        for _ in range(max_lines):
            words: list[str] = []
            line_len = 0
            while line_len < chars_per_line:
                w = rng.choice(WORDS)
                if line_len + len(w) + 1 > chars_per_line:
                    break
                words.append(w)
                line_len += len(w) + 1
            c.drawString(margin, y, " ".join(words))
            y -= line_height
        c.showPage()
    c.save()
    return path


def _smoke() -> None:
    """Quick sanity check; not run in CI."""
    import tempfile

    doc = make_document(1000)
    # One paragraph (~4-7 sentences, 5-25 words each) overshoots target by 50-100%.
    assert 1000 < len(doc) < 2500, f"unexpected len {len(doc)}"
    assert make_document(1000) == make_document(1000), "non-deterministic"
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        make_pdf(Path(f.name), pages=2)
        assert Path(f.name).stat().st_size > 0
    print("fixtures: ok")


if __name__ == "__main__":
    _smoke()
