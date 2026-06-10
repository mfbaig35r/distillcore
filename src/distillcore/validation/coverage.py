"""Content coverage validation utilities."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(s: str) -> str:
    """Strip, unicode-normalize, and collapse whitespace."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def compute_coverage(original: str, derived: str) -> float:
    """Compute what fraction of original text is present in derived text.

    Both texts are normalized before comparison. Returns 0.0-1.0.

    Note:
        Uses bag-of-words matching: counts the fraction of words from
        *original* that appear anywhere in *derived*. This can report
        high coverage even when content is reordered or duplicated.
        For highly repetitive input (tabular data, form letters), the
        metric may produce false positives — use
        :func:`compute_coverage_sequential` as a second signal in those cases.
    """
    norm_orig = normalize_text(original)
    norm_derived = normalize_text(derived)

    if not norm_orig:
        return 1.0

    orig_words = norm_orig.split()
    if not orig_words:
        return 1.0

    derived_words = set(norm_derived.lower().split())
    matched = sum(1 for w in orig_words if w.lower() in derived_words)
    return matched / len(orig_words)


def compute_coverage_sequential(original: str, derived: str) -> float:
    """Compute order-preserving word coverage of original in derived.

    Walks the original word list forward, advancing a pointer through the
    derived word list to find each word at-or-after its current position.
    Equivalent to: fraction of original words that appear as a subsequence
    of derived. O(n+m).

    This complements :func:`compute_coverage` (bag-of-words). It catches
    three classes of failure that bag-of-words misses:

    - Repetitive content where one chunk is duplicated and another dropped
      (bag-of-words still sees every word in the dictionary).
    - Reordered chunks (concatenation order swapped).
    - Tabular or form-letter input where the same vocabulary repeats.

    Returns 0.0-1.0. Empty original returns 1.0 to match bag-of-words.
    """
    norm_orig = normalize_text(original)
    norm_derived = normalize_text(derived)

    if not norm_orig:
        return 1.0

    orig_words = norm_orig.lower().split()
    if not orig_words:
        return 1.0

    derived_words = norm_derived.lower().split()
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


def find_missing_segments(original: str, derived: str, min_length: int = 50) -> list[str]:
    """Find segments of original text that are missing from derived text.

    Returns segments of at least min_length characters where less than
    50% of words appear in the derived text.
    """
    norm_orig = normalize_text(original)
    norm_derived = normalize_text(derived)

    if not norm_orig:
        return []

    segments = re.split(r"(?<=[.!?])\s+", norm_orig)
    derived_words = set(norm_derived.lower().split())
    missing = []

    for segment in segments:
        segment = segment.strip()
        if len(segment) < min_length:
            continue
        words = segment.split()
        if not words:
            continue
        matched = sum(1 for w in words if w.lower() in derived_words)
        coverage = matched / len(words)
        if coverage < 0.5:
            missing.append(segment)

    return missing
