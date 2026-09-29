"""Deterministic quote verification. An extracted quote is kept only if it appears in the source text."""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from .llm import CFG


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").lower()
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"(?m)^\s*\d{1,4}\s+", " ", t)          # statute line numbers
    t = re.sub(r"[‐-―-]", "-", t)
    t = re.sub(r"[^a-z0-9%.,;:'\"()/<>=+\- ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def verify(quote: str, source: str, normalized_source: str | None = None) -> tuple[bool, float]:
    """Return (ok, score). Exact normalized containment scores 100; otherwise fuzzy partial match."""
    q = norm(quote)
    if len(q) < 20:
        return False, 0.0
    s = normalized_source if normalized_source is not None else norm(source)
    if q in s:
        return True, 100.0
    # Ellipses in quotes: every fragment must be present, in order.
    frags = [f.strip() for f in re.split(r"\.\.\.|…", q) if len(f.strip()) >= 15]
    if len(frags) > 1:
        pos = 0
        for f in frags:
            i = s.find(f, pos)
            if i < 0:
                break
            pos = i + len(f)
        else:
            return True, 99.0
    score = fuzz.partial_ratio(q, s, score_cutoff=CFG["quote_match_threshold"])
    return score >= CFG["quote_match_threshold"], float(score)
