"""Deterministic prose linter for generated commentary and site text (rules from the maintainer's style guide)."""
from __future__ import annotations

import re

BANNED_WORDS = ["delve", "harness", "tapestry", "leverage", "leverages", "leveraging", "robust", "seamless",
                "seamlessly", "navigate", "navigating", "unleash", "elevate", "pivotal", "synergy", "holistic",
                "paradigm", "ecosystem", "journey", "unlock", "streamline", "optimize", "empower", "transformative",
                "cutting-edge", "crucial", "critical", "landscape", "realm", "underscores", "novel", "striking",
                "groundbreaking", "game-changer", "comprehensive"]
BANNED_PHRASES = ["it's worth noting", "it is worth noting", "worth flagging", "the takeaway", "the upshot",
                  "the key insight", "the headline", "what's striking", "what matters here", "the interesting part",
                  "highlights the importance", "plays a crucial role", "in today's", "in summary", "overall,",
                  "ultimately,", "not just", "not only", "it's not about", "moreover", "furthermore", "notably",
                  "importantly", "in conclusion"]
TECHNICAL_OK = {"doubly robust", "cluster-robust", "robust standard error", "robust to"}

SENT = re.compile(r"(?<=[.!?])\s+")


def lint(text: str) -> list[str]:
    """Return a list of violations (empty when clean)."""
    out = []
    low = text.lower()
    scrub = low
    for ok in TECHNICAL_OK:
        scrub = scrub.replace(ok, "")
    for w in BANNED_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", scrub):
            out.append(f"banned word: {w}")
    for p in BANNED_PHRASES:
        if p in low:
            out.append(f"banned phrase: {p}")
    if "—" in text or " -- " in text:
        out.append("em dash")
    if "!" in text:
        out.append("exclamation mark")
    for s in SENT.split(text.strip()):
        words = s.split()
        if not words:
            continue
        # colon-fragment line: short left side without a verb-like token, e.g. "The result: fewer errors."
        if ":" in s:
            left = s.split(":", 1)[0].split()
            if 0 < len(left) <= 3 and not any(t.lower() in {"is", "are", "was", "were", "found", "shows", "show"} for t in left):
                out.append(f"colon fragment: {s[:60]}")
        if len(words) <= 3 and not re.search(r"\d", s):
            out.append(f"fragment sentence: {s[:60]}")
        if re.match(r"^(two|three|four|five) (conclusions|findings|lessons|themes) (emerge|follow)", s.lower()):
            out.append(f"conclusion-first inversion: {s[:60]}")
    return out


def lint_obj(obj) -> list[str]:
    """Lint every string inside a JSON-like object."""
    if isinstance(obj, str):
        return lint(obj)
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in lint_obj(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in lint_obj(x)]
    return []
