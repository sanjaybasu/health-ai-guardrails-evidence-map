"""Fetch governance source documents as plain text and track changes by content hash."""
from __future__ import annotations

import hashlib
import io
import json
import re

import requests
import trafilatura
import yaml
from pypdf import PdfReader

from .llm import ROOT

SRC = ROOT / "data" / "governance_sources.yaml"
TEXT = ROOT / "data" / "governance" / "text"
HASHES = ROOT / "data" / "governance" / "hashes.json"
SIZES = ROOT / "data" / "governance" / "sizes.json"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/128 Safari/537.36"}


def sources() -> list[dict]:
    return yaml.safe_load(SRC.read_text())["sources"]


def _to_text(r: requests.Response) -> str:
    ct = r.headers.get("content-type", "")
    if "pdf" in ct or r.content[:4] == b"%PDF":
        pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages]
        return "\n".join(pages)
    if "json" in ct:   # eCFR renderer returns HTML inside JSON in some versions
        try:
            return trafilatura.extract(r.json().get("content", r.text)) or r.text
        except ValueError:
            pass
    return trafilatura.extract(r.text, include_tables=True, favor_recall=True) or ""


BLOCK_MARKERS = ("captcha", "are you a robot", "access denied", "unusual traffic", "enable javascript",
                 "preparing your download", "request blocked")


def _fetch_url(u: str) -> str:
    """PMC article pages block automated clients; fetch those through Europe PMC's full-text API instead."""
    m = re.search(r"pmc\.ncbi\.nlm\.nih\.gov/articles/(PMC\d+)", u)
    if m:
        r = requests.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{m.group(1)}/fullTextXML", headers=UA,
                         timeout=90)
        r.raise_for_status()
        from bs4 import BeautifulSoup
        return clean(BeautifulSoup(r.text, "lxml-xml").get_text(" "))
    r = requests.get(u, headers=UA, timeout=90)
    r.raise_for_status()
    return clean(_to_text(r))


def clean(t: str) -> str:
    t = t.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    t = re.sub(r"-\n(?=[a-z])", "", t)          # rejoin hyphenated line breaks
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def fetch_all(only: set[str] | None = None) -> dict[str, dict]:
    """Fetch every source; return {id: {"changed": bool, "chars": int}}."""
    TEXT.mkdir(parents=True, exist_ok=True)
    hashes = json.loads(HASHES.read_text()) if HASHES.exists() else {}
    sizes = json.loads(SIZES.read_text()) if SIZES.exists() else {}
    report = {}
    for s in sources():
        if only and s["id"] not in only:
            continue
        parts = []
        for u in s["urls"]:
            try:
                parts.append(_fetch_url(u))
            except Exception as e:  # keep the previous text if a fetch fails
                report[s["id"]] = {"error": repr(e)[:200]}
        if not parts or s["id"] in report:
            continue
        text = "\n\n".join(parts)
        prev_len = sizes.get(s["id"])
        low = text[:5000].lower()
        if any(b in low for b in BLOCK_MARKERS) or (prev_len and len(text) < 0.5 * prev_len):
            # A bot-check page or a truncated download is a failed fetch, not a revision.
            report[s["id"]] = {"error": f"suspect fetch ({len(text)} chars vs {prev_len} stored)"}
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        changed = hashes.get(s["id"]) != h
        if changed or not (TEXT / f"{s['id']}.txt").exists():
            (TEXT / f"{s['id']}.txt").write_text(text)
            hashes[s["id"]] = h
            sizes[s["id"]] = len(text)
        report[s["id"]] = {"changed": changed, "chars": len(text)}
    HASHES.write_text(json.dumps(hashes, indent=1, sort_keys=True))
    SIZES.write_text(json.dumps(sizes, indent=1, sort_keys=True))
    return report


def text(sid: str) -> str:
    return (TEXT / f"{sid}.txt").read_text()
