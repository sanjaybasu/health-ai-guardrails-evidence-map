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


def clean(t: str) -> str:
    t = t.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    t = re.sub(r"-\n(?=[a-z])", "", t)          # rejoin hyphenated line breaks
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def fetch_all(only: set[str] | None = None) -> dict[str, dict]:
    """Fetch every source; return {id: {"changed": bool, "chars": int}}."""
    TEXT.mkdir(parents=True, exist_ok=True)
    hashes = json.loads(HASHES.read_text()) if HASHES.exists() else {}
    report = {}
    for s in sources():
        if only and s["id"] not in only:
            continue
        parts = []
        for u in s["urls"]:
            try:
                r = requests.get(u, headers=UA, timeout=90)
                r.raise_for_status()
                parts.append(clean(_to_text(r)))
            except Exception as e:  # keep the previous text if a fetch fails
                report[s["id"]] = {"error": repr(e)[:200]}
        if not parts or s["id"] in report:
            continue
        text = "\n\n".join(parts)
        h = hashlib.sha256(text.encode()).hexdigest()
        changed = hashes.get(s["id"]) != h
        if changed or not (TEXT / f"{s['id']}.txt").exists():
            (TEXT / f"{s['id']}.txt").write_text(text)
            hashes[s["id"]] = h
        report[s["id"]] = {"changed": changed, "chars": len(text)}
    HASHES.write_text(json.dumps(hashes, indent=1, sort_keys=True))
    return report


def text(sid: str) -> str:
    return (TEXT / f"{sid}.txt").read_text()
