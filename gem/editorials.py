"""Systematic sweep of editorials, perspectives, and viewpoints on health care AI in prominent journals.

OpenAlex supplies every AI-titled item from a fixed list of venues since 2023, whether or not it has an abstract.
Records join the main corpus (so the screener classifies them), full text is read only where it is legally open
(Europe PMC for PMC articles, otherwise the open-access location OpenAlex reports), and per-journal coverage is
published so the share of editorials the map could not read is visible.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time

import requests

from . import fetch
from .llm import ROOT

VENUES = {   # display name -> ISSNs (checked against Europe PMC journal titles on 2026-09-30)
    "NEJM AI": ["2836-9386", "2836-9394"],
    "New England Journal of Medicine": ["0028-4793", "1533-4406"],
    "NEJM Catalyst": ["2642-0007"],
    "Lancet Digital Health": ["2589-7500"],
    "Lancet": ["0140-6736", "1474-547X"],
    "JAMA": ["0098-7484", "1538-3598"],
    "JAMA Internal Medicine": ["2168-6106", "2168-6114"],
    "JAMA Network Open": ["2574-3805"],
    "JAMA Health Forum": ["2689-0186"],
    "Nature Medicine": ["1078-8956", "1546-170X"],
    "npj Digital Medicine": ["2398-6352"],
    "BMJ": ["0959-8138", "1756-1833"],
    "Annals of Internal Medicine": ["0003-4819", "1539-3704"],
    "Health Affairs": ["0278-2715", "1544-5208"],
    "JAMIA": ["1067-5027", "1527-974X"],
    "Radiology: Artificial Intelligence": ["2638-6100"],
}
TITLE_TERMS = ["artificial intelligence", "machine learning", "large language model", "language models", "LLM",
               "generative AI", "ChatGPT", "chatbot", "algorithm", "predictive model", "foundation model", "AI"]
COVERAGE = ROOT / "data" / "editorial_coverage.json"
OA = "https://api.openalex.org/works"


def _abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def sweep(since: str = "2023-01-01", until: str | None = None) -> dict:
    """Pull AI-titled items from each venue; add new ones to the corpus. Returns counts per venue."""
    until = until or dt.date.today().isoformat()
    title = "|".join(TITLE_TERMS)
    counts, recs = {}, []
    for venue, issns in VENUES.items():
        cursor, n = "*", 0
        while cursor:
            params = {"filter": f"primary_location.source.issn:{'|'.join(issns)},from_publication_date:{since},"
                                f"to_publication_date:{until},title.search:{title}",
                      "per-page": 200, "cursor": cursor,
                      "select": "id,doi,title,publication_date,type,ids,abstract_inverted_index,open_access,"
                                "best_oa_location,cited_by_count"}
            d = fetch._get(OA, params).json()
            for w in d.get("results", []):
                if not re.search(r"\b(AI|artificial intelligence|machine learning|language model|LLM|chatbot|"
                                 r"ChatGPT|generative|algorithm|predictive model|foundation model)\b",
                                 w.get("title") or "", re.I):
                    continue
                ids = w.get("ids") or {}
                pmid = (ids.get("pmid") or "").rsplit("/", 1)[-1] or None
                pmcid = (ids.get("pmcid") or "").rsplit("/", 1)[-1] or None
                doi = (w.get("doi") or "").replace("https://doi.org/", "").lower() or None
                n += 1
                recs.append({"id": f"pmid:{pmid}" if pmid else f"doi:{doi}" if doi else w["id"],
                             "pmid": pmid, "pmcid": pmcid, "doi": doi, "arxiv": None,
                             "title": w.get("title") or "", "abstract": _abstract(w.get("abstract_inverted_index")),
                             "authors": "", "venue": venue, "date": w.get("publication_date"), "preprint": False,
                             "pub_types": [w.get("type") or ""], "open_access": (w.get("open_access") or {}).get("is_oa"),
                             "oa_url": (w.get("best_oa_location") or {}).get("pdf_url") or
                                       (w.get("best_oa_location") or {}).get("landing_page_url"),
                             "citations": w.get("cited_by_count"), "source": "openalex_venue_sweep",
                             "query": "editorial_sweep"})
            cursor = d.get("meta", {}).get("next_cursor")
            time.sleep(0.15)
        counts[venue] = n
    # Records without an abstract are kept here (editorials rarely have one); merge_new drops abstracts < 200 chars,
    # so add them directly with a flag instead.
    have = fetch.load()
    seen = set(have) | {p["doi"] for p in have.values() if p.get("doi")} | {fetch.norm_title(p["title"]) for p in have.values()}
    new = []
    for r in recs:
        keys = {r["id"], r.get("doi"), fetch.norm_title(r["title"])} - {None, ""}
        if keys & seen:
            continue
        seen |= keys
        r["fetched"] = dt.date.today().isoformat()
        r["no_abstract"] = len(r["abstract"]) < 200
        new.append(r)
    fetch.verify_ids(new)
    if new:
        allp = fetch.load()
        allp.update({p["id"]: p for p in new})
        fetch._write_all(allp.values())
    return {"per_venue": counts, "new_records": len(new), "without_abstract": sum(r["no_abstract"] for r in new)}


def open_text(p: dict, max_words: int = 9000) -> str:
    """Legally open full text: Europe PMC for PMC articles, else the open-access location OpenAlex reports."""
    from .evidence import fulltext
    t = fulltext(p, max_words=max_words)
    if t or not p.get("oa_url"):
        return t
    try:
        from .governance import UA, _to_text, clean
        r = requests.get(p["oa_url"], headers=UA, timeout=60)
        if r.status_code != 200:
            return ""
        text = clean(_to_text(r))
        low = text[:4000].lower()
        if len(text) < 1500 or any(b in low for b in ("captcha", "access denied", "sign in", "subscribe")):
            return ""
        return " ".join(text.split()[:max_words])
    except Exception:
        return ""


ALIASES = {"nejm ai": "NEJM AI", "the new england journal of medicine": "New England Journal of Medicine",
           "nejm catalyst innovations in care delivery": "NEJM Catalyst", "the lancet. digital health": "Lancet Digital Health",
           "lancet (london, england)": "Lancet", "jama": "JAMA", "jama internal medicine": "JAMA Internal Medicine",
           "jama network open": "JAMA Network Open", "jama health forum": "JAMA Health Forum",
           "nature medicine": "Nature Medicine", "npj digital medicine": "npj Digital Medicine",
           "bmj (clinical research ed.)": "BMJ", "annals of internal medicine": "Annals of Internal Medicine",
           "health affairs (project hope)": "Health Affairs",
           "journal of the american medical informatics association : jamia": "JAMIA",
           "radiology. artificial intelligence": "Radiology: Artificial Intelligence"}


def venue_of(p: dict) -> str | None:
    v = p.get("venue") or ""
    return v if v in VENUES else ALIASES.get(v.lower().strip())


def coverage() -> dict:
    """Per-venue share of recommendation-bearing commentaries whose text the map could read."""
    import glob
    papers = fetch.load()
    screen = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data" / "screening.jsonl").read_text().splitlines()}
    ex = {}
    for f in glob.glob(str(ROOT / "data" / "extractions" / "*.json")):
        x = json.loads(open(f).read())
        ex[x["id"]] = x
    out = {}
    for pid, p in papers.items():
        v = venue_of(p)
        if v not in VENUES or screen.get(pid, {}).get("category") != "recommendation_commentary":
            continue
        c = out.setdefault(v, {"commentaries": 0, "read": 0, "endorsements": 0})
        c["commentaries"] += 1
        x = ex.get(pid)
        if x and not x.get("skipped"):
            c["read"] += 1
            c["endorsements"] += len(x.get("endorsements", []))
    COVERAGE.write_text(json.dumps(out, indent=1, sort_keys=True))
    return out


def enrich_oa(ids: set[str]) -> int:
    """Add OpenAlex open-access locations to existing records (by DOI) so their full text can be read."""
    papers = fetch.load()
    todo = [papers[i] for i in ids if i in papers and papers[i].get("doi") and not papers[i].get("oa_url")]
    n = 0
    for k in range(0, len(todo), 50):
        chunk = todo[k:k + 50]
        try:
            d = fetch._get(OA, {"filter": "doi:" + "|".join(p["doi"] for p in chunk), "per-page": 50,
                                "select": "doi,best_oa_location,open_access"}).json()
        except RuntimeError:
            continue
        by = {(w.get("doi") or "").replace("https://doi.org/", "").lower(): w for w in d.get("results", [])}
        for p in chunk:
            w = by.get(p["doi"])
            loc = (w or {}).get("best_oa_location") or {}
            url = loc.get("pdf_url") or loc.get("landing_page_url")
            if url:
                p["oa_url"] = url
                n += 1
        time.sleep(0.15)
    fetch._write_all(papers.values())
    return n
