"""Literature retrieval: Europe PMC (PubMed/MEDLINE plus medRxiv, bioRxiv, Research Square and other preprints),
arXiv, and OpenAlex (citation counts and preprint-to-publication links). Records land in data/papers.jsonl,
deduplicated by DOI, PMID, arXiv id, and normalized title."""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path

import feedparser
import requests

from .llm import CFG, ROOT

PAPERS = ROOT / "data" / "papers.jsonl"
UA = {"User-Agent": "health-ai-guardrails-evidence-map (mailto:sanjay.basu@waymarkcare.com)"}
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _get(url, params=None, tries=7):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=180)
            if r.status_code == 200:
                return r
            last = r.status_code
        except requests.RequestException as e:
            last = repr(e)[:120]
        time.sleep(min(120, 3 * 2 ** i))
    raise RuntimeError(f"GET failed: {url} ({last})")


def europepmc(since: str, until: str, page_size: int = 500, limit: int | None = None, query: str = ""):
    q = f"({query}) AND FIRST_PDATE:[{since} TO {until}] AND (SRC:MED OR SRC:PPR)"
    cursor, n = "*", 0
    while True:
        d = _get(EPMC, {"query": q, "format": "json", "pageSize": page_size, "cursorMark": cursor,
                        "resultType": "core"}).json()
        for r in d["resultList"]["result"]:
            n += 1
            is_pre = r.get("source") == "PPR"
            venue = ((r.get("bookOrReportDetails") or {}).get("publisher") if is_pre
                     else ((r.get("journalInfo") or {}).get("journal") or {}).get("title"))
            yield {
                "id": f"pmid:{r['pmid']}" if r.get("pmid") else f"epmc:{r['id']}",
                "pmid": r.get("pmid"), "pmcid": r.get("pmcid"), "doi": (r.get("doi") or "").lower() or None,
                "arxiv": None, "title": r.get("title", "").rstrip("."), "abstract": r.get("abstractText") or "",
                "authors": r.get("authorString", ""), "venue": venue, "date": r.get("firstPublicationDate"),
                "preprint": is_pre, "pub_types": (r.get("pubTypeList") or {}).get("pubType", []),
                "open_access": r.get("isOpenAccess") == "Y", "source": "europepmc",
            }
            if limit and n >= limit:
                return
        nxt = d.get("nextCursorMark")
        if not nxt or nxt == cursor or not d["resultList"]["result"]:
            return
        cursor = nxt
        time.sleep(0.2)


def arxiv(since: str, until: str, limit: int = 2000):
    cats = " OR ".join(f"cat:{c}" for c in CFG["search"]["arxiv_categories"])
    q = f"({CFG['search']['arxiv_query']}) AND ({cats})"
    start, lo, hi = 0, since, until
    while start < limit:
        feed = feedparser.parse(_get("http://export.arxiv.org/api/query", {
            "search_query": q, "start": start, "max_results": 200, "sortBy": "submittedDate",
            "sortOrder": "descending"}).text)
        if not feed.entries:
            return
        for e in feed.entries:
            date = e.published[:10]
            if date > hi:
                continue
            if date < lo:
                return
            aid = e.id.rsplit("/abs/", 1)[-1].split("v")[0]
            yield {"id": f"arxiv:{aid}", "pmid": None, "pmcid": None, "doi": f"10.48550/arxiv.{aid}".lower(),
                   "arxiv": aid, "title": " ".join(e.title.split()), "abstract": " ".join(e.summary.split()),
                   "authors": ", ".join(a.name for a in e.authors), "venue": "arXiv", "date": date,
                   "preprint": True, "pub_types": ["Preprint"], "open_access": True, "source": "arxiv"}
        start += 200
        time.sleep(3.1)   # arXiv asks for >= 3 s between requests


def openalex_enrich(papers: list[dict]):
    """Add citation counts and, for preprints, the DOI of any later journal version (OpenAlex locations)."""
    dois = [p["doi"] for p in papers if p.get("doi")]
    for i in range(0, len(dois), 50):
        chunk = dois[i:i + 50]
        try:
            d = _get("https://api.openalex.org/works", {
                "filter": "doi:" + "|".join(chunk), "per-page": 50,
                "select": "doi,cited_by_count,locations,type"}).json()
        except RuntimeError:
            continue
        by = {(w["doi"] or "").replace("https://doi.org/", "").lower(): w for w in d.get("results", [])}
        for p in papers:
            w = by.get(p.get("doi") or "")
            if w:
                p["citations"] = w.get("cited_by_count", 0)
        time.sleep(0.2)


def verify_ids(papers: list[dict]) -> list[dict]:
    """Keep records whose PMID title matches NCBI esummary, or whose DOI resolves at doi.org (preprints, arXiv)."""
    from .llm import _key
    key = _key("NCBI_API_KEY") or _key("PUBMED_API_KEY")
    pm = [p for p in papers if p.get("pmid")]
    titles = {}
    for i in range(0, len(pm), 200):
        params = {"db": "pubmed", "id": ",".join(p["pmid"] for p in pm[i:i + 200]), "retmode": "json"}
        if key:
            params["api_key"] = key
        try:
            res = _get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params).json()["result"]
            titles.update({u: res[u].get("title", "") for u in res.get("uids", [])})
        except (RuntimeError, KeyError, ValueError):
            pass
        time.sleep(0.12 if key else 0.4)
    kept = []
    for p in papers:
        if p.get("pmid"):
            p["id_verified"] = norm_title(titles.get(p["pmid"], ""))[:60] == norm_title(p["title"])[:60]
        elif p.get("doi"):
            try:
                r = requests.head(f"https://doi.org/{p['doi']}", headers=UA, timeout=30, allow_redirects=False)
                p["id_verified"] = r.status_code in (301, 302, 303, 307, 308)
            except requests.RequestException:
                p["id_verified"] = False
        else:
            p["id_verified"] = False
        kept.append(p)
    return kept


def load() -> dict[str, dict]:
    out = {}
    if PAPERS.exists():
        for line in PAPERS.read_text().splitlines():
            p = json.loads(line)
            out[p["id"]] = p
    return out


def merge_new(records) -> list[dict]:
    """Append records not already present (by id, DOI, or normalized title). Returns the new ones."""
    have = load()
    seen = set(have)
    seen |= {p["doi"] for p in have.values() if p.get("doi")}
    seen |= {norm_title(p["title"]) for p in have.values()}
    new = []
    for r in records:
        keys = {r["id"], r.get("doi"), norm_title(r["title"])} - {None, ""}
        if keys & seen or not r["title"] or len(r.get("abstract", "")) < 200:
            continue
        seen |= keys
        r["fetched"] = dt.date.today().isoformat()
        new.append(r)
    if new:
        with PAPERS.open("a") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
    return new


def _months(since: str, until: str):
    """Monthly [lo, hi] windows; small windows keep Europe PMC cursor paging shallow."""
    a, b = dt.date.fromisoformat(since), dt.date.fromisoformat(until)
    while a <= b:
        nxt = (a.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        yield a.isoformat(), min(b, nxt - dt.timedelta(days=1)).isoformat()
        a = nxt


def update(since: str, until: str | None = None, limit: int | None = None) -> list[dict]:
    """Fetch, merge page by page (so an interrupted fetch keeps its progress), then enrich and verify new records."""
    until = until or dt.date.today().isoformat()
    new = []
    def flush(buf):
        new.extend(merge_new(buf))
        buf.clear()
    failed = []
    for name, q in CFG["search"]["europepmc_queries"].items():
        for lo, hi in _months(since, until):
            buf = []
            try:
                for r in europepmc(lo, hi, limit=limit, query=q):
                    buf.append({**r, "query": name})
            except RuntimeError as e:
                failed.append({"query": name, "since": lo, "until": hi, "error": str(e)[:160]})
                print(f"  window failed {name} {lo}..{hi}; kept {len(buf)} records", flush=True)
            flush(buf)
    (ROOT / "state" / "fetch_failed_windows.json").write_text(json.dumps(failed, indent=1))
    flush(list(arxiv(since, until, limit=limit or 2000)))
    openalex_enrich(new)
    verify_ids(new)
    if new:   # rewrite with citation counts and identifier checks
        allp = load()
        allp.update({p["id"]: p for p in new})
        tmp = PAPERS.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(p) + "\n" for p in allp.values()))
        tmp.replace(PAPERS)
    return new
