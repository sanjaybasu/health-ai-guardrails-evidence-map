"""Compute per-recommendation status and build the static site into docs/."""
from __future__ import annotations

import datetime as dt
import json
import shutil
from collections import Counter, defaultdict

import yaml
from jinja2 import Environment, FileSystemLoader

from . import governance
from .evidence import COMMENTARY_EDGES, EVIDENCE, LEVEL_LABEL
from .llm import ROOT
from .taxonomy import DOMAINS, ENDORSE, TAXONOMY

DOCS = ROOT / "docs"
SITE = ROOT / "site"
UNDERSERVED = ("medicaid", "safety_net", "limited_english_or_non_english", "low_health_literacy", "low_income")
STATUS_LABEL = {
    "evidence_supported": "Favoured by a controlled deployment study",
    "contested": "Contested",
    "limited_evidence": "Tested, weaker designs only",
    "consensus_without_evidence": "Widely endorsed, untested",
    "hazard_documented_untested": "Hazard documented, practice untested",
    "no_data": "No studies yet",
}


def _jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def status(n_endorse: int, tests: list[dict]) -> str:
    if tests:
        best = min(e["level"] for e in tests)
        at_best = [e["finding"] for e in tests if e["level"] == best]
        if "supports_practice" in at_best and "against_practice" in at_best:
            return "contested"
        if best <= 2 and "supports_practice" in at_best and not any(
                e["finding"] == "against_practice" and e["level"] <= best for e in tests):
            return "evidence_supported"
        return "limited_evidence"
    return "consensus_without_evidence" if n_endorse >= 3 else "no_data"


def compute() -> dict:
    tax = yaml.safe_load(TAXONOMY.read_text())
    srcs = {s["id"]: s for s in governance.sources()}
    endorse = defaultdict(list)
    for e in _jsonl(ENDORSE):
        s = srcs[e["source"]]
        endorse[e["rec"]].append({"source": e["source"], "title": s["title"], "issuer": s["issuer"],
                                  "kind": s["kind"], "date": str(s["date"]), "url": s["landing"],
                                  "strength": e["strength"], "quote": e["quote"]})
    for e in _jsonl(COMMENTARY_EDGES):
        endorse[e["rec"]].append({"source": e["study"], "title": e["title"], "issuer": e.get("venue") or "",
                                  "kind": "commentary", "date": e.get("date") or "",
                                  "url": f"https://doi.org/{e['doi']}" if e.get("doi") else "",
                                  "strength": "should", "quote": e["quote"]})
    comm = {c["rec"]: c for c in _jsonl(ROOT / "data" / "commentary.jsonl")}
    ev = defaultdict(list)
    for e in _jsonl(EVIDENCE):
        ev[e["rec"]].append(e)
    recs = []
    for r in tax["recommendations"]:
        es = endorse[r["id"]]
        tests = [e for e in ev[r["id"]] if e["relation"] == "tests_guardrail"]
        haz = [e for e in ev[r["id"]] if e["relation"] == "documents_hazard"]
        n_formal = len({e["source"] for e in es if e["kind"] != "commentary"})
        n_comment = len({e["source"] for e in es if e["kind"] == "commentary"})
        st = status(n_formal + n_comment, tests)
        if st in ("no_data", "consensus_without_evidence") and haz and n_formal + n_comment < 3:
            st = "hazard_documented_untested"
        by_tech = {}
        for t in tax.get("technologies", {}):
            if t == "any_ai":
                continue
            tt = [e for e in tests if t in (e.get("technology") or [])]
            th = [e for e in haz if t in (e.get("technology") or [])]
            st_t = status(n_formal + n_comment, tt)
            if st_t in ("no_data", "consensus_without_evidence") and th and n_formal + n_comment < 3:
                st_t = "hazard_documented_untested"
            by_tech[t] = {"status": st_t, "status_label": STATUS_LABEL[st_t],
                          "best_level": min((e["level"] for e in tt), default=None),
                          "n_tests": len({e["study"] for e in tt}), "n_hazard": len({e["study"] for e in th}),
                          "underserved_tested": any(any(e["population"].get(f) for f in UNDERSERVED) for e in tt)}
        recs.append({
            **r, "by_tech": by_tech, "curation": r.get("status", "machine_draft"), "domain_label": DOMAINS[r["domain"]],
            "n_endorse": n_formal + n_comment, "n_formal": n_formal, "n_commentary": n_comment,
            "n_must": len({e["source"] for e in es if e["strength"] == "must"}),
            "n_tests": len({e["study"] for e in tests}), "n_hazard": len({e["study"] for e in haz}),
            "best_level": min((e["level"] for e in tests), default=None),
            "tests_findings": dict(Counter(e["finding"] for e in tests)),
            "hazard_findings": dict(Counter(e["finding"] for e in haz)),
            "underserved_tested": any(any(e["population"].get(f) for f in UNDERSERVED) for e in tests),
            "underserved_hazard": any(any(e["population"].get(f) for f in UNDERSERVED) for e in haz),
            "status": st, "status_label": STATUS_LABEL[st],
            "commentary": {k: comm[r["id"]].get(k) for k in ("bottom_line", "certainty", "perspectives",
                                                              "research_gap", "underserved_note")} if r["id"] in comm else None,
            "endorsements": sorted(es, key=lambda e: (e["kind"] == "commentary", e["strength"] != "must")),
            "evidence": sorted(ev[r["id"]], key=lambda e: (e["relation"] != "tests_guardrail", e["level"])),
        })
    return {"recs": recs, "domains": DOMAINS, "technologies": tax.get("technologies", {}), "levels": LEVEL_LABEL, "status_labels": STATUS_LABEL,
            "sources": list(srcs.values())}


def stats(data: dict) -> dict:
    screen = _jsonl(ROOT / "data" / "screening.jsonl")
    from .fetch import load
    papers = list(load().values())
    ex = [json.loads(p.read_text()) for p in (ROOT / "data" / "extractions").glob("*.json")]
    panel = [x for x in ex if x["route"] in ("panel", "fast_panel") and x.get("agreement")]
    agree = defaultdict(list)
    for x in panel:
        for k, v in x["agreement"].items():
            agree[k].append(v)
    audit_p = ROOT / "data" / "screen_audit.json"
    audit = json.loads(audit_p.read_text()) if audit_p.exists() else None
    ev = _jsonl(EVIDENCE)
    return {
        "updated": dt.date.today().isoformat(),
        "n_papers": len(papers), "n_preprints": sum(p.get("preprint", False) for p in papers),
        "n_screened": len(screen),
        "screen": dict(Counter(s["category"] for s in screen)),
        "n_practice_testing": sum(1 for s in screen if s["tests_practice"] and s["category"] == "empirical_ai_health"),
        "routes": dict(Counter(x["route"] for x in ex)),
        "n_evidence_edges": len(ev), "n_tests_edges": sum(e["relation"] == "tests_guardrail" for e in ev),
        "panel_agreement": {k: round(sum(v) / len(v), 3) for k, v in agree.items()},
        "n_panel": len(panel),
        "screen_audit": {k: audit[k] for k in ("sampled", "disagreements")} if audit else None,
        "n_recs": len(data["recs"]), "status_counts": dict(Counter(r["status"] for r in data["recs"])),
        "n_sources": len(data["sources"]),
    }


def build():
    data = compute()
    data["stats"] = stats(data)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data").mkdir(exist_ok=True)
    # Summary for the map and table; full detail per recommendation, loaded when a recommendation is opened.
    detail_keys = ("endorsements", "evidence", "commentary")
    (DOCS / "data" / "rec").mkdir(parents=True, exist_ok=True)
    for r in data["recs"]:
        (DOCS / "data" / "rec" / f"{r['id']}.json").write_text(
            json.dumps({k: r.get(k) for k in detail_keys}, separators=(",", ":"), default=str))
    summary = {**data, "recs": [{k: v for k, v in r.items() if k not in detail_keys} for r in data["recs"]]}
    (DOCS / "data" / "map.json").write_text(json.dumps(summary, separators=(",", ":"), default=str))
    (DOCS / "data" / "map_full.json").unlink(missing_ok=True)
    for f in (SITE / "static").iterdir():
        shutil.copy(f, DOCS / f.name)
    env = Environment(loader=FileSystemLoader(SITE / "templates"), autoescape=True)
    import hashlib
    build = hashlib.sha256(b"".join((SITE / "static" / f).read_bytes() for f in ("app.js", "style.css"))
                           + (DOCS / "data" / "map.json").read_bytes()).hexdigest()[:10]
    for name in ("index.html", "methods.html", "changelog.html"):
        (DOCS / name).write_text(env.get_template(name).render(s=data["stats"], changelog=_changelog(), build=build))
    (DOCS / ".nojekyll").write_text("")
    print(f"site: {len(data['recs'])} recommendations; {data['stats']['n_evidence_edges']} evidence edges")


def _changelog():
    p = ROOT / "data" / "changelog.jsonl"
    return list(reversed(_jsonl(p)))[:52]
