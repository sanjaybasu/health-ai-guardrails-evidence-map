"""Human review packets and import of human decisions.

Packets go to notebooks/health-ai-guardrails-evidence-map/review/. Reviewer columns are left blank for a person to fill;
nothing in this module writes a reviewer decision. `import_decisions` copies filled rows back into the data files.
"""
from __future__ import annotations

import csv
import json
import random

import yaml

from .evidence import EVIDENCE
from .llm import ROOT
from .taxonomy import ENDORSE, TAXONOMY

REVIEW = ROOT.parents[1] / "notebooks" / "health-ai-guardrails-evidence-map" / "review"


def taxonomy_packet():
    REVIEW.mkdir(parents=True, exist_ok=True)
    t = yaml.safe_load(TAXONOMY.read_text())
    ends = [json.loads(l) for l in ENDORSE.read_text().splitlines()]
    by = {}
    for e in ends:
        by.setdefault(e["rec"], []).append(e)
    path = REVIEW / "taxonomy_review.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "domain", "statement", "hazard", "applies_to", "genai_specific", "n_sources", "sources",
                    "example_quote", "reviewer_decision (keep/edit/merge_into/drop)", "reviewer_statement", "reviewer_notes"])
        for r in t["recommendations"]:
            es = by.get(r["id"], [])
            w.writerow([r["id"], r["domain"], r["statement"], r["hazard"], ";".join(r["applies_to"]), r["genai_specific"],
                        len(es), ";".join(sorted(e["source"] for e in es)), es[0]["quote"] if es else "", "", "", ""])
    return path


def evidence_packet(n: int = 60, seed: int = 20260928):
    """Random sample of evidence links, stratified so every tests_guardrail link from a deployment is included."""
    REVIEW.mkdir(parents=True, exist_ok=True)
    ev = [json.loads(l) for l in EVIDENCE.read_text().splitlines()]
    must = [e for e in ev if e["relation"] == "tests_guardrail" and e["level"] <= 3]
    rest = [e for e in ev if e not in must]
    random.Random(seed).shuffle(rest)
    sample = must + rest[:max(0, n - len(must))]
    t = {r["id"]: r["statement"] for r in yaml.safe_load(TAXONOMY.read_text())["recommendations"]}
    path = REVIEW / "evidence_spotcheck.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "doi", "title", "rec", "rec_statement", "relation", "finding", "level", "design", "setting",
                    "effect", "quote", "reviewer_link_correct (y/n)", "reviewer_finding_correct (y/n)",
                    "reviewer_level_correct (y/n)", "reviewer_notes"])
        for e in sample:
            w.writerow([e["study"], e.get("doi") or "", e["title"], e["rec"], t.get(e["rec"], ""), e["relation"],
                        e["finding"], e["level"], e["design"], e["setting"], e["effect"], e["quote"], "", "", "", ""])
    low = sorted((e for e in ev if e.get("jev_support") is not None and e["jev_support"] < 0.5),
                 key=lambda e: e["jev_support"])
    with (REVIEW / "low_support_queue.csv").open("w", newline="") as f:   # prioritized queue, not an accuracy sample
        w = csv.writer(f)
        w.writerow(["study", "doi", "title", "rec", "rec_statement", "relation", "finding", "jev_support", "quote",
                    "reviewer_link_correct (y/n)", "reviewer_notes"])
        for e in low:
            w.writerow([e["study"], e.get("doi") or "", e["title"], e["rec"], t.get(e["rec"], ""), e["relation"],
                        e["finding"], e["jev_support"], e["quote"], "", ""])
    return path, len(sample)


def import_decisions():
    """Mark evidence links human-verified where a reviewer answered y to all three checks."""
    path = REVIEW / "evidence_spotcheck.csv"
    ok = set()
    with path.open() as f:
        for row in csv.DictReader(f):
            if all(row[k].strip().lower() == "y" for k in row if k.startswith("reviewer_") and k.endswith("(y/n)")):
                ok.add((row["study"], row["rec"]))
    store = ROOT / "data" / "human_verified.jsonl"
    have = {tuple(json.loads(l)) for l in store.read_text().splitlines()} if store.exists() else set()
    have |= ok
    store.write_text("".join(json.dumps(list(k)) + "\n" for k in sorted(have)))
    return len(ok)
