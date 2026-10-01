"""Command-line entry points.

  python -m gem.pipeline seed-taxonomy        # fetch sources, three-vendor extraction, merge, critique, revise
  python -m gem.pipeline backfill --since 2023-01-01
  python -m gem.pipeline weekly               # scheduled update (GitHub Actions)
  python -m gem.pipeline build                # rebuild docs/ from data/
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

from . import build_site, editorials, evidence, fetch, governance, jev, llm, taxonomy, technical
from .llm import ROOT

STATE = ROOT / "state" / "last_run.json"
HEARTBEAT = ROOT / "state" / "heartbeat.json"
CHANGELOG = ROOT / "data" / "changelog.jsonl"


def _statuses() -> dict:
    p = ROOT / "docs" / "data" / "map.json"
    if not p.exists():
        return {}
    return {r["id"]: (r["status_label"], r["statement"], r["n_tests"], r["n_hazard"], r["n_endorse"])
            for r in json.loads(p.read_text())["recs"]}


def _process(records: list[dict], label: str) -> dict:
    before = _statuses()
    n_screen = evidence.screen(records)
    print(jev.screen())
    evidence.extract_studies()
    evidence.assemble()
    jev.verify_links()
    build_site.build()
    after = _statuses()
    changes = [{"id": k, "statement": v[1], "before": before[k][0], "after": v[0]}
               for k, v in after.items() if k in before and before[k][0] != v[0]]
    d = lambda i: sum(max(0, after[k][i] - before.get(k, (None, None, 0, 0, 0))[i]) for k in after)
    entry = {"date": dt.date.today().isoformat(), "label": label, "new_records": n_screen,
             "new_tests": d(2), "new_hazard": d(3), "new_endorsements": d(4), "status_changes": changes,
             "spend_usd": round(llm.run_spend(), 2)}
    return entry


def seed_taxonomy():
    print(governance.fetch_all())
    taxonomy.extract()
    taxonomy.canonicalize()
    taxonomy.write_taxonomy()


def backfill(since: str, until: str | None):
    new = fetch.update(since, until)
    print(f"{len(new)} new records")
    entry = _process(list(fetch.load().values()), "backfill")
    evidence.screen_audit()
    build_site.build()
    _log(entry)


def weekly():
    last = json.loads(STATE.read_text())["date"] if STATE.exists() else (dt.date.today() - dt.timedelta(days=10)).isoformat()
    since = (dt.date.fromisoformat(last) - dt.timedelta(days=3)).isoformat()   # overlap covers indexing lag
    gov = governance.fetch_all()
    changed = sorted(k for k, v in gov.items() if v.get("changed"))
    if changed:   # new candidates are proposals; they enter taxonomy.yaml only through human curation
        taxonomy.extract(set(changed))
    new = fetch.update(since)
    ed = editorials.sweep(since)                       # editorials often lack abstracts; pull them by venue
    new += [p for p in fetch.load().values() if p.get("source") == "openalex_venue_sweep" and p.get("fetched") == dt.date.today().isoformat()]
    tech_before = sum(1 for _ in open(technical.TECH_EVIDENCE)) if technical.TECH_EVIDENCE.exists() else 0
    entry = _process(new, "weekly")
    technical.classify()          # new practices are recorded as candidates; the decision list changes only by curation
    technical.link()
    entry["new_technical_links"] = technical.assemble() - tech_before
    entry["new_editorials"] = ed["new_records"]
    editorials.coverage()
    build_site.build()
    entry["sources_changed"] = changed
    if dt.date.today().day <= 7:   # monthly screening audit
        evidence.screen_audit()
        build_site.build()
    _log(entry)
    STATE.write_text(json.dumps({"date": dt.date.today().isoformat()}))


def _log(entry):
    with CHANGELOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    build_site.build()
    HEARTBEAT.write_text(json.dumps({"last_run": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                     "spend_usd": entry.get("spend_usd")}))
    print(json.dumps(entry, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed-taxonomy")
    b = sub.add_parser("backfill")
    b.add_argument("--since", default="2023-01-01")
    b.add_argument("--until")
    sub.add_parser("weekly")
    sub.add_parser("build")
    a = ap.parse_args()
    try:
        if a.cmd == "seed-taxonomy":
            seed_taxonomy()
        elif a.cmd == "backfill":
            backfill(a.since, a.until)
        elif a.cmd == "weekly":
            weekly()
        else:
            build_site.build()
    except llm.BudgetExceeded as e:
        # Stop cleanly: completed items are already on disk, and the next run resumes where this one stopped.
        print(f"stopped at budget cap: {e}")
        evidence.assemble()
        jev.verify_links()
        build_site.build()


if __name__ == "__main__":
    main()
