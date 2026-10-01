"""Pre-merge check for the weekly update: fail if the rebuilt site would not render or lost data.

Usage: python -m gem.validate_site [baseline_map.json]
The baseline is the map.json currently published on main; without it only structural checks run.
"""
import json
import re
import sys
from pathlib import Path

DOCS = Path("docs")
PAGES = ["index.html", "methods.html", "technical.html", "changelog.html"]
ASSETS = ["app.js", "technical.js", "style.css"]
REC_KEYS = {"id", "statement", "domain", "status", "status_label", "by_tech"}


def check(baseline=None):
    errors = []
    for name in PAGES + ASSETS:
        p = DOCS / name
        if not p.exists() or p.stat().st_size < 200:
            errors.append(f"missing or empty {p}")
    for name in PAGES:
        p = DOCS / name
        if p.exists() and not re.search(r"</html>\s*$", p.read_text(), re.I):
            errors.append(f"{p} is truncated (no closing </html>)")

    try:
        m = json.loads((DOCS / "data/map.json").read_text())
        t = json.loads((DOCS / "data/technical.json").read_text())
    except Exception as e:  # unparseable data means a blank page
        return errors + [f"site data does not parse: {e}"]

    recs = m.get("recs") or []
    if not recs:
        errors.append("map.json has no recommendations")
    labels = m.get("status_labels", {})
    for r in recs:
        missing = REC_KEYS - set(r)
        if missing:
            errors.append(f"{r.get('id')} missing fields {sorted(missing)}")
            continue
        if r["status"] not in labels:
            errors.append(f"{r['id']} has unknown status {r['status']}")
        f = DOCS / "data/rec" / f"{r['id']}.json"
        try:
            json.loads(f.read_text())
        except Exception as e:
            errors.append(f"{f} unreadable: {e}")
    if len({r.get("id") for r in recs}) != len(recs):
        errors.append("duplicate recommendation ids")
    if not t.get("controls") or not t.get("families"):
        errors.append("technical.json has no controls or families")

    if baseline:
        b = json.loads(Path(baseline).read_text())
        old, new = b.get("stats", {}), m.get("stats", {})
        if len(recs) < len(b.get("recs", [])):
            errors.append(f"recommendations dropped {len(b['recs'])} -> {len(recs)}")
        for k in ("n_papers", "n_tests_edges", "n_evidence_edges"):
            if new.get(k, 0) < 0.98 * old.get(k, 0):
                errors.append(f"{k} fell {old.get(k)} -> {new.get(k)}")
    return errors


if __name__ == "__main__":
    errs = check(sys.argv[1] if len(sys.argv) > 1 else None)
    for e in errs[:50]:
        print("FAIL:", e)
    print(f"{len(errs)} problems")
    sys.exit(1 if errs else 0)
