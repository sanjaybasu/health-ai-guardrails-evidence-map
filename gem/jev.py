"""Jev (TypeSafe System One) calibrated judgments: a second screener and a link-support verifier.

Jev returns a calibrated probability for yes/no ("noul") questions. Inputs here are published abstracts and quotes only.
  screen(): P(tests a deployment practice), P(real deployment), P(empirical health AI study) per record; records where
            Jev and the primary screener disagree, or Jev is uncertain, are re-screened by a third model (arbitration).
  verify_links(): P(the verbatim quote reports a result bearing on the linked recommendation) per evidence edge.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from . import fetch, llm
from .llm import CFG, ROOT

URL = "https://api.typesafe.ai/v1/systemone"
JEV_SCREEN = ROOT / "data" / "jev_screen.jsonl"
PRICE_IN = 0.042   # USD per million input tokens; output is free (docs, 2026-09-18)
_lock = threading.Lock()


def ask(state, questions: dict, step: str) -> dict:
    key = llm._key("TYPESAFE_API_KEY")
    body = {"state": state, "model": "jev-latest", "questions": questions}
    for i in range(5):
        try:
            r = requests.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=60)
            if r.status_code == 200:
                d = r.json()
                tin = d.get("usage", {}).get("input_tokens", 0)
                with _lock, llm.LEDGER.open("a") as f:
                    f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step, "vendor": "jev",
                                        "model": d.get("model"), "in": tin, "out": 0,
                                        "usd": round(tin * PRICE_IN / 1e6, 7), "secs": 0}) + "\n")
                return {k: v.get("noul") for k, v in d["answers"].items()}
            if r.status_code in (400, 401, 403):
                raise RuntimeError(f"jev {r.status_code}: {r.text[:200]}")
        except requests.RequestException:
            pass
        time.sleep(2 * 2 ** i)
    raise RuntimeError("jev failed after retries")


SCREEN_Q = {
    "empirical": {"type": "noul", "instructions":
        "Is this an original empirical study in which an AI system (a language model, predictive model, imaging or "
        "diagnostic software, multimodal model, ambient scribe, or agent) is used or evaluated in a health care task, "
        "bearing on how the tool performs, is used, or is safeguarded in practice?"},
    "tests_practice": {"type": "noul", "instructions":
        "Does the study compare outcomes with and without a deployment practice or safeguard that a health care "
        "organization chooses when deploying an AI tool (for example clinician review of outputs, disclosure to "
        "patients, local validation or recalibration, silent-mode testing, alert thresholds, drift monitoring, "
        "retrieval grounding added to an existing tool, escalation rules, translation checks, reading-level "
        "constraints, staff training, or output filters), or directly measure whether such a practice achieves its "
        "aim? A new model or architecture benchmarked by its authors does not count."},
    "real_deployment": {"type": "noul", "instructions":
        "Was the AI tool used in actual care or operations with real patients, clinicians, or staff (including "
        "silent-mode deployment on live data), rather than on vignettes, retrospective data, or benchmarks?"},
}


def screen(workers: int = 16):
    """Jev second screen over every primary-screened record; then arbitrate disagreements with a third model."""
    from .evidence import SCREEN, SCREEN_SCHEMA, SCREEN_SYSTEM
    done = {json.loads(l)["id"] for l in JEV_SCREEN.read_text().splitlines()} if JEV_SCREEN.exists() else set()
    papers = fetch.load()
    rows = [json.loads(l) for l in SCREEN.read_text().splitlines()]
    todo = [r for r in rows if r["id"] not in done and r["id"] in papers]

    def one(r):
        p = papers[r["id"]]
        try:
            a = ask({"title": p["title"], "abstract": p["abstract"]}, SCREEN_Q, "jev_screen")
        except RuntimeError as e:
            print("  jev fail", r["id"], str(e)[:100], flush=True)
            return
        with _lock, JEV_SCREEN.open("a") as f:
            f.write(json.dumps({"id": r["id"], **a}) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))

    # Arbitration: disagreement on tests_practice, or Jev uncertain (0.2-0.8) on a record the primary screen kept.
    jev = {json.loads(l)["id"]: json.loads(l) for l in JEV_SCREEN.read_text().splitlines()}
    m = CFG["models"]["screen_audit"]
    changed = 0
    for r in rows:
        j = jev.get(r["id"])
        if not j or r.get("arbitrated") or j.get("tests_practice") is None:
            continue
        p_t = j["tests_practice"]
        prim = r["tests_practice"] and r["category"] == "empirical_ai_health"
        if (prim != (p_t >= 0.5)) or (0.2 <= p_t <= 0.8 and r["category"] == "empirical_ai_health"):
            p = papers[r["id"]]
            user = f"Title: {p['title']}\nPublication types: {', '.join(p.get('pub_types') or [])}\nAbstract: {p['abstract']}"
            out = llm.call(m["vendor"], m["model"], SCREEN_SYSTEM, user, SCREEN_SCHEMA, step="screen_arbitrate",
                           effort="medium", max_tokens=3000)
            r.update({"primary": {k: r[k] for k in ("category", "setting", "tests_practice", "practice")},
                      **{k: out[k] for k in ("category", "technology", "setting", "tests_practice", "practice")},
                      "arbitrated": True, "arbiter": m["model"]})
            changed += 1
        r["jev"] = j
    tmp = SCREEN.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(SCREEN)
    return {"jev_screened": len(jev), "arbitrated": changed}


def verify_links(workers: int = 16):
    """Add jev_support (probability the quote reports a result bearing on the recommendation) to evidence edges."""
    import yaml
    from .evidence import EVIDENCE, TAXONOMY
    recs = {r["id"]: r for r in yaml.safe_load(TAXONOMY.read_text())["recommendations"]}
    ev = [json.loads(l) for l in EVIDENCE.read_text().splitlines()]
    cache_p = ROOT / "data" / "jev_links.jsonl"
    cache = {}
    if cache_p.exists():
        for l in cache_p.read_text().splitlines():
            x = json.loads(l)
            cache[(x["study"], x["rec"], x["quote"])] = x["p"]

    def one(e):
        k = (e["study"], e["rec"], e["quote"])
        if k in cache:
            e["jev_support"] = cache[k]
            return
        r = recs.get(e["rec"], {})
        target = ("whether the practice achieves its aim" if e["relation"] == "tests_guardrail"
                  else f"the hazard the practice targets ({r.get('hazard', '')})")
        q = {"support": {"type": "noul", "instructions":
             f"Does the quoted sentence report a study result about {target}, for the practice: "
             f"\"{r.get('statement', '')}\"? Answer no if the quote is background, a method description, or about "
             f"a different practice."}}
        try:
            e["jev_support"] = ask({"quote": e["quote"], "reported_effect": e["effect"]}, q, "jev_verify")["support"]
            with _lock, cache_p.open("a") as f:
                f.write(json.dumps({"study": e["study"], "rec": e["rec"], "quote": e["quote"], "p": e["jev_support"]}) + "\n")
        except RuntimeError:
            e["jev_support"] = None

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, ev))
    EVIDENCE.write_text("".join(json.dumps(e) + "\n" for e in ev))
    return sum(1 for e in ev if (e.get("jev_support") or 0) < 0.5)
