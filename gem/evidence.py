"""Screen literature records and extract evidence edges linking studies to taxonomy recommendations.

Routing after screening (title and abstract):
  empirical study that tests a practice  -> three-vendor panel extraction (with open-access full text when available)
  empirical study of a hazard only       -> single-model extraction
  recommendation-bearing commentary      -> single-model endorsement extraction (commentaries are never evidence)
  review / out of scope                  -> recorded, not extracted
Every edge carries a verbatim quote verified against the text the model saw.
"""
from __future__ import annotations

import json
import random
import re
import statistics
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
import yaml

from . import fetch, llm, quotes
from .llm import CFG, ROOT
from .taxonomy import TECH

TECH_KEYS = list(TECH)

SCREEN = ROOT / "data" / "screening.jsonl"
EXTRACT_DIR = ROOT / "data" / "extractions"
EVIDENCE = ROOT / "data" / "evidence.jsonl"
COMMENTARY_EDGES = ROOT / "data" / "commentary_endorsements.jsonl"
PROPOSALS = ROOT / "data" / "proposed_recommendations.jsonl"
TAXONOMY = ROOT / "data" / "taxonomy.yaml"
_lock = threading.Lock()

# ----------------------------------------------------------------------------- screening

SCREEN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["category", "technology", "setting", "tests_practice", "practice", "user", "reason"],
    "properties": {
        "category": {"type": "string", "enum": ["empirical_ai_health", "recommendation_commentary", "review",
                                                 "out_of_scope"]},
        "technology": {"type": "array", "items": {"type": "string", "enum": TECH_KEYS}},
        "setting": {"type": "string", "enum": ["real_deployment", "offline_or_simulated", "benchmark_only",
                                                "not_applicable"]},
        "tests_practice": {"type": "boolean"},
        "practice": {"type": "string"},
        "user": {"type": "string", "enum": ["patient", "clinician_or_staff", "both", "none"]},
        "reason": {"type": "string"},
    },
}

SCREEN_SYSTEM = """You screen records for an evidence map of guardrails and deployment practices for artificial intelligence in health care: large language models and other generative AI, multimodal and foundation models, predictive risk models, imaging and diagnostic software, ambient documentation, AI agents, and simulation or world-model tools. Classify the record from its title and abstract.

category:
- empirical_ai_health: original empirical study (any design, including benchmarks, simulations, surveys, qualitative, trials, deployments) in which an AI system is used or evaluated in a health care, clinical, patient-facing, or health-administration task, and the study bears on how the tool performs, is used, or is safeguarded in practice. Pure model-development papers that report only retrospective discrimination of a new model, with no deployment, validation-in-setting, safety, equity, workflow, or human-use question, are out_of_scope.
- recommendation_commentary: commentary, perspective, editorial, viewpoint, guideline, consensus statement, or framework paper whose main content is recommendations for how health care organizations should deploy, govern, or safeguard AI.
- review: systematic, scoping, or narrative review.
- out_of_scope: anything else.

technology: the kinds of AI involved.

setting (empirical only): real_deployment if the tool was used in actual care or operations with real patients, clinicians, or staff (including silent-mode deployments on live data); offline_or_simulated for vignettes, retrospective replays, simulated patients, reader studies, or retrospective external validation; benchmark_only for automated scoring on datasets or exams.

tests_practice: true only if the study compares outcomes with and without a deployment practice or safeguard that a health care organization chooses when it deploys an AI tool (for example clinician review of outputs, disclosure to patients, local validation or recalibration before go-live, silent-mode testing, alert threshold selection, drift monitoring, retrieval grounding added to an existing tool, escalation or crisis-detection rules, translation checks, reading-level constraints, staff training, prompt guardrails, output filters), or directly measures whether such a practice achieves its aim. A new model, agent architecture, fine-tuning method, or system the authors built and benchmarked is not a deployment practice unless the study evaluates it as a safeguard added to an existing tool. Evaluating model accuracy alone is false. practice: name that practice in a few words, or "" if none."""


def screen(records: list[dict], workers: int = 16):
    done = {json.loads(l)["id"] for l in SCREEN.read_text().splitlines()} if SCREEN.exists() else set()
    todo = [r for r in records if r["id"] not in done]
    m = CFG["models"]["screen"]

    def one(r):
        user = f"Title: {r['title']}\nPublication types: {', '.join(r.get('pub_types') or [])}\nAbstract: {r['abstract']}"
        try:
            out = llm.call(m["vendor"], m["model"], SCREEN_SYSTEM, user, SCREEN_SCHEMA, step="screen",
                           effort="low", max_tokens=2000)
        except llm.BudgetExceeded:
            raise
        except Exception as e:
            print("  screen fail", r["id"], repr(e)[:120], flush=True)
            return
        with _lock, SCREEN.open("a") as f:
            f.write(json.dumps({"id": r["id"], "model": m["model"], **out}) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    return len(todo)


def screen_audit(fraction: float | None = None, seed: int = 20260928):
    """Re-screen a random sample of exclusions with a second vendor; report disagreement."""
    fraction = fraction or CFG["screen_audit_fraction"]
    rows = [json.loads(l) for l in SCREEN.read_text().splitlines()]
    papers = fetch.load()
    excl = [r for r in rows if r["category"] in ("out_of_scope", "review")]
    random.Random(seed).shuffle(excl)
    sample = excl[:max(1, int(len(excl) * fraction))]
    m = CFG["models"]["screen_audit"]
    disagree = []
    for r in sample:
        p = papers[r["id"]]
        user = f"Title: {p['title']}\nPublication types: {', '.join(p.get('pub_types') or [])}\nAbstract: {p['abstract']}"
        out = llm.call(m["vendor"], m["model"], SCREEN_SYSTEM, user, SCREEN_SCHEMA, step="screen_audit",
                       effort="low", max_tokens=2000)
        if out["category"] in ("empirical_ai_health", "recommendation_commentary"):
            disagree.append({"id": r["id"], "primary": r["category"], "audit": out["category"]})
    res = {"sampled": len(sample), "disagreements": len(disagree), "items": disagree}
    (ROOT / "data" / "screen_audit.json").write_text(json.dumps(res, indent=1))
    return res


# ----------------------------------------------------------------------------- full text

def fulltext(p: dict, max_words: int = 14000) -> str:
    """Open-access full text (Europe PMC XML) for PMC articles; methods and results are kept first."""
    if not p.get("pmcid"):
        return ""
    try:
        r = requests.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{p['pmcid']}/fullTextXML", timeout=60)
        if r.status_code != 200:
            return ""
    except requests.RequestException:
        return ""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "lxml-xml")
    body = soup.find("body")
    if not body:
        return ""
    secs = []
    for sec in body.find_all("sec", recursive=False):
        title = (sec.find("title").get_text(" ", strip=True) if sec.find("title") else "").lower()
        rank = 0 if re.search(r"method|result|finding|design|setting|participant", title) else 1
        secs.append((rank, sec.get_text(" ", strip=True)))
    secs.sort(key=lambda x: x[0])
    words = " ".join(t for _, t in secs).split()
    return " ".join(words[:max_words])


# ----------------------------------------------------------------------------- extraction schema

DESIGNS = ["rct", "cluster_rct", "quasi_experimental", "prospective_with_comparator", "pre_post_no_comparator",
           "cross_sectional_or_descriptive", "qualitative", "retrospective_replay_with_raters", "vignette_simulation",
           "automated_benchmark", "other"]
POP_FLAGS = ["medicaid", "safety_net", "limited_english_or_non_english", "low_health_literacy", "low_income",
             "racial_ethnic_minority_focus", "rural", "older_adults", "disability"]
FINDINGS = ["supports_practice", "against_practice", "null_or_mixed", "hazard_present", "hazard_absent",
            "hazard_mixed"]


def study_schema() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["design", "setting", "user", "technology", "country", "sample_size", "sample_unit", "models_evaluated",
                     "population", "languages", "links", "unlisted_practices"],
        "properties": {
            "design": {"type": "string", "enum": DESIGNS},
            "setting": {"type": "string", "enum": ["real_deployment", "offline_or_simulated", "benchmark_only"]},
            "user": {"type": "string", "enum": ["patient", "clinician_or_staff", "both"]},
            "technology": {"type": "array", "items": {"type": "string", "enum": TECH_KEYS}},
            "country": {"type": "string"},
            "sample_size": {"type": ["integer", "null"]},
            "sample_unit": {"type": "string", "description": "patients, clinicians, messages, encounters, questions, ..."},
            "models_evaluated": {"type": "array", "items": {"type": "string"}},
            "population": {"type": "object", "additionalProperties": False, "required": POP_FLAGS,
                           "properties": {f: {"type": "boolean"} for f in POP_FLAGS}},
            "languages": {"type": "array", "items": {"type": "string"}},
            "links": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["rec_id", "relation", "finding", "outcome", "effect", "quote"],
                "properties": {
                    "rec_id": {"type": "string"},
                    "relation": {"type": "string", "enum": ["tests_guardrail", "documents_hazard"]},
                    "finding": {"type": "string", "enum": FINDINGS},
                    "outcome": {"type": "string", "description": "what was measured, <= 15 words"},
                    "effect": {"type": "string", "description": "the result with numbers as reported, <= 40 words"},
                    "quote": {"type": "string", "description": "verbatim sentence from the text supporting the result"},
                }}},
            "unlisted_practices": {"type": "array", "items": {"type": "string"},
                                   "description": "practices the study tests that match no listed recommendation"},
        },
    }


def _taxonomy_block() -> tuple[str, set[str]]:
    t = yaml.safe_load(TAXONOMY.read_text())
    lines = [f"{r['id']} [{r['domain']}] {r['statement']} (targets hazard: {r['hazard']})" for r in t["recommendations"]]
    return "\n".join(lines), {r["id"] for r in t["recommendations"]}


STUDY_SYSTEM = """You extract structured evidence from a study for an evidence map of guardrails for deploying artificial intelligence in health care (generative, predictive, imaging, multimodal, agentic, and other AI). The map links each study to the recommendations below.

Link the study to a recommendation only when one of these holds:
- tests_guardrail: the study compares outcomes with and without that practice, or directly measures whether the practice achieves its aim (for example, whether clinician review catches errors in AI drafts). finding is supports_practice, against_practice, or null_or_mixed.
- documents_hazard: the study measures the specific failure the recommendation targets (for example, reading level above patient literacy, errors in non-English output, missed emergencies, performance drop at a new site, alert burden) without testing the practice. finding is hazard_present, hazard_absent, or hazard_mixed.
Do not link a study just because it is about the same topic. Most accuracy studies document a hazard at most. Prefer fewer, correct links.

For each link, quote one verbatim sentence from the text that states the result (copy exactly; quotes that are not verbatim are discarded) and report the effect with the numbers as written. Population flags are true only if the study sample is described as that population (for example Medicaid enrollees, safety-net hospital patients, patients with limited English proficiency, non-English prompts). sample_size is the number of the sample_unit analyzed, or null if not reported. Use only information in the text.

Recommendations:
{taxonomy}"""


def _call_study(vendor, model, p, text, tax_block, step):
    user = f"Title: {p['title']}\nVenue: {p.get('venue')} ({p.get('date')}){' [preprint]' if p.get('preprint') else ''}\n\n<text>\n{text}\n</text>"
    return llm.call(vendor, model, STUDY_SYSTEM.replace("{taxonomy}", tax_block), user, study_schema(), step=step,
                    effort="high", max_tokens=32000)


def _verify_links(out: dict, text: str, ntext: str, valid: set[str]) -> list[dict]:
    kept = []
    for l in out["links"]:
        if l["rec_id"] not in valid:
            continue
        if (l["relation"] == "tests_guardrail") != (l["finding"] in FINDINGS[:3]):
            continue
        ok, score = quotes.verify(l["quote"], text, ntext)
        if ok:
            kept.append({**l, "quote_score": score})
    return kept


def _majority(vals):
    c = Counter(vals).most_common()
    if not c:
        return None, 0.0
    return c[0][0], c[0][1] / len(vals)


def aggregate(outs: dict[str, dict], links: dict[str, list[dict]]) -> dict:
    """Combine per-vendor extractions: majority vote for categorical fields, >= 2 of 3 for links."""
    vendors = list(outs)
    agg, agree = {}, {}
    for f in ("design", "setting", "user", "country", "sample_unit"):
        agg[f], agree[f] = _majority([outs[v][f] for v in vendors])
    ns = [outs[v]["sample_size"] for v in vendors if outs[v]["sample_size"] is not None]
    agg["sample_size"] = int(statistics.median(ns)) if ns else None
    agree["sample_size"] = (sum(1 for n in ns if agg["sample_size"] and abs(n - agg["sample_size"]) <= 0.05 * agg["sample_size"]) / len(vendors)) if ns else 1.0
    agg["population"] = {f: _majority([outs[v]["population"][f] for v in vendors])[0] for f in POP_FLAGS}
    agree["population"] = statistics.mean(_majority([outs[v]["population"][f] for v in vendors])[1] for f in POP_FLAGS)
    agg["languages"] = sorted({x for v in vendors for x in outs[v]["languages"]})
    agg["models_evaluated"] = sorted({x for v in vendors for x in outs[v]["models_evaluated"]})
    need = 2 if len(vendors) >= 3 else 1
    keyed = {}
    for v in vendors:
        for l in links[v]:
            keyed.setdefault((l["rec_id"], l["relation"]), []).append((v, l))
    agg_links, contested = [], []
    for (rid, rel), items in keyed.items():
        vs = {v for v, _ in items}
        if len(vs) < need:
            contested.append({"rec_id": rid, "relation": rel, "vendors": sorted(vs)})
            continue
        finding, share = _majority([l["finding"] for _, l in items])
        best = max((l for _, l in items if l["finding"] == finding), key=lambda l: l["quote_score"])
        agg_links.append({"rec_id": rid, "relation": rel, "finding": finding, "finding_agreement": round(share, 2),
                          "outcome": best["outcome"], "effect": best["effect"], "quote": best["quote"],
                          "linked_by": sorted(vs)})
    agg["links"] = agg_links
    agg["contested_links"] = contested
    agg["agreement"] = {k: round(v, 2) for k, v in agree.items()}
    agg["unlisted_practices"] = sorted({x for v in vendors for x in outs[v]["unlisted_practices"]})
    return agg


def extract_studies(workers: int = 6, limit: int | None = None):
    """Extract every screened-in record not yet extracted."""
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    tax_block, valid = _taxonomy_block()
    papers = fetch.load()
    rows = [json.loads(l) for l in SCREEN.read_text().splitlines()]
    todo = [r for r in rows if r["category"] in ("empirical_ai_health", "recommendation_commentary")
            and not (EXTRACT_DIR / f"{_fn(r['id'])}.json").exists()]
    todo.sort(key=lambda r: (not r["tests_practice"], r["setting"] != "real_deployment"))
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} records to extract ({sum(r['tests_practice'] for r in todo)} practice-testing)")

    def one(r):
        p = papers[r["id"]]
        try:
            if r["category"] == "recommendation_commentary":
                res = _commentary(p, tax_block, valid)
            elif r["tests_practice"] and r["setting"] == "real_deployment":
                res = _panel(p, tax_block, valid, CFG["models"]["panel"], "panel")
            elif r["tests_practice"] and r["setting"] == "offline_or_simulated":
                res = _panel(p, tax_block, valid, CFG["models"]["fast_panel"], "fast_panel")
            else:
                res = _single(p, tax_block, valid)
        except llm.BudgetExceeded:
            raise
        except Exception as e:
            print("  extract fail", r["id"], repr(e)[:160], flush=True)
            return
        res.update({"id": r["id"], "screen": r})
        path = EXTRACT_DIR / f"{_fn(r['id'])}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(path)
        print(f"  {r['id']} {res['route']}: {len(res.get('links', []))} links; run ${llm.run_spend():.2f}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))


def _fn(pid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", pid)


def _panel(p, tax_block, valid, members, route):
    ft = fulltext(p)
    text = f"Abstract: {p['abstract']}" + (f"\n\nFull text (methods and results first):\n{ft}" if ft else "")
    ntext = quotes.norm(text)
    outs, links = {}, {}
    for vendor, model in members.items():
        try:
            outs[vendor] = _call_study(vendor, model, p, text, tax_block, f"study_{route}")
            links[vendor] = _verify_links(outs[vendor], text, ntext, valid)
        except llm.BudgetExceeded:
            raise
        except Exception as e:
            print("   panel member fail", vendor, repr(e)[:120], flush=True)
    if not outs:
        raise RuntimeError("all panel members failed")
    agg = aggregate(outs, links)
    return {"route": route, "fulltext": bool(ft), "raw": outs, **agg, "models": {v: members[v] for v in outs}}


def _single(p, tax_block, valid):
    m = CFG["models"]["hazard"]
    text = f"Abstract: {p['abstract']}"
    out = _call_study(m["vendor"], m["model"], p, text, tax_block, "study_hazard")
    links = _verify_links(out, text, quotes.norm(text), valid)
    out["links"] = [{**l, "linked_by": [m["vendor"]], "finding_agreement": None} for l in links]
    out["agreement"] = None
    out["contested_links"] = []
    return {"route": "single", "fulltext": False, "models": {m["vendor"]: m["model"]}, **out}


COMMENT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["endorsements", "unlisted_recommendations"],
    "properties": {
        "endorsements": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["rec_id", "quote"],
            "properties": {"rec_id": {"type": "string"}, "quote": {"type": "string"}}}},
        "unlisted_recommendations": {"type": "array", "items": {"type": "string"}},
    },
}

COMMENT_SYSTEM = """You map a commentary or framework paper on health care AI to the recommendations it endorses, for an evidence map of guardrails for deploying AI in health care. List a recommendation only if the text explicitly recommends that practice; quote the verbatim sentence (copy exactly). List recommendations the text makes that match nothing below as unlisted_recommendations, each as one atomic imperative statement.

Recommendations:
{taxonomy}"""


def _commentary(p, tax_block, valid):
    m = CFG["models"]["hazard"]
    text = f"Title: {p['title']}\nAbstract: {p['abstract']}"
    out = llm.call(m["vendor"], m["model"], COMMENT_SYSTEM.replace("{taxonomy}", tax_block), text, COMMENT_SCHEMA,
                   step="commentary", effort="medium", max_tokens=8000)
    nt = quotes.norm(text)
    ends = [e for e in out["endorsements"] if e["rec_id"] in valid and quotes.verify(e["quote"], text, nt)[0]]
    return {"route": "commentary", "models": {m["vendor"]: m["model"]}, "endorsements": ends,
            "unlisted_practices": out["unlisted_recommendations"], "links": []}


# ----------------------------------------------------------------------------- assemble

LEVEL_LABEL = {1: "Randomized trial in deployment", 2: "Controlled comparison in deployment",
               3: "Deployment data without comparator", 4: "Simulation, vignette, or rater study",
               5: "Automated benchmark"}


def level(design: str, setting: str) -> int:
    if setting == "benchmark_only" or design == "automated_benchmark":
        return 5
    if setting != "real_deployment":
        return 4
    if design in ("rct", "cluster_rct"):
        return 1
    if design in ("quasi_experimental", "prospective_with_comparator"):
        return 2
    return 3


def assemble():
    """Write evidence.jsonl and commentary_endorsements.jsonl from all extraction files."""
    papers = fetch.load()
    hv_path = ROOT / "data" / "human_verified.jsonl"
    verified = {tuple(json.loads(l)) for l in hv_path.read_text().splitlines()} if hv_path.exists() else set()
    ev, com, prop = [], [], []
    for f in sorted(EXTRACT_DIR.glob("*.json")):
        x = json.loads(f.read_text())
        p = papers.get(x["id"], {})
        if p.get("id_verified") is False:   # bibliographic record did not match PubMed or doi.org
            continue
        meta = {"study": x["id"], "title": p.get("title"), "venue": p.get("venue"), "date": p.get("date"),
                "doi": p.get("doi"), "pmid": p.get("pmid"), "preprint": p.get("preprint"),
                "citations": p.get("citations")}
        if x["route"] == "commentary":
            for e in x["endorsements"]:
                com.append({**meta, "rec": e["rec_id"], "quote": e["quote"]})
        else:
            lv = level(x["design"], x["setting"])
            for l in x["links"]:
                ev.append({**meta, "rec": l["rec_id"], "relation": l["relation"], "finding": l["finding"],
                           "outcome": l["outcome"], "effect": l["effect"], "quote": l["quote"],
                           "design": x["design"], "setting": x["setting"], "user": x["user"], "level": lv,
                           "technology": x.get("technology", []),
                           "sample_size": x.get("sample_size"), "sample_unit": x.get("sample_unit"),
                           "country": x.get("country"), "population": x["population"],
                           "languages": x.get("languages", []), "route": x["route"],
                           "linked_by": l.get("linked_by"), "agreement": x.get("agreement"),
                           "human_verified": (x["id"], l["rec_id"]) in verified})
        for u in x.get("unlisted_practices", []):
            prop.append({"from": x["id"], "route": x["route"], "statement": u})
    EVIDENCE.write_text("".join(json.dumps(e) + "\n" for e in ev))
    COMMENTARY_EDGES.write_text("".join(json.dumps(e) + "\n" for e in com))
    PROPOSALS.write_text("".join(json.dumps(e) + "\n" for e in prop))
    return len(ev), len(com), len(prop)
