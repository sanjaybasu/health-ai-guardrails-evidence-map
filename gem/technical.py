"""Technical controls: implementation choices a health system makes when deploying AI, framed as decisions with a
comparator ("add a keyword filter on LLM output, or not"; "two-vendor cross-verification or a single model").

Built bottom-up from the practices that studies tested and that commentaries recommended, which the main extraction
recorded as unlisted practices. Linked to studies with the same rules as the main map: every vendor sees the same
candidate list, a link needs two of three vendors (or cross-examination), and every quote must be verbatim.
"""
from __future__ import annotations

import glob
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

from . import fetch, llm, quotes
from .llm import CFG, ROOT

CAND = ROOT / "data" / "technical_candidates.jsonl"
DRAFT = ROOT / "data" / "technical_draft.json"
TECHNICAL = ROOT / "data" / "technical.yaml"
LINKS_DIR = ROOT / "data" / "technical_extractions"
TECH_EVIDENCE = ROOT / "data" / "technical_evidence.jsonl"
_lock = threading.Lock()

FAMILIES = {
    "output_filtering": "keyword, rule, blocklist, or classifier filters applied to AI inputs or outputs",
    "grounding_retrieval": "retrieval-augmented generation, source citation, knowledge graphs, EHR grounding",
    "verification_ensembles": "LLM-as-judge, self-consistency, second-model or second-vendor verification, ensembles, multi-agent debate",
    "uncertainty_deferral": "confidence thresholds, abstention, calibrated deferral to a human, conformal prediction",
    "prompting": "role, few-shot, chain-of-thought, structured, or optimized prompts and system instructions",
    "model_choice_adaptation": "fine-tuning, domain-specific versus general models, model size, open versus closed, local deployment",
    "language_literacy_handling": "native-language processing, translation or back-translation, simplification, reading-level control",
    "human_workflow_design": "draft versus autonomous modes, review designs, routing to humans, double review, workload timing",
    "alert_design": "alert thresholds, tiering, throttling, timing, and display of alerts",
    "input_data_handling": "input formatting, structured data, missing-data handling, multimodal inputs, context length",
    "monitoring_methods": "drift detection, automated evaluation, audit sampling, silent-mode runs, performance dashboards",
    "agent_tool_design": "tool use, agent architectures, action approval gates, sandboxing, memory",
    "presentation_explanation": "explanations, confidence or uncertainty display, shown citations, interface design",
    "privacy_security_engineering": "PHI redaction and de-identification, on-premises models, prompt-injection defenses, access limits",
}
TECHS = ["generative_llm", "multimodal_foundation", "predictive_risk", "imaging_diagnostic", "agentic", "speech_ambient",
         "world_model_simulation", "any_ai"]

# ----------------------------------------------------------------------------- 1. classify candidates

CLS_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["technical", "family"],
              "properties": {"technical": {"type": "boolean"},
                             "family": {"type": "string", "enum": list(FAMILIES) + ["none"]}}}
CLS_SYSTEM = ("You sort proposed practices for an evidence map of health care AI deployment. technical is true when the "
              "practice is an implementation choice an engineering or clinical-informatics team makes when building, "
              "configuring, or operating an AI tool (for example a keyword filter on outputs, retrieval grounding, a "
              "second-model verifier, an abstention threshold, a prompting strategy, fine-tuning, a translation step, an "
              "alert threshold, a review workflow design). It is false for governance, policy, procurement, training "
              "programs, research-reporting advice, or statistical analysis methods that do not change the deployed "
              "tool. family is the closest family, or none when technical is false.\n\nFamilies:\n"
              + "\n".join(f"{k}: {v}" for k, v in FAMILIES.items()))


def collect() -> list[dict]:
    out = []
    for f in glob.glob(str(ROOT / "data" / "extractions" / "*.json")):
        x = json.loads(open(f).read())
        for i, u in enumerate(x.get("unlisted_practices", []) or []):
            if isinstance(u, str) and 8 <= len(u) <= 400:
                out.append({"cid": f"{x['id']}#{i}", "from": x["id"], "route": x["route"], "statement": u.strip()})
    return out


def classify(workers: int = 24):
    done = {json.loads(l)["cid"] for l in CAND.read_text().splitlines()} if CAND.exists() else set()
    todo = [c for c in collect() if c["cid"] not in done]
    m = CFG["models"]["screen"]

    def one(c):
        try:
            out = llm.call(m["vendor"], m["model"], CLS_SYSTEM, c["statement"], CLS_SCHEMA, step="tech_classify",
                           effort="low", max_tokens=1000)
        except llm.BudgetExceeded:
            raise
        except Exception:
            return
        with _lock, CAND.open("a") as f:
            f.write(json.dumps({**c, **out}) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    rows = [json.loads(l) for l in CAND.read_text().splitlines()]
    return Counter(r["family"] for r in rows if r["technical"])


# ----------------------------------------------------------------------------- 2. canonicalize into decisions

CTRL_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["controls"],
    "properties": {"controls": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["key", "question", "control", "comparator", "technology", "aim", "variants", "members"],
        "properties": {
            "key": {"type": "string"},
            "question": {"type": "string", "description": "the decision as a question a Chief Health AI Officer would ask"},
            "control": {"type": "string", "description": "the practice, imperative, <= 25 words"},
            "comparator": {"type": "string", "description": "what it is compared against, <= 15 words"},
            "technology": {"type": "array", "items": {"type": "string", "enum": TECHS}},
            "aim": {"type": "string", "description": "what the control is meant to improve or prevent, <= 20 words"},
            "variants": {"type": "array", "items": {"type": "string"}},
            "members": {"type": "array", "items": {"type": "string"}, "description": "cluster ids"},
        }}}},
}

CTRL_SYSTEM = """You are building the technical-controls layer of an evidence map for health system leaders deploying AI. Each item is one implementation decision, stated as the question a Chief Health AI Officer would ask ("Should we add a keyword filter on top of the LLM's patient messages?", "Should we pay for a second vendor's model to cross-check the first?"), with the practice and the comparator it is weighed against. You receive practices that published studies tested or that commentaries recommended, pre-grouped into tight clusters, within one family.

Merge clusters that are the same decision and keep implementation details as variants. Keep two decisions separate only when a study would compare them separately. Return no more than 15 decisions for the family, ordered from most to least often represented. Do not create decisions no cluster supports. Assign every cluster id to one decision's members, except clusters that are too vague or not a deployable choice, which you leave unassigned. Use plain words and name the comparator explicitly."""

CRIT_SYSTEM = """You review the technical-controls list for one family of an evidence map for health system AI leaders. Report only real problems: decisions that bundle two separate choices, duplicates, decisions with no clear comparator, questions a Chief Health AI Officer would not recognize, or wording that goes beyond the member practices. For each give the keys and a concrete fix. Do not add decisions the members do not support."""
CRIT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["issues"],
               "properties": {"issues": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                              "required": ["keys", "problem", "fix"],
                              "properties": {"keys": {"type": "array", "items": {"type": "string"}},
                                             "problem": {"type": "string"}, "fix": {"type": "string"}}}}}}
REV_SYSTEM = """You maintain the technical-controls list for one family. Two reviewers from different model vendors critiqued the draft. Apply critiques you agree with, reject the rest, and return the full revised list in the same schema, keeping member cluster ids with the decision they belong to and at most 15 decisions."""


def canonicalize():
    from scipy.cluster.hierarchy import leaves_list, linkage
    from .taxonomy import _cluster, _embed
    rows = [json.loads(l) for l in CAND.read_text().splitlines()]
    by = defaultdict(list)
    for r in rows:
        if r["technical"] and r["family"] in FAMILIES:
            by[r["family"]].append(r)
    draft = json.loads(DRAFT.read_text()) if DRAFT.exists() else {}
    m = CFG["models"]["merge"]
    for fam, cs in sorted(by.items(), key=lambda kv: len(kv[1])):
        if fam in draft and draft[fam].get("revised"):
            continue
        v = _embed([c["statement"] for c in cs])
        groups = _cluster(v, 0.82)
        cents = np.array([v[g].mean(0) for g in groups])
        order = leaves_list(linkage(cents, "average", "cosine")) if len(groups) > 2 else range(len(groups))
        clusters, keys = {}, []
        for gi, idx in enumerate(order):
            g = groups[idx]
            med = g[int(np.argmax(v[g] @ cents[idx]))]
            k = f"k{gi}"
            keys.append(k)
            clusters[k] = {"members": [cs[i]["cid"] for i in g], "statement": cs[med]["statement"],
                           "n_studies": len({cs[i]["from"] for i in g if cs[i]["route"] != "commentary"}),
                           "n_commentaries": len({cs[i]["from"] for i in g if cs[i]["route"] == "commentary"})}
        # keep the ~400 best-supported clusters per family; singletons from one paper add little
        keys = sorted(keys, key=lambda k: -(clusters[k]["n_studies"] * 2 + clusters[k]["n_commentaries"]))[:400]
        out = []
        for bi in range(0, len(keys), 200):
            batch = keys[bi:bi + 200]
            listing = "\n".join(f"[{k}] studies={clusters[k]['n_studies']} commentaries={clusters[k]['n_commentaries']}: "
                                f"{clusters[k]['statement']}" for k in batch)
            r = llm.call(m["vendor"], m["model"], CTRL_SYSTEM, f"Family: {fam} ({FAMILIES[fam]}).\n\n{listing}",
                         CTRL_SCHEMA, step=f"tech_canon:{fam}", effort="high", max_tokens=64000)["controls"]
            for x in r:
                x["key"] = f"{x['key']}__b{bi // 200}"
            out += r
        if len(keys) > 200:
            listing = "\n".join(f"[{x['key']}] {x['question']} | control: {x['control']} | vs: {x['comparator']}" for x in out)
            merged = llm.call(m["vendor"], m["model"], CTRL_SYSTEM.replace("pre-grouped into tight clusters",
                              "already drafted in separate batches (members are draft keys)"),
                              f"Family: {fam}.\n\n{listing}", CTRL_SCHEMA, step=f"tech_merge:{fam}", effort="high",
                              max_tokens=48000)["controls"]
            byk = {x["key"]: x for x in out}
            for x in merged:
                x["members"] = [c for k in x["members"] if k in byk for c in byk[k]["members"]]
            out = merged
        shown = json.dumps([{k: x[k] for k in ("key", "question", "control", "comparator", "aim")} for x in out], indent=0)
        crit = {v_: llm.call(v_, CFG["models"]["panel"][v_], CRIT_SYSTEM, f"Family: {fam}.\n\n{shown}", CRIT_SCHEMA,
                             step=f"tech_critique:{fam}", effort="high", max_tokens=16000)["issues"]
                for v_ in ("openai", "gemini")}
        revised = llm.call(m["vendor"], m["model"], REV_SYSTEM,
                           f"Family: {fam}.\n\nDraft:\n{json.dumps(out, indent=0)}\n\nReviewer A:\n{json.dumps(crit['openai'])}"
                           f"\n\nReviewer B:\n{json.dumps(crit['gemini'])}", CTRL_SCHEMA, step=f"tech_revise:{fam}",
                           effort="high", max_tokens=64000)["controls"]
        expand = lambda xs: [{**x, "clusters": x["members"],
                              "members": [c for k in x["members"] if k in clusters for c in clusters[k]["members"]]}
                             for x in xs]
        draft[fam] = {"n_candidates": len(cs), "n_clusters": len(clusters), "final": expand(revised), "critiques": crit,
                      "revised": True}
        DRAFT.write_text(json.dumps(draft, indent=1))
        print(f"  {fam}: {len(cs)} practices, {len(clusters)} clusters -> {len(revised)} decisions; run ${llm.run_spend():.2f}",
              flush=True)


def write_controls():
    draft = json.loads(DRAFT.read_text())
    old = {r["key"]: r for r in yaml.safe_load(TECHNICAL.read_text())["controls"]} if TECHNICAL.exists() else {}
    nxt = max([int(r["id"][1:]) for r in old.values()] or [0]) + 1
    rows = {json.loads(l)["cid"]: json.loads(l) for l in CAND.read_text().splitlines()}
    out = []
    for fam in FAMILIES:
        for x in draft.get(fam, {}).get("final", []):
            prev = old.get(x["key"])
            cid = prev["id"] if prev else f"T{nxt:03d}"
            if not prev:
                nxt += 1
            mem = [rows[c] for c in x["members"] if c in rows]
            out.append({"id": cid, "key": x["key"], "family": fam, "question": x["question"], "control": x["control"],
                        "comparator": x["comparator"], "technology": x["technology"], "aim": x["aim"],
                        "variants": x.get("variants", [])[:8],
                        "proposed_by_studies": len({r["from"] for r in mem if r["route"] != "commentary"}),
                        "proposed_by_commentaries": len({r["from"] for r in mem if r["route"] == "commentary"}),
                        "status": prev.get("status", "machine_draft") if prev else "machine_draft"})
    TECHNICAL.write_text(yaml.safe_dump({"note": "Machine draft built from practices that studies tested and commentaries "
                                                 "recommended; wording is curated by a person before it is final.",
                                         "families": FAMILIES, "controls": out}, sort_keys=False, width=110))
    print(f"technical controls: {len(out)}")


# ----------------------------------------------------------------------------- 3. link studies to controls

TFINDINGS = ["favours_control", "favours_comparator", "no_clear_difference", "mixed"]
LINK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["links"],
    "properties": {"links": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["control_id", "comparator_used", "finding", "outcome", "effect", "tradeoff", "quote"],
        "properties": {
            "control_id": {"type": "string"},
            "comparator_used": {"type": "string", "description": "what the study compared the control against"},
            "finding": {"type": "string", "enum": TFINDINGS},
            "outcome": {"type": "string"}, "effect": {"type": "string", "description": "numbers as reported"},
            "tradeoff": {"type": "string", "description": "any cost, latency, new failure mode, or burden reported; '' if none"},
            "quote": {"type": "string", "description": "verbatim sentence stating the comparison result"},
        }}}},
}
LINK_SYSTEM = """You link a study to the technical controls it compared, for an evidence map for health system AI leaders. Link a control only when the study directly compares outcomes with the control against an alternative (the listed comparator or another stated alternative), or measures the control's effect. Topic similarity is not enough. For each link give the comparator the study actually used, which arm the result favours, the outcome, the effect with numbers as written, any tradeoff reported (cost, latency, new errors such as more omissions, workload), and one verbatim sentence from the text stating the result. Quotes that are not verbatim are discarded. Return an empty list when the study compares none of the controls.

Candidate controls:
{controls}"""


def _controls_block(fams: set[str]) -> tuple[str, set[str]]:
    cs = yaml.safe_load(TECHNICAL.read_text())["controls"]
    keep = [c for c in cs if c["family"] in fams]
    return ("\n".join(f"{c['id']} [{c['family']}] {c['control']} (vs {c['comparator']})" for c in keep),
            {c["id"] for c in keep})


def _families_for(p) -> set[str]:
    schema = {"type": "object", "additionalProperties": False, "required": ["families"],
              "properties": {"families": {"type": "array", "items": {"type": "string", "enum": list(FAMILIES)}}}}
    sys_ = ("Name up to 4 families of technical implementation choices that this study compares or evaluates.\n\n"
            + "\n".join(f"{k}: {v}" for k, v in FAMILIES.items()))
    m = CFG["models"]["screen"]
    return set(llm.call(m["vendor"], m["model"], sys_, f"Title: {p['title']}\nAbstract: {p['abstract']}", schema,
                        step="tech_route", effort="low", max_tokens=1000)["families"])


def link(workers: int = 10, only: set[str] | None = None):
    from .evidence import SCREEN, _fn, _majority, fulltext, level
    LINKS_DIR.mkdir(parents=True, exist_ok=True)
    papers = fetch.load()
    rows = [json.loads(l) for l in SCREEN.read_text().splitlines()]
    todo = [r for r in rows if r["category"] == "empirical_ai_health" and r["tests_practice"]
            and not (LINKS_DIR / f"{_fn(r['id'])}.json").exists() and (only is None or r["id"] in only)]
    todo.sort(key=lambda r: r["setting"] != "real_deployment")
    print(f"{len(todo)} practice-testing studies to link to technical controls", flush=True)

    def one(r):
        p = papers.get(r["id"])
        if not p:
            return
        try:
            fams = _families_for(p)
            if not fams:
                res = {"id": r["id"], "links": [], "families": []}
            else:
                block, valid = _controls_block(fams)
                ft = fulltext(p) if r["setting"] == "real_deployment" else ""
                text = f"Abstract: {p['abstract']}" + (f"\n\nFull text (methods and results first):\n{ft}" if ft else "")
                nt = quotes.norm(text)
                members = CFG["models"]["fast_panel"] if r["setting"] != "benchmark_only" else {"claude": CFG["models"]["hazard"]["model"]}
                votes = defaultdict(list)
                for vendor, model in members.items():
                    try:
                        out = llm.call(vendor, model, LINK_SYSTEM.replace("{controls}", block),
                                       f"Title: {p['title']}\n\n<text>\n{text}\n</text>", LINK_SCHEMA,
                                       step="tech_link", effort="high", max_tokens=16000)
                    except llm.BudgetExceeded:
                        raise
                    except Exception:
                        continue
                    for l in out["links"]:
                        if l["control_id"] in valid and quotes.verify(l["quote"], text, nt)[0]:
                            votes[l["control_id"]].append((vendor, l))
                need = 2 if len(members) >= 3 else 1
                links = []
                for cid_, items in votes.items():
                    if len({v for v, _ in items}) < need:
                        continue
                    finding, share = _majority([l["finding"] for _, l in items])
                    best = next(l for _, l in items if l["finding"] == finding)
                    links.append({**best, "finding": finding, "finding_agreement": round(share, 2),
                                  "linked_by": sorted({v for v, _ in items})})
                res = {"id": r["id"], "families": sorted(fams), "links": links,
                       "setting": r["setting"], "design_hint": r.get("practice", "")}
        except llm.BudgetExceeded:
            raise
        except Exception as e:
            print("  tech link fail", r["id"], repr(e)[:120], flush=True)
            return
        (LINKS_DIR / f"{_fn(r['id'])}.json").write_text(json.dumps(res, indent=1))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))


def assemble():
    """Join technical links with study design (from the main extraction) and write technical_evidence.jsonl."""
    from .evidence import EXTRACT_DIR, SCREEN, _clean_title, _fn, _randomization_stated, _study_tech, level
    papers = fetch.load()
    screen_tech = {json.loads(l)["id"]: json.loads(l).get("technology", []) for l in SCREEN.read_text().splitlines()}
    out = []
    for f in LINKS_DIR.glob("*.json"):
        x = json.loads(f.read_text())
        main = EXTRACT_DIR / f"{_fn(x['id'])}.json"
        if not x.get("links") or not main.exists():
            continue
        mx = json.loads(main.read_text())
        p = papers.get(x["id"], {})
        design = mx.get("design", "other")
        if design in ("rct", "cluster_rct") and not _randomization_stated(p):
            design = "prospective_with_comparator"
        lv = level(design, mx.get("setting", x.get("setting", "offline_or_simulated")))
        for l in x["links"]:
            out.append({"study": x["id"], "title": _clean_title(p.get("title")), "venue": p.get("venue"),
                        "date": p.get("date"), "doi": p.get("doi"), "pmid": p.get("pmid"), "preprint": p.get("preprint"),
                        "control": l["control_id"], "comparator_used": l["comparator_used"], "finding": l["finding"],
                        "outcome": l["outcome"], "effect": l["effect"], "tradeoff": l["tradeoff"], "quote": l["quote"],
                        "design": design, "setting": mx.get("setting"), "level": lv,
                        "technology": _study_tech(mx, screen_tech), "population": mx.get("population") or {},
                        "linked_by": l.get("linked_by"), "human_verified": False})
    TECH_EVIDENCE.write_text("".join(json.dumps(e) + "\n" for e in out))
    return len(out)
