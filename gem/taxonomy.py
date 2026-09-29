"""Seed and maintain the recommendation taxonomy.

1. extract: each panel vendor independently extracts atomic, checkable deployment recommendations (with a verbatim
   quote) from every governance source chunk. Quotes are verified against the source text; failures are dropped.
2. canonicalize: candidates are embedded and clustered; Opus merges clusters into canonical recommendations and
   assigns every candidate to one canonical recommendation (or to none, if out of scope).
3. critique and revise: GPT-6 and Gemini critique the draft taxonomy (non-atomic items, duplicates, untestable
   wording, gaps); Opus revises. The output is a machine draft that a human curates before it becomes canonical.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import yaml

from . import governance, llm, quotes
from .llm import CFG, ROOT

CAND = ROOT / "data" / "taxonomy_candidates.jsonl"
DRAFT = ROOT / "data" / "taxonomy_draft.json"
TAXONOMY = ROOT / "data" / "taxonomy.yaml"
ENDORSE = ROOT / "data" / "endorsements.jsonl"

DOMAINS = {
    "governance_accountability": "AI governance structures, named accountable owners, policies, approval before use",
    "inventory_risk_tiering": "inventory of AI tools, risk classification or tiering of use cases, scope limits by risk",
    "use_case_scoping": "restricting what the tool may do (e.g., no diagnosis, no autonomous orders, allowed tasks)",
    "pre_deployment_evaluation": "testing accuracy, safety, and failure modes before go-live; red-teaming; benchmarks",
    "local_validation": "validating on the local population and workflow, including subgroup performance",
    "human_oversight": "human review of outputs, sign-off, accountable human authorship, override ability",
    "patient_disclosure_consent": "telling patients AI is involved, how to reach a human, consent or opt-out",
    "clinician_transparency": "model cards, source attribution, training on limitations for staff users",
    "escalation_safety_net": "detecting urgent or crisis content and routing to humans; emergency instructions",
    "language_access": "limited English proficiency, translation quality, non-English performance, interpreters",
    "health_literacy_accessibility": "reading level, plain language, disability access, digital access",
    "equity_bias": "bias assessment and mitigation, nondiscrimination, fairness across groups",
    "privacy_security_data": "PHI protection, data use limits, security, vendor data terms, prompt injection",
    "monitoring_incident": "post-deployment monitoring, drift, incident reporting, audit logs, decommissioning",
    "vendor_procurement": "vendor due diligence, contracts, third-party assurance, documentation from developers",
    "workforce_training": "training and competency for staff, change management, avoiding deskilling",
    "patient_community_engagement": "involving patients and communities in design, feedback channels, complaints",
    "output_grounding_accuracy": "grounding in sources, citation of evidence, hallucination controls, uncertainty",
    "alerting_workflow": "alert thresholds and burden, workflow integration, automation bias, override and deferral",
    "model_updates_change_control": "versioning, retraining, revalidation after updates, predetermined change control",
}

TECH = {
    "any_ai": "applies to AI tools generally",
    "generative_llm": "large language models and other generative AI (text, chat, drafting, summarization)",
    "multimodal_foundation": "multimodal or foundation models combining text, images, signals, or EHR data",
    "predictive_risk": "predictive or risk-scoring models (deterioration, sepsis, readmission, rising risk)",
    "imaging_diagnostic": "imaging, pathology, or other diagnostic device software",
    "agentic": "AI agents that take actions or call tools, including autonomous workflows",
    "speech_ambient": "ambient documentation and speech recognition",
    "world_model_simulation": "world models, digital twins, or simulation-based decision tools",
}

TARGETS = ["patient_facing", "clinician_or_staff_facing", "organizational"]

CAND_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["recommendations"],
    "properties": {"recommendations": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["statement", "quote", "strength", "domain", "applies_to", "genai_specific", "technology", "hazard"],
        "properties": {
            "statement": {"type": "string", "description": "One atomic practice, imperative voice, <= 30 words"},
            "quote": {"type": "string", "description": "Verbatim sentence(s) from the source, <= 60 words"},
            "strength": {"type": "string", "enum": ["must", "should", "may", "describes"]},
            "domain": {"type": "string", "enum": list(DOMAINS)},
            "applies_to": {"type": "array", "items": {"type": "string", "enum": TARGETS}},
            "genai_specific": {"type": "boolean"},
            "technology": {"type": "array", "items": {"type": "string", "enum": list(TECH)}},
            "hazard": {"type": "string", "description": "The harm or failure this practice is meant to prevent, <= 20 words"},
        }}}},
}

EXTRACT_SYSTEM = """You are extracting deployment guardrail recommendations from a health care AI governance document for an evidence map. Health system administrators will use the map to decide which practices to adopt when deploying any AI tool for patients or for clinicians and staff: large language models and other generative AI, multimodal and foundation models, predictive risk models, imaging and diagnostic software, ambient documentation, AI agents, and simulation or world-model tools.

Extract every distinct recommendation in the text that a health care organization (or the developer supplying it) could apply when deploying or operating such a tool. Exclude recommendations addressed only to regulators' own processes and purely internal model-training procedures with no bearing on deployment. Set technology to the kinds of AI the recommendation concerns (any_ai when it applies generally) and genai_specific to true only when it concerns generative AI in particular.

Rules:
- statement: one atomic, checkable practice in imperative voice ("Require clinician review of AI-drafted patient messages before sending"). Split compound recommendations into separate items. Be specific enough that a study could test whether the practice achieves its aim.
- quote: copy the supporting text verbatim, character for character, from the document (<= 60 words). Do not paraphrase, correct, or join non-adjacent sentences. Items whose quote is not verbatim are discarded.
- strength: "must" for mandatory language (must, shall, required), "should" for recommended, "may" for optional, "describes" when the document describes a practice without recommending it.
- Do not invent recommendations that are not in the text. Return an empty list if there are none."""

SUPPLEMENT_NOTE = """

A previous pass over this document extracted only recommendations that apply to large language models and generative AI. In this pass extract only the recommendations that pass would have excluded: those that are specific to, or mainly concern, other kinds of health care AI (predictive risk models and their thresholds, calibration, alerts, and drift; imaging and diagnostic device software; device change control and updates; ambient speech; agents; simulation tools). Return an empty list if there are none."""


def _chunks(text: str, words: int = 5000, overlap: int = 200):
    w = text.split()
    i = 0
    while i < len(w):
        yield " ".join(w[i:i + words])
        i += words - overlap


def extract(only: set[str] | None = None, workers: int = 9, supplement: bool = False):
    """Run the three-vendor extraction over every source chunk in parallel; append verified candidates."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    done = set()
    if CAND.exists():
        for line in CAND.read_text().splitlines():
            r = json.loads(line)
            done.add((r["source"], r["chunk"], r["vendor"], r.get("pass", "main")))
    panel = CFG["models"]["panel"]
    lock = threading.Lock()
    pass_ = "supp" if supplement else "main"
    system = EXTRACT_SYSTEM + (SUPPLEMENT_NOTE if supplement else "")
    jobs = []
    for s in governance.sources():
        if only and s["id"] not in only:
            continue
        full = governance.text(s["id"])
        nfull = quotes.norm(full)
        for ci, chunk in enumerate(_chunks(full)):
            for vendor, model in panel.items():
                if (s["id"], ci, vendor, pass_) not in done:
                    jobs.append((s, full, nfull, ci, chunk, vendor, model))

    def run(job):
        s, full, nfull, ci, chunk, vendor, model = job
        user = f"Document: {s['title']} ({s['issuer']}, {s['date']}; kind: {s['kind']}).\n\n<document>\n{chunk}\n</document>"
        try:
            out = llm.call(vendor, model, system, user, CAND_SCHEMA, step=f"tax_extract_{pass_}:{s['id']}",
                           effort="high", max_tokens=64000)
        except Exception as e:
            print("  fail", s["id"], ci, vendor, repr(e)[:160], flush=True)
            return
        recs = []
        for j, r in enumerate(out["recommendations"]):
            ok, score = quotes.verify(r["quote"], full, nfull)
            recs.append({**r, "id": f"{s['id']}:{pass_}{ci}:{vendor}:{j}", "source": s["id"], "chunk": ci,
                         "pass": pass_, "vendor": vendor, "model": model, "quote_ok": ok, "quote_score": score})
        with lock, CAND.open("a") as f:
            for r in recs or [{"id": f"{s['id']}:{pass_}{ci}:{vendor}:none", "source": s["id"], "chunk": ci,
                               "pass": pass_, "vendor": vendor, "model": model, "empty": True}]:
                f.write(json.dumps(r) + "\n")
        print(f"  {s['id']}[{ci}] {vendor}: {len(recs)} recs, {sum(r['quote_ok'] for r in recs)} verified; "
              f"run ${llm.run_spend():.2f}", flush=True)

    print(f"{len(jobs)} extraction jobs")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, jobs))


def candidates(verified_only: bool = True) -> list[dict]:
    out = [json.loads(l) for l in CAND.read_text().splitlines()]
    return [r for r in out if not r.get("empty") and (r.get("quote_ok") or not verified_only)]


# ----------------------------------------------------------------------------- canonicalize

def _embed(texts: list[str]) -> np.ndarray:
    c = llm.client("openai")
    vecs = []
    for i in range(0, len(texts), 256):
        r = c.embeddings.create(model=CFG["models"]["embed"]["model"], input=texts[i:i + 256])
        vecs += [d.embedding for d in r.data]
    v = np.array(vecs)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _cluster(v: np.ndarray, thr: float = 0.86) -> list[list[int]]:
    """Average-linkage agglomerative clustering on cosine distance; clusters join above similarity `thr`."""
    if len(v) == 1:
        return [[0]]
    from scipy.cluster.hierarchy import fcluster, linkage
    Z = linkage(v, method="average", metric="cosine")
    lab = fcluster(Z, t=1 - thr, criterion="distance")
    groups = defaultdict(list)
    for i, l in enumerate(lab):
        groups[l].append(i)
    return list(groups.values())


CANON_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["recommendations"],
    "properties": {"recommendations": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["key", "statement", "domain", "applies_to", "genai_specific", "technology", "hazard", "variants", "members"],
        "properties": {
            "key": {"type": "string", "description": "short snake_case slug"},
            "statement": {"type": "string"},
            "domain": {"type": "string", "enum": list(DOMAINS)},
            "applies_to": {"type": "array", "items": {"type": "string", "enum": TARGETS}},
            "genai_specific": {"type": "boolean"},
            "technology": {"type": "array", "items": {"type": "string", "enum": list(TECH)}},
            "hazard": {"type": "string"},
            "variants": {"type": "array", "items": {"type": "string"}},
            "members": {"type": "array", "items": {"type": "string"}, "description": "cluster ids"},
        }}}},
}

CANON_SYSTEM = """You are building the canonical recommendation taxonomy for an evidence map of guardrails for deploying AI in health care. You receive candidate recommendations extracted by three models from governance documents, pre-grouped into tight similarity clusters within one domain.

Produce canonical recommendations at the level at which an empirical study would test a practice. A good canonical recommendation is one practice a health system either adopts or does not, stated so a study could compare outcomes with and without it (for example, "Require clinician review of AI-drafted patient messages before they are sent"). Merge candidates that differ only in wording, actor, level of detail, or implementation specifics, and record those specifics as variants. Keep two practices separate only when a study would evaluate them separately (for example, reviewing every message versus reviewing a sample; disclosing AI use versus offering an opt-out). Most domains need between 5 and 25 canonical recommendations. Do not create recommendations that no cluster supports. Assign every cluster id to exactly one canonical recommendation's members, except clusters that are too vague to test or out of scope, which you leave unassigned. The statement is neutral and imperative, 8 to 30 words, and names the actor, the action, and the object. variants lists up to 6 short phrases naming the main implementation variants."""

CRITIQUE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["issues"],
    "properties": {"issues": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["type", "keys", "explanation", "fix"],
        "properties": {
            "type": {"type": "string", "enum": ["not_atomic", "duplicate", "untestable", "vague", "misclassified",
                                                 "overreach_beyond_sources", "missing_distinction"]},
            "keys": {"type": "array", "items": {"type": "string"}},
            "explanation": {"type": "string"},
            "fix": {"type": "string"},
        }}}},
}

CRITIQUE_SYSTEM = """You are a methods reviewer for an evidence map that links health care AI deployment recommendations to empirical evidence. Review the draft taxonomy below. Recommendations are meant to sit at the level at which a study would test a practice, with implementation details kept as variants, so most domains have 5 to 25 items. Report only real problems: recommendations that bundle practices a study would evaluate separately (not_atomic), pairs that are the same practice (duplicate), statements no study could test (untestable or vague), wrong domain or target (misclassified), statements that go beyond what the member statements say (overreach_beyond_sources), and merged items whose difference matters for evidence (missing_distinction). Do not ask for splits that only separate implementation details; those belong in variants. For each, give the keys involved and a concrete fix. Do not propose new recommendations that the sources do not contain."""

REVISE_SYSTEM = """You maintain the canonical taxonomy for an evidence map of guardrails for deploying AI in health care. Two independent reviewers (from different model vendors) critiqued the draft for one domain. Apply each critique you agree with, reject ones you disagree with (including requests to split items whose difference is only an implementation detail, which belongs in variants), and return the full revised list for the domain in the same schema, preserving member cluster ids (members of a split recommendation are divided between the new items; members of merged items are combined). Keep keys stable for recommendations you do not change."""


MERGE_SYSTEM = """You are consolidating canonical recommendations for one domain of an evidence map of guardrails for deploying AI in health care. The recommendations below were produced in separate batches, so the same practice may appear more than once under different wording. Merge items that are the same practice, combining their variants; keep items separate only when a study would evaluate them separately (for example, review of every message versus review of a sample). Return no more than 30 items; if more remain, merge the closest pairs and keep their differences as variants. Each output item lists the input keys it absorbs as members. Every input key must appear in exactly one output item's members, unless the item is too vague to test, in which case leave it out. Statements are neutral, imperative, 8 to 30 words, and name the actor, the action, and the object."""


def _canon_batches(dom, clusters, keys, m):
    """Canonicalize clusters in batches of ~250; members are cluster ids."""
    out = []
    for bi in range(0, len(keys), 250):
        batch = keys[bi:bi + 250]
        listing = "\n".join(
            f"[{k}] n={len(clusters[k]['members'])} sources={','.join(clusters[k]['sources'])} ({clusters[k]['strength']}): "
            f"{clusters[k]['statement']}" + "".join(f"\n      alt: {a}" for a in clusters[k]["alt"]) for k in batch)
        user = (f"Domain: {dom} ({DOMAINS[dom]}). {len(batch)} tight clusters of candidate statements; each line shows "
                f"the cluster id, size, sources, typical strength, the most central statement, and up to two alternates. "
                f"Members are cluster ids.\n\n{listing}")
        r = llm.call(m["vendor"], m["model"], CANON_SYSTEM, user, CANON_SCHEMA, step=f"tax_canon:{dom}",
                     effort="high", max_tokens=100000)
        for x in r["recommendations"]:
            x["key"] = f"{x['key']}__b{bi // 250}"
        out += r["recommendations"]
    if len(keys) <= 250:
        return out
    listing = "\n".join(f"[{x['key']}] ({len(x['members'])} clusters) {x['statement']}" for x in out)
    merged = llm.call(m["vendor"], m["model"], MERGE_SYSTEM, f"Domain: {dom} ({DOMAINS[dom]}).\n\n{listing}",
                      CANON_SCHEMA, step=f"tax_merge:{dom}", effort="high", max_tokens=64000)
    by = {x["key"]: x for x in out}
    for x in merged["recommendations"]:
        x["members"] = [c for k in x["members"] if k in by for c in by[k]["members"]]
    return merged["recommendations"]


def canonicalize():
    from scipy.cluster.hierarchy import leaves_list, linkage
    cands = candidates()
    by_domain = defaultdict(list)
    for c in cands:
        by_domain[c["domain"]].append(c)
    draft = json.loads(DRAFT.read_text()) if DRAFT.exists() else {}
    m = CFG["models"]["merge"]
    for dom, cs in sorted(by_domain.items(), key=lambda kv: len(kv[1])):
        if dom in draft and draft[dom].get("revised"):
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
            clusters[k] = {"members": [cs[i]["id"] for i in g], "statement": cs[med]["statement"],
                           "alt": [cs[i]["statement"] for i in g if i != med][:1],
                           "sources": sorted({cs[i]["source"] for i in g}),
                           "strength": Counter(cs[i]["strength"] for i in g).most_common(1)[0][0]}
        canon = _canon_batches(dom, clusters, keys, m)
        def shown(recs):
            return json.dumps([{"key": r["key"], "statement": r["statement"], "domain": r["domain"],
                                "applies_to": r["applies_to"], "genai_specific": r["genai_specific"],
                                "technology": r.get("technology"), "hazard": r["hazard"], "n_sources": len({s_ for k in r["members"] if k in clusters for s_ in clusters[k]["sources"]}),
                                "example_member_statements": [clusters[k]["statement"] for k in r["members"][:4] if k in clusters]}
                               for r in recs], indent=0)
        crit = {}
        for vendor in ("openai", "gemini"):
            crit[vendor] = llm.call(vendor, CFG["models"]["panel"][vendor], CRITIQUE_SYSTEM,
                                    f"Domain: {dom} ({DOMAINS[dom]}).\n\nDraft:\n{shown(canon)}",
                                    CRITIQUE_SCHEMA, step=f"tax_critique:{dom}", effort="high", max_tokens=32000)["issues"]
        rev_user = (f"Domain: {dom}.\n\nDraft (members are cluster ids; keep them with the practice they belong to):\n"
                    f"{json.dumps(canon, indent=0)}\n\n"
                    f"Reviewer A (OpenAI):\n{json.dumps(crit['openai'], indent=0)}\n\n"
                    f"Reviewer B (Gemini):\n{json.dumps(crit['gemini'], indent=0)}")
        revised = llm.call(m["vendor"], m["model"], REVISE_SYSTEM, rev_user, CANON_SCHEMA, step=f"tax_revise:{dom}",
                           effort="high", max_tokens=100000)["recommendations"]
        expand = lambda recs: [{**r, "clusters": r["members"],
                                "members": [cid for k in r["members"] if k in clusters for cid in clusters[k]["members"]]}
                               for r in recs]
        assigned = {k for r in revised for k in r["members"]} & set(clusters)
        draft[dom] = {"n_candidates": len(cs), "n_clusters": len(clusters), "n_assigned_clusters": len(assigned),
                      "initial": expand(canon), "critiques": crit, "final": expand(revised), "revised": True}
        DRAFT.write_text(json.dumps(draft, indent=1))
        print(f"  {dom}: {len(cs)} cands, {len(clusters)} clusters -> {len(canon)} -> {len(revised)} recs; "
              f"{len(assigned)}/{len(clusters)} clusters assigned; issues {len(crit['openai'])}+{len(crit['gemini'])}; "
              f"run ${llm.run_spend():.2f}", flush=True)


def consolidate(max_items: int = 30):
    """Merge any domain whose final list exceeds `max_items` (single-batch domains skip the cross-batch merge)."""
    draft = json.loads(DRAFT.read_text())
    m = CFG["models"]["merge"]
    for dom, d in draft.items():
        fin = d["final"]
        if len(fin) <= max_items or d.get("consolidated"):
            continue
        listing = "\n".join(f"[{r['key']}] ({len(r['members'])} candidates) {r['statement']}"
                             + (f" | variants: {'; '.join(r.get('variants', []))}" if r.get("variants") else "") for r in fin)
        out = llm.call(m["vendor"], m["model"], MERGE_SYSTEM, f"Domain: {dom} ({DOMAINS[dom]}).\n\n{listing}",
                       CANON_SCHEMA, step=f"tax_consolidate:{dom}", effort="high", max_tokens=64000)["recommendations"]
        by = {r["key"]: r for r in fin}
        merged = []
        for x in out:
            src = [by[k] for k in x["members"] if k in by]
            if not src:
                continue
            x["clusters"] = [c for r in src for c in r.get("clusters", [])]
            x["members"] = [c for r in src for c in r["members"]]
            x["variants"] = list(dict.fromkeys(x.get("variants", []) + [v for r in src for v in r.get("variants", [])]))[:8]
            merged.append(x)
        d["pre_consolidation"] = fin
        d["final"] = merged
        d["consolidated"] = True
        DRAFT.write_text(json.dumps(draft, indent=1))
        print(f"  consolidated {dom}: {len(fin)} -> {len(merged)}", flush=True)


def write_taxonomy():
    """Materialize taxonomy.yaml (stable ids) and endorsements.jsonl from the revised draft."""
    draft = json.loads(DRAFT.read_text())
    cands = {c["id"]: c for c in candidates()}
    srcs = {s["id"]: s for s in governance.sources()}
    old = yaml.safe_load(TAXONOMY.read_text())["recommendations"] if TAXONOMY.exists() else []
    old_by_key = {r["key"]: r for r in old}
    next_n = max([int(r["id"][1:]) for r in old] or [0]) + 1
    recs, edges = [], []
    for dom in DOMAINS:
        for r in draft.get(dom, {}).get("final", []):
            members = [cands[m] for m in r["members"] if m in cands]
            if not members:
                continue
            prev = old_by_key.get(r["key"])
            rid = prev["id"] if prev else f"R{next_n:03d}"
            if not prev:
                next_n += 1
            recs.append({"id": rid, "key": r["key"], "domain": dom, "statement": r["statement"],
                         "applies_to": r["applies_to"], "genai_specific": r["genai_specific"],
                         "technology": r.get("technology", ["any_ai"]), "hazard": r["hazard"], "variants": r.get("variants", []), "status": prev.get("status", "machine_draft") if prev else "machine_draft"})
            best = {}
            for c in members:   # one endorsement edge per (recommendation, source): strongest verified quote
                rank = {"must": 0, "should": 1, "may": 2, "describes": 3}[c["strength"]]
                k = c["source"]
                if k not in best or rank < best[k][0] or (rank == best[k][0] and c["quote_score"] > best[k][1]["quote_score"]):
                    best[k] = (rank, c)
            for k, (_, c) in best.items():
                vendors = sorted({m["vendor"] for m in members if m["source"] == k})
                edges.append({"rec": rid, "source": k, "source_kind": srcs[k]["kind"], "strength": c["strength"],
                              "quote": c["quote"], "quote_score": c["quote_score"], "extracted_by": vendors,
                              "human_verified": False})
    TAXONOMY.write_text(yaml.safe_dump({"note": "Machine draft seeded by a three-vendor panel; a recommendation's status "
                                                "becomes 'curated' only when a human reviews it.",
                                        "domains": DOMAINS, "technologies": TECH, "recommendations": recs}, sort_keys=False, width=110))
    ENDORSE.write_text("".join(json.dumps(e) + "\n" for e in edges))
    print(f"taxonomy: {len(recs)} recommendations, {len(edges)} endorsement edges")
