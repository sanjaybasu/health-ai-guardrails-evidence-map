"""Per-recommendation expert-perspective commentary, grounded only in the map's own evidence.

Perspectives are defined by expert role, not by named people, so no view is attributed to a real person.
Each commentary is drafted by one model from the recommendation's endorsements and linked studies, every cited
study id is checked against the supplied evidence, a second vendor checks each statement for support, and the
draft is revised once from those flags; statements still unsupported are removed. Commentaries are regenerated
only when a recommendation's evidence changes.
"""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import yaml

from . import llm, style
from .llm import CFG, ROOT

OUT = ROOT / "data" / "commentary.jsonl"
_lock = threading.Lock()

LENSES = {
    "evidence_synthesis": "an evidence-synthesis methodologist who rates certainty the way GRADE working-group "
                          "methodologists do, weighing design, directness, consistency, and precision",
    "patient_safety": "a patient-safety scientist who reasons about failure modes, harm severity, and whether a "
                      "safeguard catches errors before they reach patients",
    "clinical_informatics": "a clinical informatics and implementation researcher who asks whether a practice works in "
                            "real workflows, at what burden, and with what fidelity",
    "health_equity": "a health-equity researcher focused on Medicaid, safety-net, limited English proficiency, and "
                     "low-literacy populations, who asks whether evidence covers those populations",
    "regulatory_policy": "a regulatory and health-policy expert on FDA device oversight, ONC transparency rules, "
                         "civil-rights law, and state AI statutes",
    "operations": "a health-system operations leader who weighs cost, staffing, and feasibility of adopting the "
                  "practice at scale",
    "patient_advocate": "a patient advocate who asks what the practice means for patients' safety, trust, and ability "
                        "to reach a human",
}

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["bottom_line", "certainty", "perspectives", "research_gap", "underserved_note"],
    "properties": {
        "bottom_line": {"type": "string", "description": "two sentences at most"},
        "certainty": {"type": "string", "enum": ["high", "moderate", "low", "very_low", "no_direct_evidence"]},
        "perspectives": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["lens", "view", "cites"],
            "properties": {"lens": {"type": "string", "enum": list(LENSES)},
                           "view": {"type": "string", "description": "<= 60 words"},
                           "cites": {"type": "array", "items": {"type": "string"}}}}},
        "research_gap": {"type": "string"},
        "underserved_note": {"type": "string"},
    },
}

SYSTEM = """You write the evidence commentary for one recommendation in an evidence map of guardrails for deploying AI in health care. The commentary speaks for a panel of expert perspectives, each reasoning the way the most widely cited and respected researchers in that field reason:
{lenses}

Use only the endorsements and studies supplied. Cite studies by their ids exactly as given, in the cites field of the perspective that relies on them; cite nothing that is not listed. When no study tests the practice, say so plainly; a hazard study shows that a problem exists, not that the practice solves it. Report numbers as given and hedge interpretations ("suggests", "may"). No superlatives or evaluative adjectives. Write complete sentences with plain words for a health system administrator. Include every perspective that has something evidence-based to say; omit a perspective rather than speculate. certainty follows GRADE conventions applied to tests of the practice only: no_direct_evidence when no study tests it. underserved_note states whether any test included Medicaid, safety-net, limited English proficiency, or low-literacy populations, and what that implies.

Style: write as an experienced health services researcher would in a journal commentary. Put the observation or number first and let the reader reach its significance. Every sentence has a subject and a finite verb; no fragments, no colon-fragment lines, no rhetorical questions. Do not use these words: delve, harness, tapestry, leverage, robust (except as a statistical term), seamless, navigate, unleash, elevate, pivotal, synergy, holistic, paradigm, ecosystem, journey, unlock, streamline, optimize, empower, transformative, cutting-edge, crucial, critical, landscape, realm, underscores, novel, comprehensive. Do not use "not just X but Y", "not only", "it is worth noting", "notably", "moreover", "furthermore", "importantly", "overall", "ultimately", "in summary", em dashes, or exclamation marks. Do not restate the bottom line at the end."""

CHECK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["unsupported"],
    "properties": {"unsupported": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["text", "reason"],
        "properties": {"text": {"type": "string"}, "reason": {"type": "string"}}}}},
}

CHECK_SYSTEM = """You check an evidence commentary for faithfulness. Given the evidence supplied to the writer and the commentary, list every statement that the evidence does not support: numbers that do not appear, claims about studies that the study entries do not make, conclusions stronger than the designs allow, or statements about populations not described. Return an empty list if every statement is supported."""


def _evidence_block(rec, ends, ev):
    lines = [f"Recommendation {rec['id']}: {rec['statement']}", f"Hazard it targets: {rec['hazard']}", "",
             f"Endorsing sources ({len(ends)}):"]
    for e in ends[:12]:
        lines.append(f"- {e['title']} ({e['kind']}, {e['strength']}): \"{e['quote'][:300]}\"")
    tests = [e for e in ev if e["relation"] == "tests_guardrail"]
    haz = [e for e in ev if e["relation"] == "documents_hazard"]
    for label, rows in (("Studies testing the practice", tests), ("Studies documenting the hazard", haz)):
        lines += ["", f"{label} ({len(rows)}):"]
        for e in sorted(rows, key=lambda e: e["level"])[:25]:
            pops = [k for k, v in (e.get("population") or {}).items() if v]
            lines.append(f"- [{e['study']}] level {e['level']}, {e['design']}, {e['setting']}, {e.get('country') or ''}"
                         f"{', n=' + str(e['sample_size']) + ' ' + (e.get('sample_unit') or '') if e.get('sample_size') else ''}"
                         f"{', populations: ' + ', '.join(pops) if pops else ''}; finding {e['finding']}; "
                         f"{e['outcome']}: {e['effect']}")
    return "\n".join(lines), {e["study"] for e in tests[:25] + haz[:25]}


def _hash(block: str) -> str:
    return hashlib.sha256(block.encode()).hexdigest()[:16]


def generate(workers: int = 8, only: set[str] | None = None):
    from .build_site import compute
    data = compute()
    have = {}
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            x = json.loads(l)
            have[x["rec"]] = x
    system = SYSTEM.replace("{lenses}", "\n".join(f"- {k}: {v}" for k, v in LENSES.items()))
    m = CFG["models"]["merge"]
    checker = CFG["models"]["panel"]["gemini"]

    def one(r):
        block, valid = _evidence_block(r, r["endorsements"], r["evidence"])
        h = _hash(block)
        if have.get(r["id"], {}).get("hash") == h:
            return
        draft = llm.call(m["vendor"], m["model"], system, block, SCHEMA, step="commentary_draft", effort="high",
                         max_tokens=16000)
        for p in draft["perspectives"]:
            p["cites"] = [c for c in p["cites"] if c in valid]
        text = json.dumps(draft, indent=0)
        flags = llm.call("gemini", checker, CHECK_SYSTEM, f"Evidence:\n{block}\n\nCommentary:\n{text}", CHECK_SCHEMA,
                         step="commentary_check", effort="high", max_tokens=8000)["unsupported"]
        if flags:
            draft = llm.call(m["vendor"], m["model"], system,
                             f"{block}\n\nYour previous draft:\n{text}\n\nA reviewer flagged these statements as not "
                             f"supported by the evidence:\n{json.dumps(flags, indent=0)}\n\nRewrite the commentary, "
                             f"removing or correcting every flagged statement.", SCHEMA, step="commentary_revise",
                             effort="high", max_tokens=16000)
            for p in draft["perspectives"]:
                p["cites"] = [c for c in p["cites"] if c in valid]
            flags2 = llm.call("gemini", checker, CHECK_SYSTEM,
                              f"Evidence:\n{block}\n\nCommentary:\n{json.dumps(draft, indent=0)}", CHECK_SCHEMA,
                              step="commentary_check", effort="high", max_tokens=8000)["unsupported"]
            bad = {f["text"] for f in flags2}
            draft["perspectives"] = [p for p in draft["perspectives"] if not any(b and b in p["view"] for b in bad)]
            draft["residual_flags"] = len(flags2)
        viol = style.lint_obj({k: draft[k] for k in ("bottom_line", "research_gap", "underserved_note")}
                              | {"views": [p["view"] for p in draft["perspectives"]]})
        if viol:
            fixed = llm.call(m["vendor"], m["model"], system,
                             f"{block}\n\nYour draft:\n{json.dumps(draft, indent=0)}\n\nThe draft breaks these style "
                             f"rules:\n{json.dumps(sorted(set(viol)))}\n\nRewrite it so no rule is broken, keeping every "
                             f"supported claim.", SCHEMA, step="commentary_style", effort="medium", max_tokens=16000)
            for p in fixed["perspectives"]:
                p["cites"] = [c for c in p["cites"] if c in valid]
            fixed["perspectives"] = [p for p in fixed["perspectives"] if not style.lint(p["view"])]
            for k in ("bottom_line", "research_gap", "underserved_note"):
                if style.lint(fixed[k]):
                    fixed[k] = draft[k] if not style.lint(draft[k]) else ""
            draft = {**draft, **fixed}
        rec = {"rec": r["id"], "hash": h, "model": m["model"], "checker": checker, "initial_flags": len(flags),
               "style_violations_initial": len(viol), **draft}
        with _lock:
            have[r["id"]] = rec
            tmp = OUT.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(x) + "\n" for x in have.values()))
            tmp.replace(OUT)

    recs = [r for r in data["recs"] if only is None or r["id"] in only]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, recs))
    return len(have)
