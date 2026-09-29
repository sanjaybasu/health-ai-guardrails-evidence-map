from gem.build_site import status
from gem.evidence import aggregate, level
from gem.quotes import verify

SRC = """Organizations should require that a licensed clinician reviews every AI-drafted message
before it is sent to a patient. 12 Health systems must disclose the use of generative AI to patients."""


def test_quote_exact_and_normalized():
    assert verify("a licensed clinician reviews every AI-drafted message before it is sent", SRC)[0]
    assert verify("Health systems MUST disclose the use of generative AI to patients.", SRC)[0]


def test_quote_fabricated_rejected():
    assert not verify("Health systems must obtain written consent before any AI use in care.", SRC)[0]
    assert not verify("short", SRC)[0]


def test_quote_ellipsis_fragments_in_order():
    assert verify("Organizations should require that a licensed clinician ... disclose the use of generative AI", SRC)[0]
    assert not verify("disclose the use of generative AI ... Organizations should require that a licensed", SRC)[0]


def test_levels():
    assert level("rct", "real_deployment") == 1
    assert level("rct", "offline_or_simulated") == 4          # randomized vignette study is not a deployment trial
    assert level("quasi_experimental", "real_deployment") == 2
    assert level("pre_post_no_comparator", "real_deployment") == 3
    assert level("automated_benchmark", "benchmark_only") == 5


def test_status_rules():
    t = lambda lv, f: {"level": lv, "finding": f}
    assert status(5, []) == "consensus_without_evidence"
    assert status(1, []) == "no_data"
    assert status(2, [t(1, "supports_practice")]) == "evidence_supported"
    assert status(2, [t(2, "supports_practice"), t(2, "against_practice")]) == "contested"
    assert status(2, [t(1, "supports_practice"), t(1, "against_practice")]) == "contested"
    assert status(2, [t(4, "supports_practice")]) == "limited_evidence"
    assert status(2, [t(2, "null_or_mixed")]) == "limited_evidence"


def _out(design="rct", n=100):
    pop = {f: False for f in ["medicaid", "safety_net", "limited_english_or_non_english", "low_health_literacy",
                              "low_income", "racial_ethnic_minority_focus", "rural", "older_adults", "disability"]}
    return {"design": design, "setting": "real_deployment", "user": "clinician_or_staff", "country": "US",
            "sample_unit": "clinicians", "sample_size": n, "population": pop, "languages": [], "models_evaluated": [],
            "unlisted_practices": [], "links": []}


def test_aggregate_two_of_three():
    outs = {"claude": _out(), "openai": _out(), "gemini": _out("quasi_experimental", 104)}
    link = lambda f: {"rec_id": "R001", "relation": "tests_guardrail", "finding": f, "outcome": "o", "effect": "e",
                      "quote": "q", "quote_score": 100.0}
    links = {"claude": [link("supports_practice")], "openai": [link("supports_practice")], "gemini": []}
    a = aggregate(outs, links)
    assert a["design"] == "rct" and a["sample_size"] == 100
    assert len(a["links"]) == 1 and a["links"][0]["linked_by"] == ["claude", "openai"]
    links = {"claude": [link("supports_practice")], "openai": [], "gemini": []}
    a = aggregate(outs, links)
    assert a["links"] == [] and a["contested_links"][0]["vendors"] == ["claude"]
