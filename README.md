# Guardrail Evidence Map

A weekly-updated map that links each recommended guardrail for deploying artificial intelligence in health care (large language models, predictive models, imaging software, multimodal models, ambient scribes, and agents) to the sources that endorse it and the studies that tested it, with attention to Medicaid, safety-net, limited English proficiency, and low-literacy populations.

The site is published at https://sanjaybasu.github.io/health-ai-guardrails-evidence-map/.

## What it answers

Governance frameworks (CHAI, Joint Commission and CHAI, the Health AI Partnership, NIST AI 600-1, the AMA), federal regulation, state statutes, and many commentaries recommend practices for AI deployment: clinician review of AI-drafted messages, disclosure to patients, local validation and silent-mode testing before go-live, alert threshold selection, drift monitoring, change control for model updates, escalation of crisis content, and others. For each practice the map reports how many distinct sources endorse it, whether any study has tested it, the strongest design among those tests, and whether any test included an underserved population. Studies that only document the hazard a practice targets are shown separately from tests of the practice.

## How it works

| Step | Code | Models |
|---|---|---|
| Fetch governance sources and hash them to detect revisions | `gem/governance.py` | none |
| Extract atomic recommendations with verbatim quotes; merge, critique, revise | `gem/taxonomy.py` | claude-opus-5-5, gpt-6-astra, gemini-3.1-pro-preview |
| Retrieve literature (Europe PMC incl. medRxiv/bioRxiv/Research Square, arXiv, OpenAlex) | `gem/fetch.py` | none |
| Screen titles and abstracts; audit a sample of exclusions with a second vendor | `gem/evidence.py` | gemini-3.8-flash, claude-sonnet-5-5 |
| Calibrated second screen; arbitrate disagreements | `gem/jev.py` | Jev (jev-latest, TypeSafe), claude-sonnet-5-5 |
| Route each study to relevant domains; extract links with a three-vendor panel; cross-examine single-vendor links | `gem/evidence.py` | real-deployment tests: claude-opus-5-5, gpt-6-astra, gemini-3.1-pro-preview; offline tests: claude-sonnet-5-5, gpt-5.5, gemini-3.8-flash |
| Score whether each quote supports its link | `gem/jev.py` | Jev |
| Verify every quote against the source text | `gem/quotes.py` | none |
| Assign evidence levels, compute status, build the site | `gem/evidence.py`, `gem/build_site.py` | none |

A link from a study to a recommendation needs at least two of the three vendors to agree, either independently or when a second vendor is shown a link the first proposed, and a quote that appears verbatim in the text the models read. Bibliographic details come only from Europe PMC, arXiv, and OpenAlex. Everything the models produce is marked machine-extracted until a person verifies it. The methods page on the site gives the evidence levels, status rules, agreement statistics, and screening-audit results.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=... TYPESAFE_API_KEY=... NCBI_API_KEY=...
python -m gem.pipeline seed-taxonomy            # once, then curate data/taxonomy.yaml by hand
python -m gem.pipeline backfill --since 2023-01-01
python -m gem.pipeline weekly                   # what the scheduled workflow runs
python -m gem.pipeline build                    # rebuild docs/ only
pytest -q
```

Budget caps live in `config.yaml` (`GEM_RUN_CAP` and `GEM_MONTH_CAP` override them). Every step is resumable: model responses are cached by prompt hash, extractions are written one file per study, and a run that hits a cap stops cleanly and resumes next time.

## Curation

- `data/taxonomy.yaml`: set a recommendation's `status` to `curated` after reviewing its wording; edit statements freely (ids are stable).
- `data/proposed_recommendations.jsonl`: practices that studies or commentaries mention but the taxonomy lacks; add them to the taxonomy by hand.
- `data/evidence.jsonl`: set `human_verified: true` only after checking a link against the paper.
- Weekly pull requests list new links and status changes; they merge automatically when `gem/validate_site.py` and the tests pass, and stay open for review otherwise.

## Scheduled updates

`.github/workflows/weekly.yml` runs every Monday at 06:00 UTC, opens a pull request with new data and a rebuilt site, merges it if the site validation passes, and commits a heartbeat to `main` so the schedule stays active. The site is served by GitHub Pages from `docs/`.

## Conflict of interest

The maintainers are employees of Waymark, a public benefit organization that provides free social and medical services for patients receiving Medicaid. Studies by Waymark authors are included under the same rules as all other studies.

## License

The code is released under the MIT license, and the data and site content under CC BY 4.0. Quotes from source documents remain the property of their authors and are reproduced as short excerpts for commentary and research.
