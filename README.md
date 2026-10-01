# Cost-Efficient Agents Literature

Searchable literature database on reducing the cost of production multi-agent LLM systems while keeping or improving task quality. It covers query-level and step-level routing, cascades, compound-system config optimization, learning policies from offline logs and live traffic, quality signals, distillation, caching, serving, and when multi-agent designs are worth the cost.

**Browse it:** https://salmitta.github.io/routing_literature/

## Contents

| Path | What it is |
|---|---|
| `literature_db.json`, `literature_db.csv` | The merged, deduplicated database, one record per source |
| `synthesis.md` | Per-cluster synthesis (key ideas, strongest methods, cost/quality tables, critical notes, open problems), two cross-cutting sections (step-level routing in multi-agent pipelines; learning policies from offline logs vs. online traffic), a must-read top 30 and a gaps list. Every claim cites record IDs |
| `taxonomy.md` | The 15-node taxonomy with record counts per node and per facet |
| `UNVERIFIED.md` | Items seen but not verifiable, records removed after verification, and coverage notes on unreachable sources |
| `docs/` | The GitHub Pages site: `index.html` with the data embedded, `embeddings.bin` for semantic search, and downloadable copies of the data |
| `docs/tools/` | Site build: `build.py` (template to page), `embed.mjs` (bge-small document embeddings), `template.html` |
| `raw/BRIEF.md` | The shared brief given to every search agent: schema, relevance scale, rules and taxonomy |
| `raw/g*.json` | Per-cluster raw outputs from the search agents, before merging |
| `raw/merge.py` | Dedup by arXiv ID, DOI, normalized title and URL; merges cluster tags; links cross-citations |
| `raw/verify_arxiv.py`, `raw/arxiv_check.json` | Batch check of every arXiv ID against the arXiv API: ID resolves and title matches |
| `raw/check_withdrawn.py`, `raw/withdrawn_check.json` | Scan of every arXiv abs page for the withdrawal banner, which the API does not reliably expose |
| `raw/exclude.json` | Records removed after verification, with the reason |
| `raw/synth/` | Per-cluster synthesis drafts and the lead's cross-cutting sections; `raw/assemble_synthesis.py` builds `synthesis.md` |
| `raw/check_citations.py`, `raw/check_numbers.py` | Checks that every cited record ID exists, and that every number in a synthesis table row appears in the cited record |

## Taxonomy

C1 Query-level routing · C2 Cascades, deferral & escalation · C3 Step/subagent-level model assignment · C4 Compound AI config optimization · C5 Online learning from live traffic · C6 Offline learning from logs · C7 Quality signals without ground truth · C8 Distillation & specialization · C9 Reasoning & token budgets · C10 Caching, context & prompts · C11 Serving & systems · C12 Architecture & when multi-agent is worth it · C13 Benchmarks & cost-aware evaluation · C14 Industry products & case studies · C15 Surveys & theory

## Caveats

- Relevance scores (1–5) rate how central an item is to the premise, not paper quality.
- Cost and quality numbers are recorded as each source reports them. They are comparable only on the same benchmark, baseline and price sheet. Vendor numbers are marked as vendor-reported.
- Many 2026 entries are unreviewed preprints. Where a venue could not be confirmed, it is recorded as "arXiv".

## Rebuilding the site

```sh
python3 raw/merge.py                      # raw/g*.json -> literature_db.{json,csv}
python3 raw/verify_arxiv.py && python3 raw/check_withdrawn.py && python3 raw/merge.py
python3 raw/taxonomy.py && python3 raw/assemble_synthesis.py && python3 raw/unverified_md.py
python3 raw/check_citations.py synthesis.md && python3 raw/check_numbers.py synthesis.md
# embeddings need @huggingface/transformers@3.8.1 installed in TFJS_DIR
TFJS_DIR=/path/with/node_modules python3 docs/tools/build.py
```
