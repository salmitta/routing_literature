# Shared brief for literature subagents

## Premise
We are building a deduplicated literature database on **reducing the cost of production multi-agent LLM systems while keeping or improving task performance**. Today every subagent and step in the pipeline calls one frontier model. The cost levers we care about are:
(a) routing at the incoming-query level;
(b) routing or downscaling at the subagent or step level inside long-horizon trajectories;
(c) learning or approximating an optimal configuration from offline customer data or live traffic;
(d) any other cost lever for production multi-agent systems: caching, context management, distillation, serving, architecture changes.
The reader is an ML researcher building an enterprise agentic system. Favor rigor, reproducible results and production evidence over marketing.

## Output
Write a single JSON file: a list of record objects, to the path given in your task prompt. Write it with the Write tool, or with python3 through Bash. Save partial progress often by overwriting the file, so work is not lost. The JSON must be valid and must parse with `python3 -c "import json;json.load(open(PATH))"`. Run that check before you finish.

## Record schema
Every field is required. Use null when a value is unknown. Never guess.
```
{
  "id": "",                    // slug: firstauthorlastname-year-shorttitle, lowercase, hyphens, e.g. "ong-2024-routellm"
  "title": "",
  "authors": [],               // full list if known; if truncated, end with "et al."
  "year": 0,
  "venue_or_source": "",       // e.g. "NeurIPS 2024", "arXiv", "ICLR 2025", "Anthropic engineering blog"
  "source_type": "",           // paper | preprint | blog | docs | talk | repo | benchmark | product
  "url": "",                   // verified, resolving link (arXiv abs page preferred: https://arxiv.org/abs/XXXX.XXXXX)
  "code_url": null,
  "topic_clusters": [],        // cluster IDs from the list below, e.g. ["C1","C2"]; multiple allowed
  "routing_granularity": "",   // query | step | subagent | token | none | n/a
  "learning_regime": "",       // training-free | supervised-offline | RL-offline | bandit-online | RL-online | hybrid | n/a
  "signals_used": [],          // e.g. "query embedding", "difficulty estimate", "confidence", "judge score", "user feedback"
  "cost_reduction_reported": "", // exactly as reported, with the benchmark and baseline, e.g. "~2x cheaper at 95% of GPT-4 quality on MT-Bench"; null if none
  "benchmarks": [],
  "models_studied": [],
  "production_evidence": false,  // true only for real deployment or real traffic data
  "summary": "",               // 2-3 sentences, paraphrased in your own words
  "key_idea": "",              // one line
  "relevance_score": 0,        // 1-5, see the scale below
  "relevance_rationale": "",
  "found_via": "",             // the search query, "snowball: related work of <id>", or "snowball: cited-by <id>"
  "cites_or_cited_by": []      // ids of other records you collected, where known
}
```

## Relevance scale
- 5 = directly addresses cost-aware routing or config optimization in agentic or multi-agent systems.
- 4 = query-level routing or cascades, clearly transferable.
- 3 = a useful supporting technique (caching, compression, distillation, judges).
- 2 = background.
- 1 = tangential. **Do not include anything scored below 2.**

## Rules
- NEVER fabricate papers, authors, numbers or URLs. Every URL must come from a search result or a page you fetched. If you cannot verify an item, leave it out of the main file. Instead, put it in `"_unverified"` notes (see below).
- Summaries must be paraphrased, not copied.
- Time range: 2019 to present, with emphasis on 2023-2026. Older foundational work is in scope when it is directly used: cascades, learning-to-defer, bandits, off-policy evaluation.
- Prefer the arXiv or official version. If a blog post has a companion paper, record the paper and mention the blog in the summary. Keep the blog as its own record only when it adds production evidence.
- Record quantitative cost/quality claims exactly as reported, including the benchmark and the baseline model. Take numbers from the abstract or paper you actually saw. Do not reconstruct them from memory.
- Keep `cites_or_cited_by` to ids of records within your own file. The lead agent will add cross-file links.

## Search method
1. Run many short, varied queries (1-6 words), combining the keywords with synonyms. Run at least 15 queries per cluster.
2. Search arXiv, Semantic Scholar (api.semanticscholar.org/graph/v1/paper/search?query=...&fields=title,year,authors,externalIds,venue,citationCount works via WebFetch), Google Scholar, OpenReview and ACL Anthology.
3. Also cover NeurIPS, ICML, ICLR, COLM, EMNLP, MLSys and OSDI proceedings, GitHub, and engineering blogs. The blogs are: Anthropic, OpenAI, Google DeepMind/Research, Microsoft Research, Meta, NVIDIA, AWS, Azure, Databricks, Anyscale, LMSYS, Berkeley Sky Lab, Together, Fireworks, Baseten, Modal, LangChain, LlamaIndex, Martian, Not Diamond, OpenRouter, Unify, Portkey, LiteLLM and Hugging Face. Substantive practitioner posts on Substack or Medium also count.
4. Snowball: for every relevance-4/5 hit, check its related-work section and its "cited by" list on Semantic Scholar (api.semanticscholar.org/graph/v1/paper/arXiv:XXXX/citations?fields=title,year,externalIds&limit=100). Add the items that qualify.
5. Collect at least 40 qualifying records per major cluster you own. Stop when new searches mostly return duplicates.
6. A verified arXiv abs URL, or a Semantic Scholar record with an arXiv ID or DOI, counts as verification. You do not have to fetch every page. You must still have seen the title, authors and year in a real result.

## Unverified seeds
If a seed item from your task cannot be found, append an object `{"_unverified": "<seed name>", "note": "<what you tried>"}` to a second file. Its path is given in your task prompt.

## Taxonomy (cluster IDs)
- C1 Query-level LLM routing
- C2 Cascades, deferral and selective escalation
- C3 Step-level, role-level and subagent-level model assignment in agents and multi-agent systems
- C4 Automated optimization of compound AI system configurations
- C5 Online learning from live traffic (bandits, online RL, A/B, drift)
- C6 Offline learning from logs and customer data (OPE, counterfactual evaluation, performance prediction, reward models for routing)
- C7 Quality signals without ground truth (judges, PRMs, verifiers, uncertainty, agent failure detection)
- C8 Distillation and specialization using production data (agent distillation, small agents)
- C9 Reasoning and token budget control
- C10 Caching, context and prompt efficiency
- C11 Serving and systems for agentic workloads
- C12 Architecture-level cost reduction, and when multi-agent is worth it
- C13 Benchmarks and evaluation for cost-quality tradeoffs
- C14 Industry products, gateways and production case studies
- C15 Surveys and theory

Tag records with every cluster that fits, not only your own.

## Final message to the lead agent
Keep it short (under 400 words). Include:
- record counts per cluster you own, and the total;
- the top 10 items as id plus a one-line reason;
- any coverage gaps you noticed;
- the output file paths.
