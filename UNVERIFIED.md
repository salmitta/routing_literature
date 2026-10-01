# Unverified and excluded items

These items were seen during search but are **not** in `literature_db.json`. Each one either couldn't be verified to the standard the brief requires (a resolving source with title, authors, year and an abstract or full text), or was removed after verification. They are listed so they can be chased by hand.

## Seeds

Every seed in the brief was found and verified, with one exception: SPROUT has no paper of its own. It is the dataset released with the CARROT paper, which is in the database as `somerstep-2025-carrot`.

## Removed after verification

| arXiv ID | Reason |
|---|---|
| [2605.03195](https://arxiv.org/abs/2605.03195) | withdrawn on arXiv (abs-page banner) |
| [2607.19338](https://arxiv.org/abs/2607.19338) | withdrawn on arXiv (abs-page banner) |

## Candidate papers that could not be verified

| Item | What was found | Found by |
|---|---|---|
| Stage-aware LLM-SLM collaboration for agentic tasks via joint routing and verification (Yang, Tang, Li, Guo, Liu, Qu, Wen; Information Processing & Management, DOI 10.1016/j.ipm.2026.105065) | Found as a cited-by of yue-2025-masrouter on Semantic Scholar; title, authors and DOI verified, but no abstract was available (S2 batch and citation records lacked it), so it could not be summarized. Highly relevant to C3 step-level LLM-SLM routing; worth a manual fetch. | g10 |
| Learning from Naturally Occurring Feedback (Don-Yehiya et al., implicit feedback in chat logs) | OpenAlex title search returned no match; arXiv API rate-limited (429) and WebSearch budget exhausted, so could not confirm ID/authors. Left out. | g5 |
| Uncertainty-Aware Adaptive Routing for Large Language Models using Constrained Thompson Sampling (Patel et al., 2026) | Found only as Zenodo/Research Square deposits via OpenAlex; no peer-reviewed or arXiv version seen. Excluded on rigor grounds. | g5 |
| Contextual Bandit Routing for LoRA Experts | Listed in Semantic Scholar cited-by of C2MAB-V with no year or ID; not verifiable. Left out. | g5 |
| Coactive Learning for Large Language Models using Implicit User Feedback (Tucker, Brantley, Cahall, Joachims; ICML 2024) | Found in Semantic Scholar search (DBLP conf/icml/TuckerBCJ24) but the record has no arXiv ID, DOI or abstract; arXiv title search returned nothing. Highly relevant to implicit-feedback learning; needs a PMLR link. | g9 |
| CoCoMaMa: Contextual Combinatorial Multi-Armed Bandit Router for Multi-Agent Systems with Volatile Arms (Rau, Bader, Wiesner, Kao; HyperAgents@ECAI 2025) | Seen in the Semantic Scholar reference list of li-2025-llm-bandit (2502.02743). No arXiv ID, DOI or abstract; arXiv title search returned nothing. Likely a relevance-4 C3/C5 item. | g9 |
| Learning to Cascade: A POMDP Approach to Sequential Model Selection (Salehi Darjani, Zois; ICASSP 2026) | Found via S2 citations of a seed; DOI 10.1109/icassp55912.2026.11464865 exists, but no abstract was available to summarize, so it is left out. | g9 |
| From Metrics to Meaning: Estimating User Feedback Using LLM-Based Evaluation | Unlinked reference string in a seed's S2 reference list; title search on arXiv found no match. | g9 |
| Continuous feedback loops: Online fine-tuning of LLMs with user signals (2025) | S2 search hit with no arXiv ID, DOI or abstract. | g9 |
| Whom to Trust? Analyzing the Divergence Between User Satisfaction and LLM-as-a-Judge in E-Commerce RAG Systems (2026) | S2 search hit with no arXiv ID or DOI; arXiv title search returned nothing. Relevant to implicit-feedback-versus-judge reliability. | g9 |

## Sources that couldn't be reached, and coverage notes

These notes record where a source was unreachable or the session's web-search quota ran out. They show where coverage is thinner than the brief asked for.

| Source | Note | Found by |
|---|---|---|
| HF papers search / OpenAlex coverage | Hugging Face papers search returned Internal Error and OpenAlex returned daily-budget-exceeded during this run; Semantic Scholar keyword search mostly returned 429. Searches relied on arXiv API plus S2 citation snowballing. | g10 |
| S2 cited-by for paul-2026-agentrouter-step-tiers (2609.22951) and ganz-2026-handoff-tax (2608.24358) | Semantic Scholar returned 0 citing papers for both (too recent); no snowball possible. | g10 |
| Google Cloud Vertex AI Model Optimizer | Old docs URLs (cloud.google.com/vertex-ai/generative-ai/docs/model-reference/vertex-ai-model-optimizer and variants) now redirect to the generic Gemini Enterprise Agent Platform models overview or return 404; no current optimizer/router page found in its link list. Added the data-driven prompt optimizer docs instead. | g11 |
| Predibase case studies (e.g., LoRA Land blog, Checkr) | predibase.com now redirects to rubrik.com, which returns 403 Access Denied to curl. LoRA Land paper already in index (zhao-2024-lora-land). | g11 |
| OpenPipe case studies / blog | openpipe.ai/blog, /case-studies and /blog/art-e-mail-agent return an empty JS shell (title 'OpenPipe', no content). ART already in index (openpipe-2025-art). | g11 |
| Modal blog cost posts | modal.com/blog index only exposed /blog/how-to-price-serverless and the atom feed; no substantive LLM cost post found without search. | g11 |
| Baseten blog cost posts | Index showed only product/news posts; Baseten Loops posts lacked numbers and were folded into gola-2026-langsmith-fine-tuning. | g11 |
| Anyscale blog (LLM router) | anyscale.com/blog renders empty to curl; the Anyscale router post is already in the index (almahairi-2024-anyscale-router). | g11 |
| Hugging Face blog routing/cost posts | Current blog index had no routing, caching or distillation-for-cost posts; no older URL known without search. | g11 |
| OpenAI API model distillation announcement (openai.com/index/api-model-distillation) | openai.com returns 403 (JS/cookie challenge) to curl; used developers.openai.com model-optimization track instead. | g11 |
| Skipped as duplicates/companions (not unverified) | LMSYS RouteLLM blog (paper ong-2024 in index, no production data); Google Research speculative cascades blog (narasimhan-2024 in index); LMSYS SGLang/RadixAttention blog (zheng-2023-sglang in index); vLLM signal-decision blog (companion to liu-2026-vllm-semantic-router); AWS Bedrock distillation blogs (aws-bedrock-model-distillation in index); Anthropic context engineering, Manus, GPTCache, LiteLLM router, Portkey, Martian docs, NVIDIA router blueprint, Azure model router, Bedrock prompt routing already in index. | g11 |
| Industry blogs on A/B testing, shadow deployment and online routing (Martian, Not Diamond, OpenRouter, LMSYS RouteLLM blog) | WebSearch budget for the session was exhausted (200/200) before blog searches; only academic APIs (OpenAlex, arXiv, Semantic Scholar via WebFetch) were usable. Not searched. | g5 |
| OpenAI 'Prompt Caching in the API' announcement page (openai.com/index/api-prompt-caching/) | Seen in web search results but WebFetch returned HTTP 403, so its launch-date and discount figures could not be checked. The OpenAI prompt-caching docs page was fetched and verified, and it is recorded as openai-2024-prompt-caching. All other C10/C11 seeds were verified via arXiv abs pages or official pages. | g7 |
| DoorDash LLM-based Dasher support automation blog | Known URL careersatdoordash.com/blog/large-language-modules-based-dasher-support-automation/ returned HTTP 403 to both curl and WebFetch; content not seen, so excluded. | g8 |
| Martian model router product page | withmartian.com now shows interpretability research; docs.withmartian.com describes only a Gateway API; /martian-model-router returned 404. Recorded docs page and RouterBench only. | g8 |
| Not Diamond as the engine behind OpenRouter auto-router | Current OpenRouter docs describe a market-spend-based router and do not mention Not Diamond; Not Diamond site quotes OpenRouter CEO. Relationship not confirmed. | g8 |
| Databricks blog posts on LLM routing/cost | WebSearch budget for the session was exhausted before searching; only Databricks AI Gateway docs verified. | g8 |
| Together AI blog on routing/cost (beyond Batch API) | WebSearch budget exhausted; only the Together Batch API post verified. | g8 |
| Unify routing-era blog/benchmarks | unify.ai has pivoted to continual post-training; no routing documentation found on current site. | g8 |
| Hugging Face posts on routing | WebSearch budget exhausted; only the EvalEval eval-costs post and LLM-Perf leaderboard verified. | g8 |
| Google Vertex AI Model Optimizer (auto routing) | Known docs URL redirected to a generic Agent Platform models overview; feature not confirmed. | g8 |
