# Taxonomy

Final topic taxonomy for `literature_db.json` (1,326 unique records). Records can carry several cluster tags, so the per-node counts add up to more than the total.

| ID | Node | Records tagged | Relevance 4–5 | Production evidence | Scope |
|---|---|---:|---:|---:|---|
| C1 | Query-level LLM routing | 213 | 146 | 17 | Pick one model per incoming request from a pool, using predicted quality and cost (classifiers, kNN, matrix factorization, preference-trained and RL routers). |
| C2 | Cascades, deferral and selective escalation | 141 | 94 | 10 | Run a cheap model first and escalate on low confidence, failed verification or disagreement; includes learning-to-defer and abstention. |
| C3 | Step-, role- and subagent-level model assignment | 315 | 218 | 13 | Choose models per turn, per action, per role or per subagent inside agent trajectories and multi-agent systems; includes topology and communication pruning. |
| C4 | Automated optimization of compound AI system configurations | 140 | 78 | 1 | Search over per-module model choice, prompts, workflow structure and hyperparameters, often jointly for cost and quality. |
| C5 | Online learning from live traffic | 120 | 64 | 21 | Bandits, online RL, budget pacing, drift handling, A/B tests and implicit feedback for routing and config choice. |
| C6 | Offline learning from logs and customer data | 144 | 63 | 22 | Off-policy evaluation, counterfactual learning from logged outcomes, performance prediction and reward models for routing. |
| C7 | Quality signals without ground truth | 229 | 79 | 7 | LLM judges, process and outcome reward models, verifiers, uncertainty estimates, probes and agent failure attribution. |
| C8 | Distillation and specialization | 182 | 62 | 13 | Training small models on agent traces or production data, SFT and RL for small agents, task-specific specialists. |
| C9 | Reasoning and token budget control | 158 | 65 | 5 | Adaptive reasoning effort, thinking budgets, overthinking, think/no-think routing and test-time compute allocation. |
| C10 | Caching, context and prompt efficiency | 174 | 57 | 18 | Prompt and KV caching, semantic caches, plan caching, context compression and agent memory or context management. |
| C11 | Serving and systems for agentic workloads | 158 | 44 | 10 | Agent- and program-aware scheduling, prefix sharing, speculative decoding, multi-model and multi-adapter serving. |
| C12 | Architecture-level cost reduction and when multi-agent is worth it | 158 | 94 | 9 | Single- vs multi-agent comparisons, orchestration patterns, token-consumption analyses and cost-aware agent evaluation. |
| C13 | Benchmarks and evaluation for cost-quality tradeoffs | 188 | 83 | 13 | Routing benchmarks, agent benchmarks with cost reporting, reliability metrics (pass^k) and Pareto evaluation. |
| C14 | Industry products, gateways and production case studies | 162 | 70 | 72 | Routers and gateways sold as products, provider docs, and company reports of cost reduction on real traffic. |
| C15 | Surveys and theory | 80 | 19 | 0 | Surveys of routing, efficient inference, efficient agents and reasoning; formal treatments of routing as optimization or stopping. |

## Cross-cutting facets

Every record is also classified along three facets that cut across clusters.

### Routing granularity

| Value | Records |
|---|---:|
| none | 406 |
| query | 368 |
| step | 249 |
| subagent | 134 |
| n/a | 134 |
| token | 35 |

### Learning regime

| Value | Records |
|---|---:|
| n/a | 365 |
| supervised-offline | 344 |
| training-free | 342 |
| RL-offline | 86 |
| bandit-online | 78 |
| hybrid | 73 |
| RL-online | 35 |
| null | 3 |

### Source type

| Value | Records |
|---|---:|
| preprint | 870 |
| paper | 330 |
| blog | 64 |
| docs | 34 |
| product | 12 |
| benchmark | 8 |
| repo | 8 |

### Relevance score

| Score | Records |
|---|---:|
| 5 | 167 |
| 4 | 357 |
| 3 | 518 |
| 2 | 284 |

### Step-level and subagent-level work by learning regime

This facet matters most for the premise, so it is broken out here.

| Learning regime | step | subagent |
|---|---:|---:|
| training-free | 103 | 51 |
| supervised-offline | 59 | 26 |
| RL-offline | 13 | 12 |
| bandit-online | 7 | 5 |
| RL-online | 10 | 2 |
| hybrid | 22 | 8 |
| n/a | 33 | 30 |

### Year

| Year | Records |
|---|---:|
| n.d. | 27 |
| 2026 | 516 |
| 2025 | 416 |
| 2024 | 222 |
| 2023 | 104 |
| 2022 | 16 |
| 2021 | 3 |
| 2020 | 5 |
| 2019 | 1 |
| 2018 | 2 |
| 2017 | 2 |
| 2015 | 3 |
| 2014 | 1 |
| 2013 | 2 |
| 2012 | 2 |
| 2011 | 2 |
| 2010 | 2 |
