# Part 4: Must-read top 30

The ranking is by usefulness for this premise: cutting the cost of a frontier-everywhere multi-agent pipeline using its own logs and traffic. It is not a ranking of general paper quality. "Vendor" means the numbers are company-reported from internal traffic.

| # | Record | Year · venue | Why read it |
|---|---|---|---|
| 1 | [okeefe-2026-cursor-router] | 2026 · Cursor blog (vendor) | The most complete production loop in the corpus. It learns a per-turn router from implicit user feedback, prices cache loss, gates frontier upgrades on confidence, and validates offline before going live. |
| 2 | [fireworks-2026-firerouter-with-opus] | 2026 · Fireworks blog (vendor) | The only randomized session-level live A/B of a cache-aware per-turn router: 57% (±19 points) lower cost per session at 98.1% of Opus-only accuracy. |
| 3 | [gonuguntla-2026-replay-gap] | 2026 · COLM | Replaying logged trajectories mispredicts the effect of swapping models mid-trajectory. Read it before evaluating any step router offline. |
| 4 | [kumar-2026-task-session-routing-eval] | 2026 · arXiv | Tested open-source routers mostly emit near-constant tiers and don't beat content-blind allocation. Sets the share-matched control every router claim needs. |
| 5 | [li-2026-llmrouterbench] | 2026 · ACL | A unified re-evaluation in which many routers, commercial ones included, barely beat simple baselines. |
| 6 | [paul-2026-agentrouter-step-tiers] | 2026 · ICML workshop | A small per-step tier classifier: 72% lower cost at 97.3% of frontier quality. It beats RouteLLM and FrugalGPT applied per step. |
| 7 | [hemadri-2026-r2v-agent] | 2026 · arXiv | Distil the small model first, then train a calibrated per-step router on its residual failures and escalate only when needed. |
| 8 | [ganz-2026-handoff-tax] | 2026 · arXiv | Escalating mid-trajectory recovers less than half of the quality gap at a cost premium, while downshifting pays. Changes how switching should be designed. |
| 9 | [kwartler-2026-harness-tokenomics] | 2026 · arXiv | Cache-aware routing at session, side-lane and subagent boundaries in an emulated enterprise, with the conditions under which a switch pays back. |
| 10 | [liu-2026-vllm-sr-session-aware-agentic-routing] | 2026 · vLLM blog | Session-aware routing with ablations: it puts a price on each switch and locks the model during tool loops. Open source. |
| 11 | [lin-2026-cursor-agent-swarm-economics] | 2026 · Cursor blog (vendor) | A frontier planner with cheap workers reached similar quality at $1,339, against $10,565 with one model in every role. Workers carry most tokens. |
| 12 | [cai-2026-think-big-search-small] | 2026 · arXiv | A controlled capacity sweep per role. Capacity belongs in the delegator, and a 1.7B distilled executor matches a frontier sub-agent. |
| 13 | [hua-2026-agentopt] | 2026 · arXiv | Model combinations differ 13–32x in cost at matched accuracy. Gives a cheap bandit-style search over role assignments. |
| 14 | [yue-2025-masrouter] | 2025 · ACL | The standard formulation of multi-agent-system routing: collaboration mode, roles and a model per agent. |
| 15 | [ye-2025-x-mas] | 2025 · arXiv | A large-scale study of which LLM suits which agent function. Heterogeneous multi-agent systems beat homogeneous ones. |
| 16 | [chen-2025-llmselector] | 2025 · arXiv | Treats model choice per module of a compound system as the decision variable. |
| 17 | [huang-2026-scope] | 2026 · KDD | Quality-constrained per-module model selection with guarantees: up to 6x lower final cost. |
| 18 | [su-2025-toolorchestra] | 2025 · arXiv | An RL-trained 8B orchestrator that calls models and tools beats GPT-5 on HLE, and on tau2-Bench and FRAMES at about 30% of the cost. |
| 19 | [yang-2026-ares-reasoning-effort] | 2026 · arXiv | A per-step reasoning-effort router for agents. The lowest-friction first lever, because it keeps the same model and cache. |
| 20 | [anthropic-2026-optimizing-cost-and-intelligence] | 2026 · Claude docs (vendor) | Measured trade-offs between caching, effort, retries and delegation, including where delegating to cheaper workers loses. |
| 21 | [guo-2026-routenlp] | 2026 · ACL Industry | A peer-reviewed closed loop of routing, conformal cascade and distillation, with an 8-week enterprise pilot: 58% lower cost at 91% acceptance. |
| 22 | [tsiourvas-2025-causal-routing] | 2025 · NeurIPS | Learns routing from logs in which only the deployed model's outcome is observed. |
| 23 | [khosravi-2026-which-llm-which-work] | 2026 · arXiv | Budgeted assignment of LLMs to work from confounded, proxy-scored production logs. Shows better quality information beats better optimisation. |
| 24 | [woisetschlager-2026-slarouter] | 2026 · arXiv | Online cost-optimal routing from sparse, one-sided feedback with SLA guarantees: up to 2.2x lower cost. |
| 25 | [taberner-miller-2026-paretobandit] | 2026 · arXiv | Budget pacing, forgetting for drift, and hot-swapping new models in one open-source bandit router. |
| 26 | [panda-2025-pilot] | 2025 · EMNLP Findings | The reference warm-start-then-bandit design: LinUCB initialised from preference data, with a budget policy. |
| 27 | [ong-2024-routellm] | 2024 · arXiv (ICLR 2025 per authors) | The standard query-level router baseline trained on preference data. Start here for the method family. |
| 28 | [dekoninck-2024-cascade-routing] | 2024 · ICML (per Semantic Scholar) | A unified, optimal treatment of routing plus cascading. Finds that the quality estimator is the bottleneck. |
| 29 | [kim-2025-science-of-scaling-agent-systems] | 2025 · arXiv | Controlled evidence that multi-agent gains depend on the task, ranging from +80.8% to −70.0% against a single agent. |
| 30 | [rajput-2026-cheap-verifiers-blind-spots] | 2026 · arXiv | Cascades monitored by their own cheap verifier can hide degradation. Read it before building any closed loop. |

# Part 5: Gaps where no good work exists

Several cluster drafts flagged the same gaps independently. They are consolidated here, starting with the most consequential for this premise.

1. **No independent randomized production study of step-level routing in an enterprise multi-agent pipeline outside coding.**
   - Every live A/B is vendor-reported and on coding agents [fireworks-2026-firerouter-with-opus] [okeefe-2026-cursor-router].
   - Peer-reviewed industry evidence covers query-level or narrow workflows [guo-2026-routenlp].
   - Enterprise multi-agent evaluations are small: 54 tasks [singh-2026-planner-as-router] and 22 tasks [ali-2026-harness-effect].
   - Nothing exists for ITSM, CRM or enterprise-workflow multi-agent systems [drouin-2024-workarena].
2. **No validated off-policy evaluation for step-level model swaps.**
   - Replay fails [gonuguntla-2026-replay-gap].
   - World-model OPE for agents has not been checked against live outcomes [liu-2026-adwm-ope-llm-agents].
   - No method trains step routers on counterfactual, branched rollouts at scale.
3. **No method learns a router from frontier-only logs without collecting new data.** Causal routing assumes some variation in outcomes across models [tsiourvas-2025-causal-routing], and a frontier-everywhere log has none. No paper gives a principled design for this cold start: how much to re-run, which inputs, and with what exploration.
4. **No online bandits with safety guarantees at step level inside multi-agent graphs, evaluated on live traffic.** Sequential and multi-step bandits exist only in simulation [atalar-2025-sequential-bandits] [guan-2026-symphony-coord].
5. **No compute-matched comparison on the same pipeline** of heterogeneous step routing against a single frontier model at tuned effort. Vendor docs come closest [anthropic-2026-optimizing-cost-and-intelligence].
6. **Little on the costs of switching models in multi-agent systems.**
   - Handoff tax and drift are measured only for single-agent coding and conversation [ganz-2026-handoff-tax] [khraishi-2026-model-switching-drift]. Their interaction with role specialization is unstudied.
   - No work on whether a cheaper model can continue reliably from compacted or compressed context [chen-2026-governance-decay].
7. **No reuse of the provider-API prompt cache across models.** Cross-model KV reuse exists only for self-hosted serving [woo-2026-prefillshare]. This makes every provider-side switch pay a full prefill.
8. **No per-subagent calibration that survives a model swap, and no step-level credit assignment good enough to decide where a cheaper model is safe.** Audits of early outcome prediction fail across swaps [cao-2026-early-outcome-prediction-audit]. Failure attribution exists but is not yet used for routing [zhang-2025-which-agent-when] [zhang-2026-credit-without-ground-truth].
9. **No joint optimisation of effort, model tier, distillation and budget under one objective.** Existing work covers only parts of it [huang-2026-uniscale] [guo-2026-routenlp].
10. **No standard way to account for cost.**
    - Few comparisons include cache effects and the serving regime [weinberger-2026-token-reduction-not-cost] [dong-2026-learning-to-stop-cost-study].
    - There is no per-role cost-attribution method for heterogeneous multi-agent systems [singh-2026-total-cost-of-agency].
    - The cost of search, labelling and re-optimisation is rarely set against the savings [el-2025-inefficiencies-meta-agents] [lai-2026-saverouter].
11. **Enterprise agent benchmarks lack cost reporting.** They rarely report cost-of-pass or pass^k [jha-2025-itbench] [huang-2025-crmarena-pro] [erol-2025-cost-of-pass]. No benchmark combines heterogeneous roles, cache pricing and live switching without replay [jiang-2026-agentcard] [notdiamond-2026-interactive-benchmarks-routing].
12. **No independent replication of vendor router claims.** The independent re-evaluations that exist are negative for the routers they tested [li-2026-llmrouterbench] [kumar-2026-task-session-routing-eval].
13. **No validated false-hit rates for semantic caches on enterprise agent traffic.** At standard thresholds, most cache "hits" are not valid substitutes [kulkarni-2026-cache-eviction-study] [baral-2026-semantic-cache-operational-gap].
