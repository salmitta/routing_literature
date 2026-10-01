# Synthesis: cutting the cost of production multi-agent LLM systems

**Scope.** This synthesis looks at how to lower the cost of an agentic pipeline in which every subagent and step calls one frontier model, without losing task quality. The cost levers covered are:
- query-level routing;
- step- and subagent-level routing or downscaling;
- learning configurations from offline logs or live traffic;
- supporting levers: caching, context management, distillation, reasoning budgets and serving.

**Source.** Everything here is written from the merged database only: `literature_db.json`, 1,326 unique records. Claims cite record IDs in square brackets. Every number in the tradeoff tables appears in the cited record's own fields, and an automated check confirmed this for every table row. Records whose arXiv versions were withdrawn have been removed.

**How to read the numbers.**
- Cost/quality results are copied as reported. They can be compared only within the same benchmark, baseline and price sheet.
- Vendor blog and product numbers are marked as vendor-reported. They come from internal traffic and are not independently replicated.
- A large share of the step-level and online-learning work is unreviewed 2026 preprints. Treat these as hypotheses to test, not settled results.

**Coverage.** 1,326 records: 524 at relevance 4–5, 92 with production evidence, 249 step-level and 134 subagent-level. Records per cluster: C1 213, C2 141, C3 315, C4 140, C5 120, C6 144, C7 229, C8 182, C9 158, C10 174, C11 158, C12 158, C13 188, C14 162, C15 80. Full counts are in `taxonomy.md`.

## Contents
- Part 1: Topic clusters C1–C15
- Part 2: Approaches applicable to step-level routing in multi-agent pipelines
- Part 3: Approaches for learning routing/config policies from offline logs vs. online traffic
- Part 4: Must-read top 30
- Part 5: Gaps where no good work exists

## Executive summary

1. **Lower cost is achievable at the levels below, but the strongest evidence is vendor-reported and from coding agents.**
   - The two most complete production results are a per-turn router learned from implicit user feedback and validated on live traffic [okeefe-2026-cursor-router], and a randomized session-level A/B test of a cache-aware per-turn router [fireworks-2026-firerouter-with-opus]. The A/B reported 57% (±19 points) lower cost per session at 98.1% of Opus-only accuracy.
   - A peer-reviewed industry pilot reports 58% lower inference cost at 91% response acceptance over 8 weeks [guo-2026-routenlp].
2. **Beat the cheap baselines before claiming a gain.**
   - Independent re-evaluations find that many routers, including commercial ones, do not beat simple baselines [li-2026-llmrouterbench].
   - Tested open-source routers did no better than content-blind allocation at matched tier share [kumar-2026-task-session-routing-eval].
   - Tuning reasoning effort and prompt caching on the existing frontier model is a strong first baseline [anthropic-2026-optimizing-cost-and-intelligence] [yang-2026-ares-reasoning-effort].
3. **Mid-trajectory model switching has costs that query-level routing ignores.** Switching breaks the prompt cache [kwartler-2026-harness-tokenomics] [liu-2026-vllm-sr-session-aware-agentic-routing]. Escalating mid-trajectory pays a handoff tax [ganz-2026-handoff-tax], and model pairs drift [khraishi-2026-model-switching-drift]. Replaying logged trajectories mispredicts the effect of a swap [gonuguntla-2026-replay-gap]. The practical default is:
   - assign models per role or subagent at launch;
   - switch only at safe boundaries;
   - prefer downshifting over escalating.
4. **Per-role assignment is the highest-yield structural lever.**
   - At matched accuracy, model combinations differ 13–32x in cost [hua-2026-agentopt].
   - Mixed-model teams beat single-model ones [ye-2025-x-mas] [jiang-2026-agentcard].
   - Capacity belongs in the planner or delegator more than in executors [cai-2026-think-big-search-small] [lin-2026-cursor-agent-swarm-economics].
5. **Learning from your own logs is possible but causally delicate.**
   - A log in which one frontier model was always used contains no counterfactual outcomes, so some re-running or exploration is unavoidable.
   - Once routing exists, the logs are confounded by it [tsiourvas-2025-causal-routing].
   - The best-documented path runs: offline warm start, then shadow and randomized A/B, then online bandit refinement with budget pacing and drift handling [okeefe-2026-cursor-router] [bao-2026-orcarouter] [taberner-miller-2026-paretobandit] [woisetschlager-2026-slarouter].

# Part 1: Topic clusters

## C1. Query-level LLM routing

### Key ideas

- **Pre-generation predictors over a model pool.** The core line of work trains a lightweight predictor that estimates each candidate's quality (and sometimes cost) from the query alone, then picks a model through a threshold or utility rule. Supervision has come from human preference data [ong-2024-routellm], [frick-2025-p2l], benchmark correctness tables [shnitzer-2023-routing-benchmark-datasets], [hu-2024-routerbench], distilled reward-model scores [lu-2023-zooter], and judge labels [almahairi-2024-anyscale-router]. Model representations range from contrastive embeddings [chen-2024-routerdc], [shirkavand-2025-cscr] and matrix factorisation [zhuang-2024-embedllm] to IRT ability/difficulty parameters [song-2025-irt-router], [fernandez-2025-radar] and graph edge prediction [feng-2024-graphrouter]. CARROT shows that estimating cost and accuracy separately, then combining them in a linear utility, is minimax rate-optimal [somerstep-2025-carrot].
- **Onboarding new models without retraining.** Several methods describe each model by its behaviour on a small set of anchor or profiling prompts, so a new model can be added by profiling it rather than retraining the router [jitkrittum-2025-unirouter], [wang-2025-icl-router], [zhang-2025-model-sat], [feng-2024-graphrouter].
- **Internal-state routers.** Probes on pre-generation or prefill activations predict whether a given policy will succeed [lugoloobi-2026-llms-encode-failures], [cencerrado-2025-no-answer-needed], [varshney-2026-prefill-activation-router].
- **Learning online under partial feedback.** Contextual bandits (LinUCB, NeuralUCB, Thompson sampling, dueling bandits) learn from the chosen model's outcome only [panda-2025-pilot], [wei-2025-barp], [tsai-2026-neuralucb-routing], [chiang-2025-dueling-routing], [wang-2025-mixllm]. Extensions add budget pacing and drift [taberner-miller-2026-paretobandit], [lee-2026-drift-aware-routing], [zu-2026-barouter], SLA guarantees [woisetschlager-2025-mess-plus], [woisetschlager-2026-slarouter], and implicit signals such as user retrials [bae-2026-contextual-queueing-bandits]. A separate approach learns from logged observational data by minimising regret [tsiourvas-2025-causal-routing].
- **Wider action spaces.** Routers now choose over (model, test-time samples) [ding-2025-best-route], (model, reasoning strategy or budget) [pan-2025-route-to-reason], [xue-2026-r2-router], think/no-think modes [wang-2025-when-to-reason], [zhang-2025-synapseroute], (model, harness) pairs [chen-2026-hm-router], and agent paradigms [zhou-2026-select-then-solve].
- **From query to session and step.** The 2026 work moves routing inside agents: per tool call [agarwal-2026-switchcraft], per turn with switch and cache costs priced in [okeefe-2026-cursor-router], [fireworks-2026-firerouter-with-opus], [liu-2026-vllm-sr-session-aware-agentic-routing], [notdiamond-2026-interactive-benchmarks-routing], [kwartler-2026-harness-tokenomics], multi-turn sequential routing [zhang-2026-mtrouter], [xu-2026-seqroute], [zhang-2026-dialrouter], and per-step (role, backbone) choices in multi-agent systems [feng-2026-graphplanner], [yue-2025-masrouter], [cui-2026-uno-orchestra].

### Strongest methods

- **Cursor Router** [okeefe-2026-cursor-router] (vendor-reported) has the most complete pipeline from offline training to online validation. It trains on hundreds of thousands of production turns labelled with implicit satisfaction and real cost, including cache misses caused by switching. A frontier model is admitted for a task label only if it beats the cheap model with 75% one-sided confidence. Candidate policies are checked on held-out logs and then on live traffic. The design choices that matter: the label is implicit user correction, the cost includes cache loss, and upgrades are gated on statistical confidence.
- **FireRouter** [fireworks-2026-firerouter-with-opus] is the only randomised live A/B test in this cluster (sessions randomised, more than a month). It prices lost prompt-cache value into each per-turn decision. The numbers are vendor-reported.
- **Session-aware routing** [liu-2026-vllm-sr-session-aware-agentic-routing] is open-source and includes ablations. Its parts are hard locks during tool loops, safe reset boundaries and prefix-cache-aware switch pricing. Sticky sessions alone save more but lose more quality.
- **IPR** [feng-2025-ipr] is a peer-reviewed industry-track paper on a deployed system. It uses frozen encoders with model-specific adapters and a user tolerance knob.
- **RouteNLP** [guo-2026-routenlp] closes the loop between routing, conformal cascading and distillation on escalation failures, and reports an 8-week enterprise pilot.
- **kNN and simple estimators.** A well-tuned kNN matches complex routers [li-2025-knn-routing]. CARROT supplies theory for the simple separate-estimator design [somerstep-2025-carrot]. Quality-estimator accuracy is the main factor in whether any selection scheme succeeds [dekoninck-2024-cascade-routing].
- **Bandit routers with budget control** [panda-2025-pilot], [taberner-miller-2026-paretobandit], [bao-2026-orcarouter]. Warm-starting offline and then updating with LinUCB is the common production-oriented pattern.

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| RouteLLM | Blog: >85% cost reduction on MT-Bench, 45% on MMLU, 35% on GSM8K vs GPT-4 at 95% of GPT-4 quality | MT-Bench, MMLU, GSM8K | GPT-4 | [ong-2024-routellm] |
| IPR (deployed) | 43.9% cost reduction at quality parity with the strongest Claude-family model, sub-150ms latency | IPRBench (1.5M prompts, 11 LLMs) | strongest Claude-family model | [feng-2025-ipr] |
| Cursor Auto Balance (vendor) | outperforms Opus 4.8 at 41% lower cost with +3% satisfaction | live Cursor traffic | Opus 4.8 | [okeefe-2026-cursor-router] |
| FireRouter with Opus (vendor, A/B) | cost per session $15.36 -> $6.63 (57% +/- 19 ppts lower); graded-turn accuracy 78.7% vs 80.2% (98.1% of Opus-only) | internal coding traffic A/B test | Opus-only | [fireworks-2026-firerouter-with-opus] |
| Per-task oracle over 18 models (vendor) | 97.6% pass@1 at $1.88/task vs best single model GPT-6 Astra 74.1% at $6.52 | DeepSWE v1.1 | best single model | [fireworks-2026-frontier-is-a-router] |
| RouteNLP pilot | 58% lower inference cost at 91% response acceptance, p99 latency 1,847 ms to 387 ms | 8-week enterprise pilot (~5K queries/day) | not stated | [guo-2026-routenlp] |
| Switchcraft | 82.9% accuracy, matching/exceeding best single model, with 84% lower inference cost | five function-calling benchmarks (incl. BFCL) | best single model | [agarwal-2026-switchcraft] |
| SAAR | 79.29% fewer model switches, 3,836 unsafe switches eliminated, 78.71% estimated physical-model cost reduction vs single-turn routing (quality delta -0.0453) | deterministic replay of 21,600 agent turns | single-turn routing | [liu-2026-vllm-sr-session-aware-agentic-routing] |
| Harness Tokenomics | Recovers 13-21% of model spend ($3.3M-$5.1M/year) in an emulated 10,000-seat enterprise at Anthropic list prices | ~10,000 real sessions from public datasets | not stated | [kwartler-2026-harness-tokenomics] |
| MTRouter | On ScienceWorld surpasses GPT-5 while reducing total cost by 58.7% | ScienceWorld | GPT-5 | [zhang-2026-mtrouter] |
| ToolOrchestra | on tau2-Bench and FRAMES surpasses GPT-5 using about 30% of the cost | tau2-Bench, FRAMES | GPT-5 | [su-2025-toolorchestra] |
| WebRouter | 87.8% lower operational cost vs GPT-4o baseline with 3.8% accuracy drop on five WebVoyager websites | WebVoyager | GPT-4o | [li-2025-webrouter] |
| Prefill-activation router | Closes 45.58% of the gap between best single model and oracle while saving 74.31% cost relative to the most expensive model | not stated | best single model / most expensive model | [varshney-2026-prefill-activation-router] |
| Bedrock IPR (vendor) | 56% savings for Anthropic family (Haiku 3.5 + Sonnet 3.5 v2) at Sonnet quality | internal AWS datasets incl. RAG | Sonnet | [aws-2025-bedrock-intelligent-prompt-routing] |
| EquiRouter | Reduces cost by about 17% at GPT-4-level performance vs strongest prior router on RouterBench | RouterBench | strongest prior router | [lai-2026-routing-collapse] |

### Critical notes

- **Routers often fail to beat simple baselines.** Under unified evaluation on 400K+ instances, many routers perform about the same, and several recent and commercial routers do not beat a simple baseline. A large oracle gap remains because routers fail to recall which model will succeed [li-2026-llmrouterbench]. Four open-source routers evaluated on agentic benchmarks mostly output near-constant tiers. Their gains follow tier composition, not task-specific targeting, so the authors found no evidence of advantage over content-blind allocation at matched shares [kumar-2026-task-session-routing-eval]. A tuned kNN matches complex learned routers [li-2025-knn-routing]. Most of the oracle gap is explained by task type [lee-2026-routing-gap-task-type]. Required controls: share-matched fixed tiers, kNN, and a task-type lookup.
- **Noise and oracle inflation.** 5.37% of model-question pairs flip between identical runs, which makes small router wins uninterpretable [lee-2026-routing-gap-task-type]. Under stochastic decoding, hindsight oracles are a different decision class from a router that commits to one model before seeing outputs, so they overstate achievable gains [chen-2026-routing-oracle-stochastic]. The vendor oracle study states that its oracle upside is biased upward [fireworks-2026-frontier-is-a-router]. Any policy that returns one member's answer is capped at one minus the all-models-fail rate [chen-2026-co-failure-ceiling].
- **Label artifacts.** Judge verbosity bias, truncation and format mismatch create apparent "unsolvable" items. They corrupt router labels and cause collapse to the majority class [garg-2026-unsolvability-ceiling]. Routers trained on scalar scores drift toward the most expensive model as budgets rise [lai-2026-routing-collapse].
- **Robustness and safety.** Preference-trained routers decide by category. For example, a BERT router sent all code and math to the strongest model and jailbreaks to weaker ones [kassem-2025-router-robustness]. Router vulnerabilities across the life cycle are catalogued in [lin-2025-router-vulnerabilities]. A negative-results study found classifier routing for reasoning is not feasible in all scenarios [srivatsa-2024-lessons-routing].
- **Attribution.** In Scrouting, an ablation shows the verified handoff from the cheap scout carries the gain, not the routing decision [bhola-2026-scrouting-superscout]. On agent routing, F1 drops 16-23 points as the catalog grows to 110 agents, and shortlisting recovers 10-17 [gillespie-2026-enterprise-agent-routing-scaling].
- **Comparability.** Most C1 headline numbers are abstract-level, use different baselines (GPT-4, GPT-5, random routing), and come from unreviewed 2026 arXiv preprints, for example [agarwal-2026-switchcraft], [cui-2026-uno-orchestra], [feng-2026-graphplanner]. Vendor production figures [okeefe-2026-cursor-router], [fireworks-2026-firerouter-with-opus], [notdiamond-2026-interactive-benchmarks-routing], [aws-2025-bedrock-intelligent-prompt-routing], [wood-2025-martian-up-and-to-the-left] are not independently reproducible. The Martian figures have no methodology [wood-2025-martian-up-and-to-the-left]. Bedrock's router is pairwise only, English and chat-optimised, and cannot be adapted to application data [aws-2025-bedrock-intelligent-prompt-routing]. Counting supervision cost moves break-even volume by ~1.9-9.5x [lai-2026-saverouter].

### Open problems

- Closing the model-recall gap to the oracle without overfitting to task type [li-2026-llmrouterbench], [lee-2026-routing-gap-task-type].
- Credit assignment for delayed agent rewards. Per-call routers versus task-level pinning [raj-2026-trace-router] versus per-step RL [zhang-2026-budget-aware-agentic-routing] has not been settled.
- Standard interactive or agentic router benchmarks that model cache expiry and session length [notdiamond-2026-interactive-benchmarks-routing], [yang-2026-twinrouterbench].
- Certifying routing gain before deployment [sun-2026-routeguard], [khosravi-2026-which-llm-which-work].
- Robustness to drift in provider quality, which is not predicted by price [he-2026-market-aware-routing].

## C2. Cascades, deferral and selective escalation

### Key ideas

- **Query-level cascades.** A cheap model answers, a deferral signal decides whether to escalate, and the final answer is taken from the stopping tier [chen-2023-frugalgpt], [aggarwal-2023-automix]. Deferral signals include token-level uncertainty [gupta-2024-lm-cascades-token-uncertainty], [ramirez-2024-two-tier], answer consistency [yue-2023-mot-cascades], ensemble or semantic agreement [kolawole-2024-agreement-based-cascading], [soiffer-2025-semantic-agreement-cascades], learned confidence tokens [chuang-2024-self-ref], and calibrated per-model confidence [zhang-2026-cauc], [kotte-2026-ucci].
- **Guarantees.** Conformal and risk-control methods bound accuracy or cost. Examples are conformal set size [dou-2026-conformal-cascade], cost-overrun control [valkanas-2025-c3po], provable human agreement for judge cascades [jung-2024-trust-or-escalate], and statistical guarantees for data processing [zeighami-2025-bargain], [shankar-2026-task-cascades], [patel-2024-lotus].
- **Unification with routing.** Cascade routing (route, then escalate) is shown to be optimal and beats either alone [dekoninck-2024-cascade-routing]. Learned variants use RL [huang-2026-rlcascaderouter] or cluster-then-escalate [moslem-2026-cluster-route-escalate].
- **Step-level escalation inside agents.** This is the most relevant line for multi-agent pipelines:
  - a cheap executor consults an expensive advisor at hard points [anthropic-2026-advisor-strategy], [kon-2026-swe-protege], [zeng-2026-learning-to-seek-help];
  - per-step uncertainty or calibrated-risk deferral [piatrashyn-2026-redact], [hemadri-2026-r2v-agent], [farzaneh-2026-tsds-edge-deferral], [lee-2025-steer];
  - escalation triggered by monitors or self-evaluation [wei-2026-step-level-cua-cascade], [gao-2026-agentcollab], [sun-2024-adaswitch];
  - permanent handoff timing [lei-2026-tacit-switch];
  - escalating only the critical reasoning steps [kapoor-2026-trim-stepwise-routing].
- **Verifier-gated cascades for code.** Escalation on failing tests [together-2026-deepswe-cascade-deepseek-gpt], [together-2026-deepswe-cascade-glm-flash], self-generated tests [chen-2024-model-cascading-code], and compile/lint diagnostics [abdollahi-2026-hdlforge].
- **Escalating the topology, not the model.** Moving from a single agent to debate or a multi-agent system only when needed [fan-2025-imad], [gao-2025-single-or-multi-agent-why-not-both], [chang-2026-cascadedebate].
- **Learning cascades from deferrals.** Online imitation [nie-2024-online-cascade-learning], student distillation [ramirez-2023-cache-and-distil], reusable strategies written by the strong model [wu-2025-inter-cascade], and distillation from failure clusters [guo-2026-routenlp].

### Strongest methods

- **Verifier-gated escalation where a cheap, reliable verifier exists.** For code, test-gated cascades beat both the cheap and the frontier model in accuracy and cost (vendor-reported, 904 rollouts) [together-2026-deepswe-cascade-deepseek-gpt]. The design choice that matters is an execution-grounded signal rather than self-confidence.
- **Calibrate first, then threshold.** UCCI's isotonic calibration followed by constrained cost minimisation reports a confidence interval on a 75,000-query production workload [kotte-2026-ucci]. Per-model calibration gives thresholds a shared scale [zhang-2026-cauc]. Uncertainty-correctness alignment is what matters most in 1500+ settings [chuang-2025-confident-or-seek-stronger].
- **Calibrated step-level risk routers trained on the student's residual failures** [hemadri-2026-r2v-agent] give tunable cost/reliability without retraining.
- **Distil one agent step plus LLM fallback.** Intercom replaced an LLM escalation decision with ModernBERT trained on 4M production examples, falling back to the LLM on uncertain cases, and validated it with an A/B test (company-reported) [yarullin-2025-intercom-escalation].
- **Advisor pattern** [anthropic-2026-advisor-strategy] (vendor). This is a training-free, executor-decided consultation within one API request.
- **Cascade routing theory** [dekoninck-2024-cascade-routing]. A pre-generation router beats the best cascade on 4 of 5 datasets [bouchard-2026-escalation-decision-theory], so whether to generate cheaply first should be decided per workload.

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| Advisor (vendor) | Sonnet executor + Opus advisor: +2.7 points on SWE-bench Multilingual over Sonnet alone with 11.9% lower cost per agentic task | SWE-bench Multilingual | Sonnet alone | [anthropic-2026-advisor-strategy] |
| Test-gated cascade (vendor) | Cascade (DeepSeek V4 Pro first, escalate to GPT-5.6 Sol on test failure): 83.0% of DeepSWE tasks at $3.35/task vs Sol alone 72.7% at $8.37 | DeepSWE | GPT-5.6 Sol alone | [together-2026-deepswe-cascade-deepseek-gpt] |
| Distilled-to-parent cascade (vendor) | 80.9% of DeepSWE tasks at $1.70 vs GLM-5.3 alone 69.0% at $3.99 (57% lower cost) | DeepSWE | GLM-5.3 alone | [together-2026-deepswe-cascade-glm-flash] |
| R2V-Agent | 93.3% TerminalBench at 33.9% LLM calls, about half heuristic-router cost | TerminalBench | heuristic router | [hemadri-2026-r2v-agent] |
| SWE-Protégé | Qwen2.5-Coder-7B reaches 42.4% Pass@1 on SWE-bench Verified (+25.4% over prior SLM SOTA) using ~4 expert calls per task and 11% of tokens | SWE-bench Verified | prior SLM SOTA | [kon-2026-swe-protege] |
| Hera | 92.5% of cloud-only success with cloud use in only 46.3% of steps | ALFWorld, WebShop, AppWorld | cloud-only | [zhang-2026-hera-device-cloud] |
| ReDAct | Deferring about 15% of decisions to the large model matches exclusive large-model quality at significantly lower cost | ALFWorld, MiniGrid | exclusive large model | [piatrashyn-2026-redact] |
| TRIM | advanced policies match the strong model using 80% fewer expensive-model tokens | MATH-500 | strong model | [kapoor-2026-trim-stepwise-routing] |
| UCCI | Cuts inference cost by 31% (95% CI 27-35%) at micro-F1 0.91 on a production NER workload of 75,000 queries (4B/12B LLMs) | production NER workload | not stated | [kotte-2026-ucci] |
| Intercom escalation (company) | custom ModernBERT handles 90% of cases with LLM fallback for 10%; >98% escalation accuracy; latency -0.5s; cost per resolution ~-3% | production A/B test | LLM escalation decision | [yarullin-2025-intercom-escalation] |
| RLM-Cascade | 88.8% draft-use over 125 production requests, 45.8% lower API cost vs direct Opus, p50 latency 2,026 ms vs 3,698 ms | 125 production Claude Code requests | direct Opus | [wu-2026-rlm-cascade] |
| Post-retrieval cascade | +0.140 Composite Overall vs Always-HyDE, 31.8% lower latency, 72.2% of real queries served without LLM augmentation | Danish National Encyclopedia production traffic (20,000 query-workflow pairs) | Always-HyDE | [hussain-2026-coverage-illusion-rag] |
| BudgetMLAgent | 94.2% cost reduction ($0.931 to $0.054 per run) with 32.95% vs 22.72% success rate compared to GPT-4 single agent | MLAgentBench | GPT-4 single agent | [gandhi-2024-budgetmlagent] |
| Self-improving cascade audit | cascade dashboard shows flat 3% error while true error swings up to 32% | MATH | not stated | [rajput-2026-cheap-verifiers-blind-spots] |

### Critical notes

- **In-loop metrics can hide degradation.** Measured through a cheap verifier, cascade error looked flat while true error swung widely. Verifier blind spots grew with student capability, and corrective fine-tuning collapsed the student [rajput-2026-cheap-verifiers-blind-spots]. Production cascades need gold-labelled audit samples.
- **Escalation can hurt.** A large model can break a correct small-model answer, so routing on uncertainty alone is not enough [wang-2026-signed-rescue-routing]; that preprint's abstract still contains placeholder numbers. Frontier-model failures more often break tests that were already passing [together-2026-deepswe-cascade-deepseek-gpt]. Passing small-model reasoning upward can help or hurt [khaled-2026-pyramid-moa].
- **Predicting which escalation pays is hard.** Probes predict failure well (0.88 AUROC) but struggle to choose which collaboration protocol helps, and routers under- or over-escalate [yang-2026-protocol-routing-cost-aware]. Fixed escalation beat a resampling selector in a small MBPP+ study [chen-2026-resample-or-reroute]. Tuning the initial budget is often cheaper than selective verification [dip-2026-sevra-selective-verification].
- **Cascades are not always best.** Long fixed chains underperform the pairwise envelope, and a pre-generation router beats the best cascade on 4 of 5 datasets [bouchard-2026-escalation-decision-theory]. Simple baselines such as retry and escalation sit on the agent cost-accuracy frontier [kapoor-2024-ai-agents-that-matter].
- **Signal pitfalls.** Sequence-level uncertainty has a length bias [gupta-2024-lm-cascades-token-uncertainty]. Confidence-based deferral breaks under perturbation [mitra-2026-cascades-under-perturbations], and cascades widen the adversarial attack surface [sun-2026-cascade-adversarial-attack].
- **Evaluation realism.** Synthetic queries overstated the need for LLM augmentation compared with real traffic (90% vs 27.8%) [hussain-2026-coverage-illusion-rag]. Several figures are projected rather than measured [liu-2026-adaptive-vlm-routing-cua], [basu-2026-w5h2-intent-caching]. RLM-Cascade's production sample is 125 requests [wu-2026-rlm-cascade]. Agent-step deferral papers from 2026 are mostly unreviewed arXiv preprints [piatrashyn-2026-redact], [hemadri-2026-r2v-agent], [lei-2026-tacit-switch]. The advisor, Together and Intercom numbers are vendor-reported [anthropic-2026-advisor-strategy], [together-2026-deepswe-cascade-deepseek-gpt], [yarullin-2025-intercom-escalation].
- **Handoff cost.** Restarting a task on the stronger model wastes progress. Completing only the residual keeps accuracy at 22.0%-34.6% of full-task cost [deng-2026-cfrc-residual-completion-handoffs].

### Open problems

- Deferral signals for open-ended agent steps that lack a test oracle [kirchhof-2025-uq-agents-position], [mehta-2026-behavioral-consistency].
- Harm-aware escalation, which predicts both rescue and breakage [wang-2026-signed-rescue-routing].
- Keeping self-improving cascades auditable over time [rajput-2026-cheap-verifiers-blind-spots], [guo-2026-routenlp].
- Choosing which agent role should be strong. One preliminary finding is that putting the large model on execution rather than planning gave higher accuracy [gao-2026-agentcollab].
- Calibrating thresholds with very few labels [zellinger-2025-rational-tuning-cascades].

## C3. Step-, role- and subagent-level model assignment in agents and multi-agent systems

### Key ideas

**Per-step and per-turn routers inside single-agent loops.** The most direct answer to "one frontier model for every step" is a lightweight classifier that picks a model tier for each tool call, step or user turn. AgentRouter maps each trajectory step to one of four tiers using a 12M-parameter classifier on five step features, trained on 50,000 annotated steps [paul-2026-agentrouter-step-tiers]. Switchcraft uses a DistilBERT classifier for each tool call [agarwal-2026-switchcraft]. A LightGBM router conditioned on harness state also logs every decision as a training record, so the router improves from its own traffic [liu-2026-harness-native-routing]. MTRouter learns a turn-level utility from logged trajectories using joint history-model embeddings, and it switches models less often than earlier multi-turn routers [zhang-2026-mtrouter]. Ares routes reasoning *effort* rather than model, and is trained on labels of the minimum effort that still succeeds [yang-2026-ares-reasoning-effort]. CogRouter learns a per-step cognitive depth [yang-2026-cogrouter]. Production systems now do this at turn level and report live-traffic results: Cursor [okeefe-2026-cursor-router], Fireworks [fireworks-2026-firerouter-with-opus], Not Diamond [notdiamond-2026-interactive-benchmarks-routing] and GitHub Copilot [github-copilot-auto-model-selection]. All of them price the prompt-cache loss that a model switch causes.

**Uncertainty-triggered escalation and deferral within trajectories.** In these designs a cheap model runs by default and hands individual decisions to a stronger model when a signal fires. The signals vary:
- Token-level predictive uncertainty: ReDAct [piatrashyn-2026-redact].
- Perplexity combined with a convergence probe, with thresholds calibrated by Learn-Then-Test for finite-sample guarantees on reward and cloud-call rate: TSDS [farzaneh-2026-tsds-edge-deferral].
- Brier-calibrated failure risk under a CVaR objective: R2V [hemadri-2026-r2v-agent].
- Self-reflection on whether the agent is making progress: AgentCollab [gao-2026-agentcollab].
- Learned stall and drift monitors in GUI agents [wei-2026-step-level-cua-cascade].
- Compiler and test diagnostics: HDLFORGE [abdollahi-2026-hdlforge] and a test-gated DeepSWE cascade [together-2026-deepswe-cascade-deepseek-gpt].
- PRM scores on critical reasoning steps: TRIM [kapoor-2026-trim-stepwise-routing].
- Small-model logit confidence: STEER [lee-2025-steer].

Three variants decide *when* to hand off rather than whether:
- TACIT-Switch models the permanent handoff time as a survival problem learned from censored teacher annotations [lei-2026-tacit-switch].
- SWE-Router reads a short cheap exploration before choosing a model [son-2026-swe-router].

Early-abort probes save compute in a different way, by stopping doomed episodes [ruan-2026-early-abort-agent-episodes].

**Planner-big / executor-small splits.** Several controlled studies agree on where capacity matters. In hierarchical search agents, scaling the delegation backbone gains about 11 EM points against about 2.6 for scaling the executor, and a 1.7B distilled executor matches a frontier one [cai-2026-think-big-search-small]. Orchestrator capacity drives performance, while reasoning in subagents adds little [zywot-2026-small-agents-beat-large-llm]. In Cursor's swarm, workers carry most tokens, so a frontier planner with cheap workers matched single-frontier quality at a fraction of the cost [lin-2026-cursor-agent-swarm-economics]. MinionS has the remote model decompose a task while the local model executes chunks [narayan-2025-minions].

The inverse pattern also appears: a cheap executor with an expensive advisor consulted at hard points [anthropic-2026-advisor-strategy], or a 7B agent trained to call an expert selectively [kon-2026-swe-protege]. Two caveats limit any universal rule. AgentCARD finds that whether planner or executor is the bottleneck depends on the domain [jiang-2026-agentcard]. AgentCollab reports higher accuracy with the large model on execution [gao-2026-agentcollab].

**Per-role and per-agent model assignment in multi-agent systems.** Offline search over role-to-model maps:
- Bandits: AgentOpt [hua-2026-agentopt].
- Confidence-bound search under a quality constraint: SCOPE [huang-2026-scope].
- Bayesian optimization: MALBO [sabbatella-2025-malbo] and syftr [conway-2025-syftr].
- Greedy search with LLM-estimated module quality: LLMSelector [chen-2025-llmselector].
- Cost-based optimizers for semantic operators: Abacus [russo-2025-abacus] and MOAR [wei-2025-moar].

Learned per-query controllers instead pick the collaboration mode, roles and the LLM for each agent [yue-2025-masrouter], [zhao-2026-sc-mas], [wang-2026-oi-mas], [liu-2026-caster]. X-MAS benchmarks 27 LLMs across MAS functions and finds that heterogeneous assignment helps without any change to structure [ye-2025-x-mas]. The value of substituting a stronger model depends on role and deployment, and can reverse across tasks [wang-2026-beyond-tier-labels].

**Topology and communication pruning.** One family removes agents or edges:
- AgentPrune prunes the message graph once [zhang-2024-agentprune].
- AgentDropout drops agents and edges per round [wang-2025-agentdropout].
- Adaptive Graph Pruning learns how many agents and which edges to keep per task [li-2025-adaptive-graph-pruning].
- SafeSieve prunes progressively, moving from heuristics to experience [zhang-2025-safesieve].
- AgentSlimming removes agents or downgrades them, accepting each change only if performance holds [chen-2026-agentslimming].
- LW2S learns which components to skip from counterfactual skip interventions [xu-2026-learning-what-to-skip].

Another family generates the topology per task: G-Designer [zhang-2024-g-designer], ARG-Designer [li-2025-arg-designer], MaAS [zhang-2025-maas-agentic-supernet] and DAAO [su-2025-daao]. In sparse debate, placing stronger models at central nodes helps [li-2024-sparse-topology-debate].

**RL-trained orchestrators.** Several small orchestrators are trained to call larger models or tools:
- ToolOrchestra, an 8B model [su-2025-toolorchestra].
- Conductor, a 7B model [nielsen-2025-conductor].
- xRouter [qian-2025-xrouter] and Router-R1 [zhang-2025-router-r1].
- CoRL, conditioned on a budget [jin-2025-corl-centralized-budget].

Others work at the level of the whole system. Puppeteer learns which agent acts next and prunes the rest [dang-2025-puppeteer-evolving-orchestration]. MAS-Orchestra generates a full MAS in one pass [ke-2026-mas-orchestra]. Uno-Orchestra jointly decides whether to decompose and where to send each subtask [cui-2026-uno-orchestra]. GraphPlanner chooses a (role, backbone) pair at each step [feng-2026-graphplanner]. The critique-and-routing controller of [fang-2026-critique-and-routing-controller] is a constrained MDP. When rollouts are too expensive, MCTS-collected data replaces online RL [guo-2026-sciorch], and an orchestration reward model cuts training cost [tsang-2026-orchrm].

**Budget-constrained agent execution.** Methods in this group optimize under explicit budgets:
- ZEBRA splits a monetary budget across pipeline phases with a knapsack [hamri-2026-zebra-budget-allocation].
- BAMAS provisions LLMs with an ILP [yang-2025-bamas].
- AgentBalance assigns backbones under budget first, then topology [cai-2025-agentbalance].
- Another approach selects a configuration per step by MILP under time and energy budgets [sedghani-2026-agentic-budget-constraints].
- SeqRoute applies offline RL over a session with a global budget [xu-2026-seqroute].
- Microsoft's boundary-guided RL works under per-task budgets [zhang-2026-budget-aware-agentic-routing].
- Turn-adaptive thinking budgets [jali-2026-turn-adaptive-budgets].
- Anthropic's task budgets expose a remaining-token countdown to the model [anthropic-task-budgets-docs].

**The costs of switching models mid-trajectory.** Switching has costs of its own:
- *Handoff tax.* Escalating a cheap model's trajectory recovers less than half of the quality gap, at a cost premium. Downshifting after the hard reasoning is done is favourable, and giving the escalated model less inherited context helps [ganz-2026-handoff-tax].
- *Drift.* A single handoff moves Multi-IF strict success by -8 to +13 pp, depending on the model pair [khraishi-2026-model-switching-drift].
- *Collaboration gap.* Strong-first, weak-later relays close much of the heterogeneous collaboration gap [davidson-2025-collaboration-gap].
- *Lost cache.* Switches lose cache and break tool or provider state. SAAR adds hard locks during tool loops and prices prefix-cache warmth into each switch [liu-2026-vllm-sr-session-aware-agentic-routing]. Kwartler routes only at session start, side lanes and subagent launch, and derives when a mid-task switch pays back [kwartler-2026-harness-tokenomics].
- *Restarts.* Residual-completion contracts avoid restarting the task after a handoff [deng-2026-cfrc-residual-completion-handoffs].

### Strongest methods

1. **Production per-turn routers with implicit-feedback labels and cache-aware cost** [okeefe-2026-cursor-router], [fireworks-2026-firerouter-with-opus]. These have the strongest external validity: live traffic, and in Fireworks' case sessions randomised to router versus Opus-only. Four design choices matter:
   - Labels come from whether the user moves on or corrects the agent.
   - Cost includes switch-induced cache misses.
   - A frontier model is admitted for a task label only when it beats the cheap model with 75% one-sided confidence.
   - Every policy is validated online after offline cross-validation.

   Both are vendor-reported.
2. **Calibrated step-level escalation with guarantees** [hemadri-2026-r2v-agent], [farzaneh-2026-tsds-edge-deferral]. Two choices matter. First, train the router on the cheap model's *residual* failures. Second, calibrate the threshold with conformal risk control or Learn-Then-Test, so that the escalation rate and quality can be controlled without retraining.
3. **Role-level heterogeneity with capacity on the decomposer** [cai-2026-think-big-search-small], [lin-2026-cursor-agent-swarm-economics], [jiang-2026-agentcard]. The design choice here is to measure where tokens are spent (mostly in workers) and where the capacity bottleneck sits (mostly in decomposition) before assigning tiers.
4. **Cache- and continuity-aware switching policy** [liu-2026-vllm-sr-session-aware-agentic-routing], [kwartler-2026-harness-tokenomics]. Ablations show that switch locks and cache pricing account for most of the savings. Naive per-turn routing does not.
5. **Simple verifier-gated cascades where cheap verification exists** [together-2026-deepswe-cascade-deepseek-gpt], [abdollahi-2026-hdlforge]. These are cheap to build and robust. They need executable tests.
6. **Offline search over role-to-model maps** [hua-2026-agentopt], [huang-2026-scope]. These are practical for fixed pipelines. They expose cost spreads of 13-32x at matched accuracy [hua-2026-agentopt].

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| Cursor Router (Auto Balance) | Auto Balance: outperforms Opus 4.8 at 41% lower cost with +3% satisfaction (relative to frontier models, measured on live traffic; vendor-reported) | live Cursor traffic | Opus 4.8 | [okeefe-2026-cursor-router] |
| FireRouter | Internal A/B test (>1 month, sessions randomised to router vs Opus-only): cost per session $15.36 -> $6.63 (57% +/- 19 ppts lower); graded-turn accuracy 78.7% vs 80.2% (98.1% of Opus-only); cache hit rate 94.2% vs 97.8% (vendor-reported) | internal coding traffic A/B test | Opus-only | [fireworks-2026-firerouter-with-opus] |
| AgentRouter (per-step tiers) | 72% cost reduction vs frontier-only while retaining 97.3% of frontier quality; RouteLLM and FrugalGPT applied per-step achieve 31% and 44% | 50,000 annotated agent trajectory steps | frontier-only | [paul-2026-agentrouter-step-tiers] |
| Harness-native LightGBM router | LightGBM router preserves task quality while cutting realized cost ~90% on PinchBench and ~43% on DRACO vs fixed strong model | PinchBench, DRACO | fixed strong model | [liu-2026-harness-native-routing] |
| SAAR session-aware routing | Across 21,600 deterministic turns: 79.29% fewer model switches, 3,836 unsafe switches eliminated, 78.71% estimated physical-model cost reduction vs single-turn routing (quality delta -0.0453); sticky sessions 98.65% cost reduction but -0.1433 quality | deterministic replay of 21,600 agent turns | single-turn routing | [liu-2026-vllm-sr-session-aware-agentic-routing] |
| Harness Tokenomics | Recovers 13-21% of model spend ($3.3M-$5.1M/year) in an emulated 10,000-seat enterprise at Anthropic list prices | ~10,000 real sessions from public datasets | not stated | [kwartler-2026-harness-tokenomics] |
| MTRouter | On ScienceWorld surpasses GPT-5 while reducing total cost by 58.7% | ScienceWorld | GPT-5 | [zhang-2026-mtrouter] |
| R2V Agent | 93.3% TerminalBench at 33.9% LLM calls, about half heuristic-router cost | TerminalBench | heuristic router | [hemadri-2026-r2v-agent] |
| ReDAct | Deferring about 15% of decisions to the large model matches exclusive large-model quality at significantly lower cost | ALFWorld, MiniGrid | exclusive large model | [piatrashyn-2026-redact] |
| Hera | 92.5% of cloud-only success with cloud use in only 46.3% of steps | ALFWorld, WebShop, AppWorld | cloud-only | [zhang-2026-hera-device-cloud] |
| TACIT-Switch | +7.4-11.1 pp success over task-level, step-level and fixed-prefix routing at comparable cost (simulation) | ALFWorld, DABench | task-level, step-level, fixed-prefix routing | [lei-2026-tacit-switch] |
| Test-gated cascade | Cascade (DeepSeek V4 Pro first, escalate to GPT-5.6 Sol on test failure): 83.0% of DeepSWE tasks at $3.35/task vs Sol alone 72.7% at $8.37 (10 pts better, 60% cheaper); 904 rollouts (vendor-reported) | DeepSWE | GPT-5.6 Sol alone | [together-2026-deepswe-cascade-deepseek-gpt] |
| Advisor (Sonnet + Opus) | Sonnet executor + Opus advisor: +2.7 points on SWE-bench Multilingual over Sonnet alone with 11.9% lower cost per agentic task | SWE-bench Multilingual | Sonnet alone | [anthropic-2026-advisor-strategy] |
| Orchestrator-worker vs solo | on full BrowseComp, solo Fable 5 at lower effort was 22-30% cheaper than the orchestrator at same accuracy | BrowseComp | Fable 5 coordinator + Sonnet 5 worker | [anthropic-2026-optimizing-cost-and-intelligence] |
| SWE-Protege | Qwen2.5-Coder-7B reaches 42.4% Pass@1 on SWE-bench Verified (+25.4% over prior SLM SOTA) using ~4 expert calls per task and 11% of tokens | SWE-bench Verified | prior SLM SOTA | [kon-2026-swe-protege] |
| Think Big, Search Small | Scaling delegation backbone improves EM ~11 points vs ~2.6 for execution; a 1.7B distilled executor matches a frontier sub-agent with 37% fewer sub-agent tokens | five multi-hop QA benchmarks | frontier sub-agent | [cai-2026-think-big-search-small] |
| Cursor planner/worker swarm | model mixes reached similar quality but cost ranged from $1,339 (Opus 4.8 planner + Composer 2.5 workers) to $10,565 (GPT-5.5 for all roles) | held-out SQL test suite (SQLite-from-scratch task) | GPT-5.5 for all roles | [lin-2026-cursor-agent-swarm-economics] |
| AgentCARD | Heterogeneous teams improve accuracy by up to 44% over cost-equivalent homogeneous teams, or match the strongest homogeneous team at up to 12x lower per-task cost via hybrid deployment | not stated | homogeneous teams | [jiang-2026-agentcard] |
| AgentOpt | At matched accuracy, best vs worst model combinations differ 13-32x in cost; UCB-E recovers near-optimal accuracy with 62-76% less evaluation budget than brute force | four benchmarks | brute-force search | [hua-2026-agentopt] |
| Uno-Orchestra | 77.0% macro pass@1, about 16% above strongest workflow baseline at roughly an order of magnitude lower per-query cost | 13-benchmark suite | strongest workflow baseline | [cui-2026-uno-orchestra] |
| ToolOrchestra | On HLE 37.1% vs GPT-5 35.1% while 2.5x more efficient; on tau2-Bench and FRAMES surpasses GPT-5 using about 30% of the cost | HLE, tau2-Bench, FRAMES | GPT-5 | [su-2025-toolorchestra] |
| MasRouter | Up to 52.07% overhead reduction vs SOTA on HumanEval; 17.21%-28.17% overhead reduction when plugged into mainstream MAS | HumanEval, MBPP | SOTA MAS | [yue-2025-masrouter] |
| ZEBRA | At 0.5 of unconstrained spend, recovers 94.4% of unconstrained quality vs 88.1% for LLM-direct on APPS | APPS | LLM-direct allocation | [hamri-2026-zebra-budget-allocation] |
| Scrouting | Matches best single model's solve rate on SWE-bench Pro Python slice (159 vs 158 of 266) at about a fifth of total cost per solve; the handoff, not routing, carries the result | SWE-bench Pro (Python, 266 tasks) | best single model | [bhola-2026-scrouting-superscout] |

### Critical notes

- **Replay evaluation is unreliable for in-agent switching.** On live forked SWE-bench rollouts, a swap rewrites 61-94% of later actions, and replay mispredicts every success-relevant outcome [gonuguntla-2026-replay-gap]. Many step routers are trained or scored on logged trajectories or deterministic replays, including AgentRouter [paul-2026-agentrouter-step-tiers], SAAR [liu-2026-vllm-sr-session-aware-agentic-routing] and Harness Tokenomics [kwartler-2026-harness-tokenomics]. Their reported savings should be treated as upper-bound estimates until dynamic evaluation such as TwinRouterBench's live track [yang-2026-twinrouterbench] is applied.
- **Handoff tax and drift.** Escalation recovers less than half the gap [ganz-2026-handoff-tax]. Drift depends on the model pair [khraishi-2026-model-switching-drift]. Scrouting's ablation credits the verified handoff artifact, not the routing decision [bhola-2026-scrouting-superscout].
- **Routing headroom is overstated.** Oracle gains are biased upward, and real routers fail on recall [fireworks-2026-frontier-is-a-router]. On web agents, 12-14% run-to-run noise inflates oracle gains, and no policy robustly beats a well-chosen fixed mode [wei-2026-routing-least-learnable]. Larger model pools often fall below the best single model [marjanovic-2026-model-pools-mas]. DecisionBench shows 15-31 points of headroom that agents do not realize [gao-2026-decisionbench].
- **Compute-matched negatives.** Delegating to cheaper workers loses to a solo model at lower effort on full BrowseComp, and advisor gains shrink as the executor approaches the advisor's capability [anthropic-2026-optimizing-cost-and-intelligence]. Replit argues that an external router is less capable than the model it routes for [furman-2026-replit-free-the-models]. The pruning-gain critique in C12 also applies here [zhang-2026-rethinking-mas-efficiency-evaluation].
- **Weak baselines and missing numbers.** Many MAS methods compare against other MAS ("SOTA"), not against a budget-matched single frontier agent [yue-2025-masrouter], [zhao-2026-sc-mas], [wang-2026-oi-mas]. Several abstracts give no headline number [gao-2026-agentcollab], [li-2026-progrouter], [son-2026-swe-router], [zhang-2026-budget-aware-agentic-routing]. Some evidence is small or simulated: n=50 and n=30 [sethi-2026-trace-action-agreement], 125 requests [wu-2026-rlm-cascade], and simulation [lei-2026-tacit-switch], [shaikh-2026-bayesian-self-escalation].
- **Vendor numbers.** The production results [okeefe-2026-cursor-router], [fireworks-2026-firerouter-with-opus], [notdiamond-2026-interactive-benchmarks-routing], [lin-2026-cursor-agent-swarm-economics], [amp-2026-subagent-model-assignment], [together-2026-deepswe-cascade-deepseek-gpt] and [anthropic-2026-advisor-strategy] are self-reported. They are not independently replicated.
- **Preprint share.** About half of C3 (157 of 316 records) are unreviewed 2026 arXiv preprints. The share is similar among the highest-relevance records (50 of 114). None of the core 2026 step-routing results has been independently replicated.

### Open problems

- Training step routers on counterfactual (branched) rollouts rather than replays, at acceptable cost [gonuguntla-2026-replay-gap], [jiang-2026-cota-tiny-advisors].
- Credit assignment for delayed rewards when the model choice changes at each step. TRACE-Router sidesteps this by pinning one model per task [raj-2026-trace-router], and RoRo proposes process rewards [ye-2026-roro-process-reward-routing].
- A theory of when a downshift or escalation pays back, net of cache and handoff costs [ganz-2026-handoff-tax], [kwartler-2026-harness-tokenomics].
- Role assignment that transfers across domains, given that planner and executor bottlenecks flip between domains [jiang-2026-agentcard], [wang-2026-beyond-tier-labels].
- Interaction between routing and harness adaptation or distillation of the cheap model [yao-2026-mera], [yang-2026-harness-adaptation-slm].

## C4. Automated optimization of compound AI system configurations

### Key ideas

This cluster treats a multi-step LLM system as a set of discrete and continuous knobs to search over against a metric. The knobs include which model fills each module or role, the prompts, the topology and the inference technique. There are four main lines of work.

1. **Per-module model assignment under cost or quality constraints.** One line poses "which LLM for which module" as a constrained search problem. LLMSelector greedily assigns models module by module, relying on the observation that end-to-end quality is often monotone in per-module quality [chen-2025-llmselector]. SCOPE minimizes cost subject to a quality threshold, using per-query results and confidence bounds [huang-2026-scope]. AgentOpt uses bandit and BO search over per-role assignments [hua-2026-agentopt]. MALBO uses multi-objective BO with GP surrogates [sabbatella-2025-malbo]. MILP and ILP formulations add budgets, SLOs and placement [sedghani-2026-agentic-budget-constraints] [gravara-2026-atlas] [yang-2025-bamas]. ZEBRA splits a fixed dollar budget across phases with a knapsack solved by water-filling [hamri-2026-zebra-budget-allocation].
2. **Whole-flow configuration search.** Multi-objective BO and hierarchical autotuning search over RAG and agent flows, including the model choice per component [conway-2025-syftr] [barker-2025-mo-hpo-rag] [he-2025-cognify]. Cost-based query optimizers do the same for semantic operator pipelines [russo-2025-abacus] [wei-2025-moar] [patel-2024-lotus] [shankar-2026-task-cascades].
3. **Workflow, topology and MAS design.** This line uses search or learned generators: MCTS [zhang-2024-aflow], evolution [zhang-2025-evoflow] [borthwick-2026-robophd] [xu-2026-hyevo], supernets [zhang-2025-maas-agentic-supernet], graph generators [zhang-2024-g-designer], RL meta-agents [gao-2025-flowreasoner] [nie-2025-w4s] [dang-2025-puppeteer-evolving-orchestration], and joint role/edge/LLM controllers [zhao-2026-sc-mas] [zhang-2026-hetero-designer] [cai-2025-agentbalance].
4. **Prompt and program optimization that enables model downgrades.** DSPy, MIPRO and GEPA optimize prompts and demonstrations against a metric [khattab-2023-dspy] [opsahlong-2024-mipro] [agrawal-2025-gepa]. Their main cost relevance is "re-optimize for the cheaper model" [wang-2026-dropbox-dash-dspy] [singhvi-2025-databricks-gepa-90x] [notdiamond-2025-prompt-adaptation-weaker-models]. Harness adaptation applies the same logic to tools and loops [yang-2026-harness-adaptation-slm].

How these relate: lines 1-2 hold the program fixed and choose models and settings. Lines 3-4 change the program itself, and that often matters more for cost than model choice, because a better harness or workflow lets a smaller model suffice [zhang-2024-aflow] [yang-2026-harness-adaptation-slm] [tan-2025-langprobe]. Serving-aware variants add live load as a runtime input [dai-2025-aragog] [gravara-2026-compass] [kabir-2026-inframind] [chaudhry-2025-murakkab]. Step-level routers bridge to C3 [paul-2026-agentrouter-step-tiers] [li-2026-rsi-router] [zhang-2026-evoroute].

### Strongest methods

- **Constrained per-module selection with statistical guarantees** [huang-2026-scope]. It optimizes cost subject to a quality floor, the right objective for "keep quality, cut cost". It uses confidence bounds on per-query results and was published at KDD.
- **Bandit and BO assignment search with a small evaluation set** [hua-2026-agentopt] [sabbatella-2025-malbo]. AgentOpt documents 13-32x cost spreads among model combinations at matched accuracy. That makes search worth doing, and its UCB-E search reduces evaluation budget [hua-2026-agentopt].
- **Pareto-frontier flow search** [conway-2025-syftr] [he-2025-cognify] [russo-2025-abacus]. Design choices that matter: optimizing the cost objective jointly rather than after the fact, early stopping or budget reallocation across search layers [he-2025-cognify], and tracking subplan Pareto frontiers [russo-2025-abacus].
- **Composed accuracy estimates instead of full end-to-end profiling** [gravara-2026-atlas] [li-2026-flowcompile]. Atlas uses a Markovian stage-to-stage accuracy predictor and reports lower profiling cost [gravara-2026-atlas].
- **Prompt re-optimization on model swap, with production evidence** [wang-2026-dropbox-dash-dspy]. This is one of the few prod=Y results. It was measured against human labels, and adaptation time fell from weeks to days [wang-2026-dropbox-dash-dspy].
- **Step-tier classifier trained on annotated steps** [paul-2026-agentrouter-step-tiers]. It is cheap at inference (<5ms) and compares directly against per-step RouteLLM and FrugalGPT.

Design choices that recur in the stronger papers:
- an explicit quality constraint rather than a scalarized reward [huang-2026-scope] [hamri-2026-zebra-budget-allocation];
- an acceptance test anchored to a baseline before each change [chen-2026-agentslimming];
- role-level diagnostics, since which role is the bottleneck depends on the domain [jiang-2026-agentcard] [wang-2026-beyond-tier-labels].

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| AgentSlimming | Reduces average token cost by up to 78.9% with negligible performance degradation | eight benchmarks | not stated | [chen-2026-agentslimming] |
| ZEBRA | At 0.5 of unconstrained spend, recovers 94.4% of unconstrained quality vs 88.1% for LLM-direct on APPS; beats LLM-direct by 14.3pp on 3-phase HotpotQA pipeline | APPS, HotpotQA | LLM-direct allocation; unconstrained spend | [hamri-2026-zebra-budget-allocation] |
| AgentOpt | At matched accuracy, best vs worst model combinations differ 13-32x in cost; UCB-E recovers near-optimal accuracy with 62-76% less evaluation budget than brute force | four benchmarks | brute-force search | [hua-2026-agentopt] |
| SCOPE | Under same search budget and quality constraint, up to 20x lower cost than best competitor during search and up to 6x lower final cost in returned solution (three data processing tasks, 7 baselines) | three data processing tasks | 7 baselines | [huang-2026-scope] |
| RSI-Router | Surpasses DeepSeek-only baseline at about 48.3% of the cost across five agentic benchmarks; 74.7-82.2% cost cuts with better performance on ALFWorld, ScienceWorld, WebShop; +16.7% relative on Terminal-Bench 2.0 at 18.0% lower cost | ALFWorld, ScienceWorld, WebShop, Terminal-Bench 2.0 | DeepSeek-only | [li-2026-rsi-router] |
| AgentRouter | 72% cost reduction vs frontier-only while retaining 97.3% of frontier quality; RouteLLM and FrugalGPT applied per-step achieve 31% and 44% | 50,000 annotated agent trajectory steps | frontier-only; per-step RouteLLM, FrugalGPT | [paul-2026-agentrouter-step-tiers] |
| Dropbox Dash judge + GEPA/MIPROv2 (prod, vendor-reported) | Adapting the production relevance judge from o3 to gpt-oss-120b with GEPA cut NMSE 45% (8.83 to 4.86); enabled 10-100x more labeling at the same cost; on gemma-3-12b (MIPROv2) NMSE 46.88 to 17.26 and malformed JSON down >97% | Dropbox Dash relevance judgments (internal) | unoptimized prompt on target model | [wang-2026-dropbox-dash-dspy] |
| Automated harness adaptation | best small-model agent recovered 89.7% of LLM performance at 4% of the cost; optimized harnesses improved 16 of 21 task-SLM pairs, closing the gap in seven | seven business-oriented agentic tasks | LLM agent | [yang-2026-harness-adaptation-slm] |
| syftr | Finds flows on average ~9x cheaper while preserving most of the accuracy of the most accurate Pareto-frontier flows, across multiple RAG benchmarks | multiple RAG benchmarks | most accurate Pareto-frontier flows | [conway-2025-syftr] |
| Abacus | Abacus-optimized systems achieve 6.7%-39.4% better quality and are 10.8x cheaper and 3.4x faster than the next best system on average (BioDEX, CUAD, MMQA) | BioDEX, CUAD, MMQA | next best system | [russo-2025-abacus] |
| MOAR | 27% higher accuracy than ABACUS on average; matches ABACUS's best accuracy at 55% of its cost, across six workloads | six workloads | ABACUS | [wei-2025-moar] |
| AFlow | Smaller models outperform GPT-4o on specific tasks at 4.55% of its inference cost in dollars; 5.7% average improvement over SOTA baselines across six benchmarks | six benchmark datasets | GPT-4o; SOTA baselines | [zhang-2024-aflow] |
| GEPA on gpt-oss-120b (vendor-reported) | GEPA-optimized gpt-oss-120b surpasses Claude Sonnet 4 and Claude Opus 4.1 baselines by ~3% while being ~22x and ~90x cheaper to serve on IE Bench; GEPA ~20% cheaper to serve than SFT alternatives | IE Bench (Databricks internal information extraction) | Claude Sonnet 4, Claude Opus 4.1; SFT | [singhvi-2025-databricks-gepa-90x] |
| CLEAR evaluation | Accuracy-only optimization yields agents 4.4-10.8x more expensive than cost-aware alternatives with comparable performance (300 enterprise tasks); up to 50x cost variation for similar precision | 300 enterprise tasks | cost-aware alternatives | [mehta-2025-clear] |
| Meta-agent design cost audit | Only on two datasets does total design-plus-deployment cost of meta-designed agents fall below human-designed agents, and only when deployed on over 15,000 examples | not stated | human-designed agents | [el-2025-inefficiencies-meta-agents] |

### Critical notes

- **Quality-only optimizers.** Much of the workflow and prompt-search literature optimizes accuracy and reports cost only as a side effect [hu-2024-adas] [zhou-2025-mass] [opsahlong-2024-mipro] [agrawal-2025-gepa]. CLEAR finds accuracy-only optimization yields agents 4.4-10.8x more expensive than cost-aware alternatives [mehta-2025-clear]. Prefer methods that take cost as a constraint or objective [huang-2026-scope] [conway-2025-syftr] [bussing-2026-mo-capo].
- **Search and design cost is rarely amortized in the reporting.** Meta-designed agents recoup design cost only on two datasets, and only past roughly 15,000 deployed examples [el-2025-inefficiencies-meta-agents]. Per-query workflow generation is often unnecessary [wang-2026-scale-query-level-workflows]. Amortized transfer is one answer [du-2026-swift].
- **Prompt optimization is noisy.** In one large study, 49% of runs on Claude Haiku 4.5 scored below zero-shot, and agent-prompt interactions were never significant [zhang-2026-po-coin-flip]. Hidden design choices decide whether generative optimization works [nie-2026-generative-opt-challenges].
- **Vendor and internal benchmarks.** The 90x and 22x claims use an internal benchmark [singhvi-2025-databricks-gepa-90x]. The Not Diamond clinc150 figure is vendor-reported [notdiamond-2025-prompt-adaptation-weaker-models]. Dropbox's numbers are internal production data, though measured against human labels [wang-2026-dropbox-dash-dspy].
- **Static offline evaluation sets.** Nearly every method optimizes against a fixed dev set. None of the high-relevance C4 records report production deployment except [wang-2026-dropbox-dash-dspy]. Search-time "online" bandits [hua-2026-agentopt] [wei-2025-moar] [xu-2025-boad] are bandits over configurations evaluated on a benchmark, not over live traffic.
- **Upgrade value is context-dependent.** The benefit of upgrading a call depends on its role and deployment and can reverse across tasks [wang-2026-beyond-tier-labels]. The bottleneck role is domain-dependent [jiang-2026-agentcard]. Tier labels transfer poorly.
- **Headline-number hygiene.** DAAO's cost figure comes from a "search summary" rather than the abstract [su-2025-daao]. FlowCompile reports speedup, not dollar cost [li-2026-flowcompile]. Several records report no headline cost number [barker-2025-mo-hpo-rag] [sedghani-2026-agentic-budget-constraints].

### Open problems

- Optimizers that use customer logs instead of curated dev sets. Kiyohara et al. is the only off-policy prompt optimizer in this cluster [kiyohara-2025-prompt-opt-logged-bandit].
- Re-optimization triggered by model drift or price changes, beyond one-off migration [wang-2026-dropbox-dash-dspy] [gravara-2026-compass].
- Credit assignment across modules for per-role model choice without exhaustive grids [wu-2025-optimas] [yang-2025-shapleyflow] [zehle-2026-cantante].
- Honest total-cost accounting that includes search, profiling and re-optimization [el-2025-inefficiencies-meta-agents] [gravara-2026-atlas].
- Joint optimization of prompts and model choice, given evidence that prompts must be re-tuned per model [notdiamond-2025-prompt-adaptation-weaker-models] [wang-2026-dropbox-dash-dspy].

## C5. Online learning from live traffic

### Key ideas

The core formalism is the contextual bandit over models. The router sees query features, picks one model, and observes only that model's reward and cost. The cluster has five main lines of work.

1. **Cost-aware contextual bandits for routing** [nguyen-2024-metallm] [wang-2025-mixllm] [li-2025-llm-bandit] [wei-2025-barp] [tsai-2026-neuralucb-routing] [ziller-2026-greenserv]. Preference conditioning lets one policy serve many cost-quality tradeoffs [wei-2025-barp] [li-2025-llm-bandit].
2. **Budgeted and paced bandits.** These use primal-dual pacing [taberner-miller-2026-paretobandit] [zhang-2026-m-cmab], knapsack control [panda-2025-pilot], workload-level budgets [li-2026-wiserouter], live budget consumption [zu-2026-barouter], or Lyapunov virtual queues for SLA guarantees [woisetschlager-2025-mess-plus]. The classical bandits-with-knapsacks theory underpins them [badanidiyuru-2013-bwk] [agrawal-2015-linear-cbwk].
3. **Drift and pool changes.** Methods include forgetting and hot-swapping models [taberner-miller-2026-paretobandit], audit windows and shadow prices [lee-2026-drift-aware-routing], and models that improve over time [xia-2024-which-llm-to-play]. Onboarding new models uses model fingerprints [jitkrittum-2025-unirouter] or no-retrain addition [ziller-2026-greenserv] [chen-2025-concur]. Provider-level drift for the same weights is a separate case [he-2026-market-aware-routing] [li-2026-same-model-not-same-service] [chen-2023-chatgpt-behavior-over-time].
4. **Feedback design.** Sources include sparse one-sided complaints [woisetschlager-2026-slarouter], user retrials [bae-2026-contextual-queueing-bandits], pairwise or dueling preferences [chiang-2025-dueling-routing] [lai-2026-routejudge], surrogate judge rewards with misspecification-robust regret [sridhar-2026-correlated-bandits-routing], and limited labeling budgets [wei-2026-llm-experts-limited-feedback].
5. **Online learning inside agent pipelines.** This includes per-step or per-subtask bandits [atalar-2025-sequential-bandits] [li-2026-progrouter] [guan-2026-symphony-coord] [zhang-2026-evoroute] and task-level pinning [raj-2026-trace-router]. Online cascades and distillation, where the cheap model learns from the expensive model's live answers, also belong here [nie-2024-online-cascade-learning] [wu-2025-inter-cascade] [ramirez-2023-cache-and-distil] [stogiannidis-2023-cache-me-if-you-can].

Production systems in this cluster mostly combine an offline prior with online signals rather than learning from scratch [okeefe-2026-cursor-router] [openai-2025-gpt5-system-card] [bao-2026-orcarouter] [liskowski-2026-compositional-online-learning].

### Strongest methods

- **Cursor Router** [okeefe-2026-cursor-router]. A per-turn router is trained on hundreds of thousands of production turns with implicit satisfaction labels and actual cost, including cache misses caused by model switches. It admits a frontier model for a task label only at 75% one-sided confidence of beating the cheap model, mixes models within a per-mode budget, and is validated on held-out logs and then on live traffic. Vendor-reported.
- **FireRouter** [fireworks-2026-firerouter-with-opus]. This is the only randomized live A/B result in the cluster: sessions were randomized over more than a month, and the router prices the prompt-cache value lost by switching models. Vendor-reported.
- **ParetoBandit** [taberner-miller-2026-paretobandit]. It combines a primal-dual budget pacer, geometric forgetting and hot-swappable models, which addresses the budget, drift and onboarding problems together. Open-source.
- **SLARouter and MESS+** [woisetschlager-2026-slarouter] [woisetschlager-2025-mess-plus]. These encode a quality SLA as a constraint, not a penalty. SLARouter learns from sparse one-sided feedback, which matches what production actually yields.
- **BaRP** [wei-2025-barp]. It trains under the same partial-feedback restriction as deployment, and the operator can move the tradeoff at test time.
- **PILOT and OrcaRouter** [panda-2025-pilot] [bao-2026-orcarouter]. Both warm-start from offline preference data or a full-information matrix, then refine online with LinUCB. This is the recommended structure for teams that already hold logs.
- **Compositional online learning in Snowflake Cortex AISQL** [liskowski-2026-compositional-online-learning]. A prod=Y case study that separates per-component savings and shows realistic erosion from an upper bound of 11.4x to about 8x.

Design choices that matter:
- including switch and cache costs in the reward [okeefe-2026-cursor-router] [fireworks-2026-firerouter-with-opus];
- confidence-gated upgrades [okeefe-2026-cursor-router];
- hard budget pacing [taberner-miller-2026-paretobandit] [zu-2026-barouter];
- multiplicative rather than additive latency-quality objectives, since additive rewards cause collapse [chu-2026-lqm-contextroute];
- fail-safe anchors [he-2026-market-aware-routing].

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| FireRouter (vendor-reported, live A/B) | Internal A/B test (>1 month, sessions randomised to router vs Opus-only): cost per session $15.36 -> $6.63 (57% +/- 19 ppts lower); graded-turn accuracy 78.7% vs 80.2% (98.1% of Opus-only); cache hit rate 94.2% vs 97.8% (vendor-reported) | internal coding traffic A/B test | Opus-only | [fireworks-2026-firerouter-with-opus] |
| Cursor Router (vendor-reported, live traffic) | Auto Intelligence: above Fable-level user satisfaction at 68% lower cost; Auto Balance: outperforms Opus 4.8 at 41% lower cost with +3% satisfaction (relative to frontier models, measured on live traffic; vendor-reported) | live Cursor traffic, held-out offline split of production turns | frontier models (Fable, Opus 4.8) | [okeefe-2026-cursor-router] |
| Not Diamond cost-tradeoff knob (vendor-reported, before/after) | Enabling the cost-tradeoff parameter on Not Diamond's consumer chatbot cut inference cost 51% on average (~$750,000 annualised); power-user median messages fell from 5 to 4/day while week-over-week message growth was 43% vs 16% the prior week (vendor-reported, before/after on live traffic) | live chatbot traffic before/after | quality-maximising router setting | [notdiamond-2024-reduce-own-inference-costs] |
| Compositional online learning (prod) | Learning components compose to an 11.4x upper bound on per-row LLM cost reduction for a representative conjunction-filter workload, reduced to a realistic ~8x after self-selection, sample-budget shrinkage and selectivity drift | Cortex AISQL production case study | not stated | [liskowski-2026-compositional-online-learning] |
| FACET (market-aware provider routing) | Certification moved live traffic from a premium anchor provider to a substantially cheaper certified endpoint (matched quality) | not stated | premium anchor provider | [he-2026-market-aware-routing] |
| OrcaRouter | 75.54% accuracy at USD 1.00 per 1,000 queries; ranked second on RouterArena (arena score 72.08) | RouterArena | not stated | [bao-2026-orcarouter] |
| ParetoBandit | budget compliance within 0.4%; up to +0.071 quality lift under price/quality shifts (1,824 prompts, three-model portfolio) | 1,824 benchmark prompts | not stated | [taberner-miller-2026-paretobandit] |
| SLARouter | reduces operating cost by up to 2.2x over existing baselines while meeting SLA constraints | wide range of LLM benchmarks | existing baselines | [woisetschlager-2026-slarouter] |
| MESS+ | Average 2x cost savings vs existing LLM routing techniques (abstract) | LLM benchmarks | existing LLM routing techniques | [woisetschlager-2025-mess-plus] |
| MixLLM | 97.25% of GPT-4's quality at 24.18% of the cost under the time constraint (abstract) | routing benchmark with mixed LLMs | GPT-4 | [wang-2025-mixllm] |
| BaRP | Outperforms strong offline routers by at least 12.46% and the largest LLM by at least 2.45% (abstract) | routing benchmarks (unseen tasks) | strong offline routers; largest LLM | [wei-2025-barp] |
| TRACE-Router | On tau2-Bench beats latency-matched interpolation between models by 7-8 accuracy points; non-dominated Pareto points on three agentic benchmarks | tau2-Bench, Terminal-Bench, third agentic benchmark | latency-matched interpolation between models | [raj-2026-trace-router] |
| Online cascade learning | Parallels LLM accuracy while cutting inference costs by as much as 90% across four benchmarks | four stream benchmarks | LLM | [nie-2024-online-cascade-learning] |
| Inter-Cascade | Up to 33.06% weak-model and 6.35% system accuracy gain; up to 48.05% fewer strong-model calls and up to 49.63% fee savings vs standard cascades | multiple reasoning benchmarks | standard cascades | [wu-2025-inter-cascade] |
| NeuralUCB routing (simulated online) | substantially lower inference cost than max-quality reference with competitive reward (RouterBench, simulated online) | RouterBench | max-quality reference | [tsai-2026-neuralucb-routing] |

### Critical notes

- **Simulation-only bandit results dominate.** Most academic bandit routers are evaluated by streaming fixed benchmark datasets (RouterBench, RouterArena, a 1,824-prompt set) rather than live users. They are prod=N in the digests [taberner-miller-2026-paretobandit] [bao-2026-orcarouter] [woisetschlager-2026-slarouter] [woisetschlager-2025-mess-plus] [wang-2025-mixllm] [ziller-2026-greenserv]. Tsai et al. state explicitly "simulated online" [tsai-2026-neuralucb-routing], and Symphony-Coord reports "simulation" [guan-2026-symphony-coord]. Replaying a full-information benchmark hides the core production problem: rewards are delayed, noisy and partial.
- **Theory-heavy, evidence-light.** Several papers mainly offer regret bounds [lee-2026-drift-aware-routing] [li-2025-stageroute] [duy-2026-online-l2d-varying-experts] [poon-2025-online-multi-llm-context-evolution]. Of the drift-aware work, the database notes "little empirical evidence" [lee-2026-drift-aware-routing].
- **Production evidence is vendor-reported.** This applies to Cursor [okeefe-2026-cursor-router], Fireworks [fireworks-2026-firerouter-with-opus], Not Diamond [notdiamond-2024-reduce-own-inference-costs], Cursor Tab [jackson-2025-cursor-tab-online-rl] and the GPT-5 router description, which gives no numbers [openai-2025-gpt5-system-card]. Only FireRouter reports randomization and an uncertainty interval (57% +/- 19 ppts) [fireworks-2026-firerouter-with-opus]. Not Diamond's is a before/after comparison with concurrent engagement changes, so it is confounded by time [notdiamond-2024-reduce-own-inference-costs].
- **Surrogate rewards.** Online learners frequently optimize judge scores [chu-2026-lqm-contextroute] [sridhar-2026-correlated-bandits-routing] or implicit signals such as corrections, retrials and model switches [okeefe-2026-cursor-router] [bae-2026-contextual-queueing-bandits] [openai-2025-gpt5-system-card]. Implicit feedback is a noisy learning signal [liu-2025-user-feedback-noisy-signal] and invites proxy hacking [pang-2023-implicit-feedback-deployment-dialogue].
- **Credit assignment in agents.** Per-call routers cannot attribute delayed terminal reward [raj-2026-trace-router]. Per-step bandits [atalar-2025-sequential-bandits] [li-2026-progrouter] must model how upstream choices change downstream state.
- **Exploration cost on enterprise traffic.** Few papers bound the quality harm of exploration on real users. SLA-constrained designs [woisetschlager-2026-slarouter] [woisetschlager-2025-mess-plus] and fail-safe anchors [he-2026-market-aware-routing] are the exceptions.
- **Same model is not the same service.** Provider drift breaks the stationarity assumption even when the model weights are fixed [li-2026-same-model-not-same-service] [he-2026-market-aware-routing]. API models change over months [chen-2023-chatgpt-behavior-over-time] [ma-2023-prompt-getting-worse].

### Open problems

- Randomized, peer-reviewed live evaluations of bandit routers in enterprise agent workloads. The current evidence is [fireworks-2026-firerouter-with-opus] and vendor blogs.
- Safe exploration with quality guarantees at step level inside multi-agent pipelines [woisetschlager-2026-slarouter] [atalar-2025-sequential-bandits].
- Combining surrogate-reward bandits with debiasing of implicit feedback [sridhar-2026-correlated-bandits-routing] [morandi-2026-selection-bias-sparse-feedback].
- Pricing cache, KV and switching costs inside the bandit reward beyond turn-level coding agents [fireworks-2026-firerouter-with-opus] [okeefe-2026-cursor-router].
- Jointly handling drift, budget and onboarding of new models with empirical validation on live traffic [taberner-miller-2026-paretobandit] [lee-2026-drift-aware-routing].

## C6. Offline learning from logs and customer data

### Key ideas

This cluster covers what can be learned before touching live traffic. There are five main lines of work.

1. **Supervised routers on preference or correctness data.** RouteLLM uses Arena preferences [ong-2024-routellm]. Hybrid LLM predicts the quality gap between models [ding-2024-hybrid-llm]. Other routers use correctness classifiers and benchmarks [shnitzer-2023-routing-benchmark-datasets] [sakota-2023-forc], model embeddings and IRT [zhuang-2024-embedllm] [song-2025-irt-router], kNN [li-2025-knn-routing], separate cost and accuracy estimators [somerstep-2025-carrot], or distilled reward-model rankings [lu-2023-zooter]. Arch-Router routes by human-written policy descriptions [tran-2025-arch-router].
2. **Causal and counterfactual learning from logs where only the chosen model's outcome is observed.** Examples are regret-minimizing causal routing [tsiourvas-2025-causal-routing], causal debiasing of judge labels [zhang-2025-meta-router], off-policy prompt learning [kiyohara-2025-prompt-opt-logged-bandit] [kiyohara-2025-opl-sentence-personalization], and the position that routing from logs is a causal question [frauen-2026-causal-methods-llm] [loghmani-2025-deconfoundlm].
3. **Off-policy evaluation (OPE).** Work here includes OPE from logged human feedback [bhargava-2024-ope-logged-human-feedback], optimal-transport doubly-robust OPE for black-box LLMs [xiang-2026-otrope], world-model OPE for multi-turn agents [liu-2026-adwm-ope-llm-agents], and an empirical estimator comparison for tool choice [li-2026-counterfactual-tool-ranking]. It rests on the classical DR, SNIPS and replay foundations [dudik-2011-doubly-robust] [swaminathan-2015-snips] [li-2010-unbiased-offline-eval] [saito-2022-ope-large-action-embeddings].
4. **Offline RL and trajectory learning for agents.** This covers session-level CQL routing [xu-2026-seqroute], Lagrangian SLA routers [bhatti-2026-proteus], conformal constrained policies [si-2025-ccpo], per-turn routers trained on logged trajectories [zhang-2026-mtrouter] [paul-2026-agentrouter-step-tiers], escalation timing learned from censored labels [lei-2026-tacit-switch], recovery routing, skip policies [xu-2026-learning-what-to-skip], and state-matched credit [liu-2026-treecredit].
5. **Turning logs into cheaper models and better evaluations.** Examples are distillation and post-training from production traces [gola-2026-langsmith-fine-tuning] [tsymboi-2026-production-traffic-post-training] [databricks-2025-tao] [yao-2026-mera], performance predictors [pacchiardi-2025-predictaboard] [xiao-2025-generalized-correctness-models] [zhang-2026-v0-generalist-value-model], and statistically valid evaluation that mixes judge and human labels [angelopoulos-2023-ppi] [boyeau-2024-autoeval-done-right] [fisch-2024-stratified-ppi].

### Strongest methods

- **Causal LLM Routing** [tsiourvas-2025-causal-routing]. A NeurIPS paper that learns end-to-end from logs where only the deployed model's outcome is recorded. That is the actual structure of customer logs.
- **Budgeted allocation under uncertain evaluation** [khosravi-2026-which-llm-which-work]. Tested on a production log, it certifies when an assignment is optimal across all quality tables consistent with the evidence, and finds that better quality information saves more than further assignment optimization.
- **Empirical OPE comparison with exact propensities** [li-2026-counterfactual-tool-ranking]. Direct regression beat doubly robust estimation in-distribution (MAE 0.0139 vs 0.0272), while DR was better under shift. This is directly actionable for estimator choice.
- **Step and turn routers trained on logged trajectories** [paul-2026-agentrouter-step-tiers] [zhang-2026-mtrouter] [liu-2026-harness-native-routing]. They show 43-90% cost reductions at preserved quality on agentic benchmarks. MTRouter makes fewer model switches, which matters for cache cost.
- **Escalation routing with calibrated or constrained risk** [lei-2026-tacit-switch] [si-2025-ccpo].
- **SeqRoute** [xu-2026-seqroute]. Offline RL with hindsight budget relabeling expands 10,000 sessions into 2.38M transitions, and the policy keeps bankruptcy under 1%.
- **Strong simple baselines** [li-2025-knn-routing] [somerstep-2025-carrot]. kNN matches complex learned routers, and CARROT is minimax-rate-optimal.

Design choices that matter:
- include the full cost, including escalations [manoharan-2026-t2mo] [okeefe-2026-cursor-router];
- condition on partial trajectories [son-2026-swe-router];
- account for the cost of collecting supervision [lai-2026-saverouter];
- debias judge and implicit labels [zhang-2025-meta-router] [morandi-2026-selection-bias-sparse-feedback].

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| TACIT-Switch | +7.4-11.1 pp success over task-level, step-level and fixed-prefix routing at comparable cost (simulation); best held-out success on ALFWorld (48.5% with 4B cheap, 45.5% with 9B cheap) and DABench (73.1%) | ALFWorld, DABench | task-level, step-level and fixed-prefix routing | [lei-2026-tacit-switch] |
| MERA | Deployed policy retains 88.3% pass at 60.8% of always-teacher cost on HumanEval+MBPP; Qwen2.5-Coder-1.5B improves 28.7% to 49.7% after four cycles | HumanEval+, MBPP | always-teacher | [yao-2026-mera] |
| MTRouter | On ScienceWorld surpasses GPT-5 while reducing total cost by 58.7%; on HLE competitive accuracy with 43.4% lower cost than GPT-5 | ScienceWorld, Humanity's Last Exam | GPT-5 | [zhang-2026-mtrouter] |
| Harness-native routing | LightGBM router preserves task quality while cutting realized cost ~90% on PinchBench and ~43% on DRACO vs fixed strong model | DRACO, PinchBench | fixed strong model | [liu-2026-harness-native-routing] |
| SeqRoute | Reduces operational costs by 6.0-73.5% while maintaining or improving quality; bankruptcy under 1% (abstract) | 10,000 raw sessions expanded to 2.38M transitions | not stated | [xu-2026-seqroute] |
| CCPO | Up to 30% cost reduction vs other cost-aware baselines and LLM-guided methods without compromising reliability on two multi-hop QA benchmarks | two multi-hop QA benchmarks | other cost-aware baselines and LLM-guided methods | [si-2025-ccpo] |
| PROTEUS | 90.1% accuracy on RouterBench within 1.3% of oracle; cost savings up to 89.8% vs best fixed model (abstract) | RouterBench, SPROUT | oracle; best fixed model | [bhatti-2026-proteus] |
| Post-retrieval cascade (prod) | Post-retrieval cascade: +0.140 Composite Overall vs Always-HyDE, 31.8% lower latency, 72.2% of real queries served without LLM augmentation | Danish National Encyclopedia production traffic (20,000 query-workflow pairs) | Always-HyDE | [hussain-2026-coverage-illusion-rag] |
| SaveRouter | Uses ~33-41% of training feedback at competitive quality; reduces break-even deployment volume ~1.9-9.5x (abstract) | four routing benchmarks | not stated | [lai-2026-saverouter] |
| Web-agent routing bounds | Sending only tasks no mode solves to the cheapest mode cuts cost 9.5-30.6% at unchanged success; no tested routing policy robustly beats one well-chosen fixed mode | VisualWebArena, WebArena | one well-chosen fixed mode | [wei-2026-routing-least-learnable] |
| LangSmith fine-tuning (vendor-reported) | Engine agent: Kimi K3 + SFT 96.0 vs Kimi K3 90.0 vs GPT-5.6 Sol 87.0 on internal IssueBench subset; OpenSWE Review: Qwen-3.8-27B + SFT F1 48.9%->53.7% with 29.8% fewer model calls and 29.4% fewer tool requests per review (vendor-reported, internal agents) | IssueBench (internal), internal PR review set | Kimi K3; GPT-5.6 Sol; base Qwen-3.8-27B | [gola-2026-langsmith-fine-tuning] |
| Production-traffic post-training (prod) | model absorbs 50% of platform traffic (116M requests/month) at a fraction of the serving cost | in-house Arena, instruction following, function-calling | not stated | [tsymboi-2026-production-traffic-post-training] |
| RouteLLM | Blog: >85% cost reduction on MT-Bench, 45% on MMLU, 35% on GSM8K vs GPT-4 at 95% of GPT-4 quality; paper: >2x cost reduction in some cases without quality loss | MT-Bench, MMLU, GSM8K | GPT-4 | [ong-2024-routellm] |
| Optimizing costs of LLM usage (prod) | Reduces costs by 40%-90% while improving quality by 4%-7% (enterprise and open-source document-processing datasets) | enterprise document datasets, open-source summarisation/QA datasets | not stated | [shekhar-2024-optimizing-costs-llm-usage] |

### Critical notes

- **Partial feedback and confounding.** Production logs record only the chosen model's outcome, and the routing that produced them depended on the query. Supervised routers trained on such logs inherit selection bias [tsiourvas-2025-causal-routing] [frauen-2026-causal-methods-llm] [loghmani-2025-deconfoundlm]. Most router papers sidestep this by training on full-information benchmark matrices [ong-2024-routellm] [somerstep-2025-carrot] [shnitzer-2023-routing-benchmark-datasets], which customer logs do not provide.
- **The replay gap.** In live forked SWE-bench rollouts, mid-trajectory model swaps rewrite 61-94% of subsequent actions, and replay evaluation mispredicts every success-relevant outcome [gonuguntla-2026-replay-gap]. Executed replay also shows step-level credit signals are unreliable [zhang-2026-credit-without-ground-truth]. This undermines offline evaluation of step-level routers trained on fixed trajectories [paul-2026-agentrouter-step-tiers] [zhang-2026-mtrouter], unless they are validated with counterfactual rollouts [liu-2026-treecredit] [xu-2026-learning-what-to-skip] or live tests [okeefe-2026-cursor-router].
- **Proxy and judge labels.** Quality tables built from proxies are uncertain because models are rarely compared on the same work [khosravi-2026-which-llm-which-work]. Judge and crowd labels need debiasing [zhang-2025-meta-router] [guerdan-2025-doubly-robust-llm-judge]. Thumbs feedback is selection-biased [morandi-2026-selection-bias-sparse-feedback]. Implicit signals are noisy [liu-2025-user-feedback-noisy-signal].
- **Oracle inflation and label scarcity.** Routing oracles under stochastic decoding overstate achievable gains [chen-2026-routing-oracle-stochastic]. Run-to-run noise of 12-14% inflates oracle gains, and supervision is scarcest where routing matters [wei-2026-routing-least-learnable]. Certify routing gain before shipping [sun-2026-routeguard] [chen-2026-resample-or-reroute].
- **Synthetic vs real traffic.** Synthetic queries overstated the need for LLM augmentation (90% vs 27.8% on real traffic) [hussain-2026-coverage-illusion-rag]. Eval sets drift from production [levit-2026-proxydrift].
- **Calibration transfer.** Early-outcome predictors fail to stay calibrated for specific target agents [cao-2026-early-outcome-prediction-audit].
- **Simulation and vendor flags.** TACIT-Switch's headline is labelled simulation [lei-2026-tacit-switch]. LangSmith [gola-2026-langsmith-fine-tuning] and TAO [databricks-2025-tao] are vendor-reported on internal benchmarks.

### Open problems

- OPE estimators for multi-step agent policies that are validated against live outcomes. Current options are world models [liu-2026-adwm-ope-llm-agents] and tool-choice OPE [li-2026-counterfactual-tool-ranking].
- Logging designs with propensities, so customer logs support OPE. Only [li-2026-counterfactual-tool-ranking] uses exact propensities, and [sun-2023-datacope] asks when OPE is supportable at all.
- Label budgeting: where to spend human labels to correct judge bias [chae-2026-optimal-sequential-annotations-ope] [miranda-2024-hybrid-preferences] [lai-2026-saverouter].
- Closing the router-plus-distillation loop with quality gates [yao-2026-mera] [liu-2026-harness-native-routing].

## C7. Quality signals without ground truth

### Key ideas

Routing and escalation need a quality estimate at decision time, before ground truth exists. The digest shows five lines of work on this.

1. **Self-signals from the cheap model.** These include token or sequence uncertainty [gupta-2024-lm-cascades-token-uncertainty] [ramirez-2024-two-tier], verbalised or trained confidence [chuang-2024-self-ref] [zhang-2026-corea] [tian-2023-just-ask-for-calibration], sampling agreement [yue-2023-mot-cascades] [soiffer-2025-semantic-agreement-cascades] [wang-2022-self-consistency], and self-verification [aggarwal-2023-automix]. Agent-specific versions apply these per step: uncertainty-gated deferral [piatrashyn-2026-redact], inter-rollout action agreement [sethi-2026-trace-action-agreement], and trajectory-level self-assessment [bouchard-2026-trajectory-uq-agents].
2. **Internal-state probes.** Linear probes on activations predict correctness before generation [lugoloobi-2026-llms-encode-failures] [cencerrado-2025-no-answer-needed] [varshney-2026-prefill-activation-router], approximate semantic entropy in one pass [kossen-2024-semantic-entropy-probes], verify reasoning steps [ni-2025-uncertainty-heads] [zhang-2025-reasoning-models-know-right], detect wrong tool calls [healy-2026-tool-selection-hallucination-probes], and abort doomed agent episodes [ruan-2026-early-abort-agent-episodes].
3. **Small trained verifiers, critics and process reward models (PRMs).** Examples are web and agent PRMs [chae-2025-web-shepherd] [choudhury-2025-agentprm-choudhury] [xi-2025-agentprm-xi], multi-agent PRMs [yazdani-2025-masprm] [tsang-2026-orchrm], tiny critics and comparators [gandhi-2026-steer-dont-solve-critic] [jiang-2026-cota-tiny-advisors], distilled fact-checkers [tang-2024-minicheck], and small failure-attribution models [zhang-2025-agentracer] [li-2026-afanet-gnn-attribution] [zhang-2026-agentforesight].
4. **Judge cascades with statistical guarantees.** Cheap judges are escalated to stronger ones under calibrated confidence [jung-2024-trust-or-escalate] [li-2026-jev-as-a-judge]. Juries of small judges [verga-2024-poll-juries], conformal deferral [dou-2026-conformal-cascade], and calibrate-then-threshold escalation [kotte-2026-ucci] [zhang-2026-cauc] [zeighami-2025-bargain] also belong here.
5. **Production and logged feedback as the label source.** These works use implicit user satisfaction [okeefe-2026-cursor-router], sparse rejection feedback [unell-2026-deployment-centered-rejection-risk], thumbs feedback with false discovery rate (FDR) control [lee-2025-online-conformal-abstention], and off-policy evaluation (OPE) from logs collected on another model [bhargava-2024-ope-logged-human-feedback] [xiang-2026-otrope].

These lines depend on each other. Lines 1-3 produce the signals. Line 4 turns a signal into a decision rule with a reliability guarantee. Line 5 supplies labels for calibrating both.

### Strongest methods

The criterion here is signals cheap enough to sit on the critical path of a per-step routing decision.

- **Calibrated per-step risk routers on a distilled small language model (SLM).** In [hemadri-2026-r2v-agent], the router is trained on the SLM's residual failures with a Brier-calibrated, CVaR-constrained objective. It reports escalation rates from 0.60% to 41.7% across three benchmarks. It is the clearest step-level design in the cluster, though it is an arXiv preprint.
- **Calibrate first, then threshold.** [kotte-2026-ucci] uses isotonic calibration of token margins and reports a confidence interval on its cost cut for a 75,000-query production NER workload. That is unusually rigorous reporting. [zhang-2026-cauc] gives a common reliability scale when the cascade mixes heterogeneous models.
- **Telemetry and activation monitors at near-zero cost.** These cost microseconds or a single forward pass: telemetry monitors [dubey-2026-telemetry-failure-detection], recall-controlled probe cascades for early abort [ruan-2026-early-abort-agent-episodes], and pre-generation probes [lugoloobi-2026-llms-encode-failures]. They need white-box access to the model or to telemetry.
- **Black-box signals for API models.** Options include semantic agreement among cheap models [soiffer-2025-semantic-agreement-cascades], small-model generation uncertainty [ramirez-2024-two-tier] (which beat more complex routers in 25 of 27 setups), a draft-model gate that scores a black-box agent [grotov-2026-speculative-uncertainty], and experiential confidence from a single generation [zhang-2026-xconf-experiential-confidence].
- **Execution-grounded verification.** Where tests exist, a test-gated cascade beats both models in the comparison [together-2026-deepswe-cascade-deepseek-gpt] (vendor-reported).
- **Judge cascades with guarantees.** [jung-2024-trust-or-escalate] and [li-2026-jev-as-a-judge] apply when the judge itself is the expensive step.

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| R2V-Agent step router | 94.3% HumanEval+ success with 0.60% LLM escalation; TextWorld 64.6% to 98.2% at 41.7% escalation; 93.3% TerminalBench at 33.9% LLM calls, about half heuristic-router cost | HumanEval+, TextWorld, TerminalBench | heuristic router | [hemadri-2026-r2v-agent] |
| ReDAct deferral | Deferring about 15% of decisions to the large model matches exclusive large-model quality at significantly lower cost | ALFWorld, MiniGrid | exclusive large model | [piatrashyn-2026-redact] |
| Recall-controlled probe cascade | Saves 1.5-8.8x more compute than best single-gate baseline at 90% recall; up to 60.2% (TextCraft) and 54.9% (WebShop) fewer generated tokens at 90% recall | TextCraft, WebShop | best single-gate baseline | [ruan-2026-early-abort-agent-episodes] |
| UCCI calibrated cascade | Cuts inference cost by 31% (95% CI 27-35%) at micro-F1 0.91 on a production NER workload of 75,000 queries (4B/12B LLMs); ECE from 0.12 to 0.03 | production NER workload | entropy thresholds, split-conformal routing, FrugalGPT-style thresholds | [kotte-2026-ucci] |
| Telemetry monitors | Microsecond-cost monitors detect 0.71 of failures at 5% false-alarm (AUROC 0.872), versus per-step LLM judging that costs more than the agent | AFTraj-2K, 2,823 agent episodes | per-step LLM judging | [dubey-2026-telemetry-failure-detection] |
| Cursor per-turn router | Auto Intelligence: above Fable-level user satisfaction at 68% lower cost; Auto Balance: outperforms Opus 4.8 at 41% lower cost with +3% satisfaction (relative to frontier models, measured on live traffic; vendor-reported) | live Cursor traffic, held-out offline split of production turns | frontier models | [okeefe-2026-cursor-router] |
| Pre-generation probes | Routing with probes exceeds the best single model while reducing inference cost by up to 70% on MATH | MATH | best single model | [lugoloobi-2026-llms-encode-failures] |
| JEV judge cascade | Within 3 points of a SOTA LLM judge at 0.36% of its fee; cascade retains 99% of comparator accuracy at lower cost | not stated | SOTA LLM judge | [li-2026-jev-as-a-judge] |
| Cascaded Selective Evaluation | On a Chatbot Arena subset where GPT-4 almost never reaches 80% human agreement, cascaded selective evaluation with models like Mistral-7B guarantees over 80% agreement at almost 80% coverage | Chatbot Arena subset | GPT-4 judge | [jung-2024-trust-or-escalate] |
| Web-Shepherd PRM | About 30 points better accuracy than GPT-4o on WebRewardBench; as verifier on WebArena-lite with GPT-4o-mini policy, 10.9 points better at roughly 10x less cost than GPT-4o-mini as verifier | WebRewardBench, WebArena-lite | GPT-4o; GPT-4o-mini as verifier | [chae-2025-web-shepherd] |
| Semantic entropy probes | Reduces semantic-uncertainty overhead to almost zero versus 5-10x sampling cost of semantic entropy | not stated | sampled semantic entropy | [kossen-2024-semantic-entropy-probes] |
| XConf | Matches or beats ten-sample self-consistency AUROC on 23/24 comparisons at a tenth of the generation cost | not stated | ten-sample self-consistency | [zhang-2026-xconf-experiential-confidence] |
| MiniCheck | GPT-4-level fact-checking accuracy at 400x lower cost (MiniCheck-FT5, 770M) | LLM-AggreFact | GPT-4 | [tang-2024-minicheck] |
| Dropbox judge migration (DSPy) | Adapting the production relevance judge from o3 to gpt-oss-120b with GEPA cut NMSE 45% (8.83 to 4.86); enabled 10-100x more labeling at the same cost | Dropbox Dash relevance judgments (internal) | o3 judge, human labels | [wang-2026-dropbox-dash-dspy] |
| Self-improving cascade audit | Frontier verifier drives blind spot to ~0.05 but escalates 46% of hard-MATH queries vs a 39% true error rate; cascade dashboard shows flat 3% error while true error swings up to 32% | MATH | true error rate | [rajput-2026-cheap-verifiers-blind-spots] |

### Critical notes

- **In-loop metrics can hide degradation.** When a cheap verifier both gates the cascade and measures it, a dashboard can show flat error while true error swings up to 32% [rajput-2026-cheap-verifiers-blind-spots]. Fine-tuning the student on verifier rejections collapsed it in the same study. A self-improving loop needs ground truth held out from the verifier.
- **Calibration does not transfer reliably.** Early-outcome predictors lose calibration on target agents they were not trained on [cao-2026-early-outcome-prediction-audit]. Tool use changes agent calibration [xuan-2026-confidence-dichotomy-tool-agents]. Probes generalise poorly across tasks [orgad-2024-llms-know-more-than-show] and fail out of distribution on math [cencerrado-2025-no-answer-needed]. Token-probability scores are sensitive to how they are aggregated across turns [bouchard-2026-trajectory-uq-agents]. Each new subagent or model swap needs its own recalibration.
- **Predicting failure is not the same as predicting benefit.** Failure probes reach 0.88 AUROC, yet routers still under- or over-escalate when choosing a collaboration protocol [yang-2026-protocol-routing-cost-aware]. Intervention advantage, not failure risk, should drive the decision [zhang-2026-calibration-not-control].
- **Judge bias.** Documented biases include position bias [wang-2023-llms-not-fair-evaluators] and a broader catalogue [ye-2024-calm-judge-biases]. Judges are unreliable for deep-research agents [wang-2026-reflect-judges-research-agents]. Judge-labelled routing data needs debiasing [zhang-2025-meta-router]. Verbalised confidence is unreliable for tool gating [han-2024-uala-uncertainty-aware-agent], and intrinsic self-correction is not a dependable signal [huang-2023-cannot-self-correct].
- **Attribution is weak at step granularity.** On Who&When, the best method reaches 53.5% agent accuracy but only 14.2% step accuracy [zhang-2025-which-agent-when]. Step-level credit signals are unreliable when checked against executed replay [zhang-2026-credit-without-ground-truth].
- **Provenance.** Several headline numbers are vendor-reported or blog-based [okeefe-2026-cursor-router] [together-2026-deepswe-cascade-deepseek-gpt] [vllm-2025-halugate]. Some evaluations are very small: n=50 and n=30 in [sethi-2026-trace-action-agreement], and 257 MBPP tasks in [shaikh-2026-bayesian-self-escalation]. Most records are arXiv preprints.

### Open problems

- Calibrating signals per subagent in heterogeneous pipelines, and recalibrating cheaply after model swaps [cao-2026-early-outcome-prediction-audit] [chuang-2025-confident-or-seek-stronger].
- Black-box step-level signals for closed frontier APIs. Most probe methods need weights, and [grotov-2026-speculative-uncertainty] is an exception.
- Correcting selection bias in implicit and thumbs feedback before it is used as a reward [morandi-2026-selection-bias-sparse-feedback] [liu-2025-user-feedback-noisy-signal].
- Estimating the value of an intervention rather than the risk of failure [zhang-2026-calibration-not-control].
- OPE for multi-step agent policies, as opposed to single responses [bhargava-2024-ope-logged-human-feedback] [chae-2026-optimal-sequential-annotations-ope].

## C8. Distillation and specialization using production data

### Key ideas

The cluster has six lines of work.

1. **Trajectory distillation into SLM agents.** Small agents are trained on teacher trajectories [chen-2023-fireact] [kang-2025-agent-distillation-retrieval-code] [liu-2025-structured-agent-distillation], on synthetic verified data [prabhakar-2025-apigen-mt] [liu-2024-apigen], or with RL from judge rewards [openpipe-2025-art].
2. **Role-level downsizing inside a multi-agent system.** Capacity goes to planning or delegation, and executors are distilled [cai-2026-think-big-search-small] [shen-2024-alpha-umi-multi-llm-agent] [lee-2025-mapcoder-lite] [bijoy-2025-prost-slm-multi-agent]. Small orchestrators are trained to call expensive models only when needed [su-2025-toolorchestra] [yuan-2026-paramanager-small-orchestrator] [guo-2026-sciorch].
3. **Closed-loop distillation behind a router.** Escalation failures are mined, the student is distilled on them, and the router is retrained [guo-2026-routenlp] [yao-2026-mera] [hemadri-2026-r2v-agent] [li-2026-rsi-router]. Online variants train the student from live LLM calls [nie-2024-online-cascade-learning] [ramirez-2023-cache-and-distil] [stogiannidis-2023-cache-me-if-you-can], or store strategies in context instead of updating weights [wu-2025-inter-cascade].
4. **Production-trace specialisation.** Fine-tuning on logged production traffic or traces is reported by [joglekar-2025-intercom-small-llms], [yarullin-2025-intercom-escalation], [gola-2026-langsmith-fine-tuning], [tsymboi-2026-production-traffic-post-training], [selvaraj-2026-adept] and [ayala-2025-slm-vs-llm-low-code-workflows]. [databricks-2025-tao] does it label-free from production prompts.
5. **Non-weight specialisation.** Instead of changing weights, these methods adapt the harness [yang-2026-harness-adaptation-slm], workflow memory [wang-2024-agent-workflow-memory], teacher memory [kim-2026-agent-memory-distillation], or crystallise behaviour into deterministic workflows and code [malik-2026-progressive-crystallization] [lenton-2026-distilling-repeated-work-into-code].
6. **Compressing a multi-agent system into one model.** [luo-2026-agentark] [li-2025-chain-of-agents-afm] [zeng-2026-mixture-of-roles-single-agent]

### Strongest methods

- **Closed-loop route-and-distil.** [guo-2026-routenlp] is the only record with both an enterprise pilot (8 weeks, about 5K queries/day) and a benchmark. [yao-2026-mera] admits a student update only when replay shows quality is preserved, which addresses the collapse failure in [rajput-2026-cheap-verifiers-blind-spots].
- **Executor downsizing guided by a capacity sweep.** [cai-2026-think-big-search-small] shows that the delegation role is the bottleneck, while a 1.7B distilled executor matches a frontier sub-agent. This gives a principled answer to which subagent to shrink first.
- **SLMs that learn when to consult an expert.** [kon-2026-swe-protege] reaches 42.4% on SWE-bench Verified with about 4 expert calls per task. [hemadri-2026-r2v-agent] pairs distillation with a calibrated handoff.
- **Component replacement from production data.** Intercom's escalation classifier with LLM fallback was validated by a production A/B test [yarullin-2025-intercom-escalation] (company-reported). ServiceNow's fine-tuned SLMs beat prompted GPT-4o on structured low-code generation [ayala-2025-slm-vs-llm-low-code-workflows].
- **Harness adaptation before fine-tuning.** [yang-2026-harness-adaptation-slm] recovers most LLM performance at a small fraction of the cost on routine business tasks, without any training.

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| RouteNLP closed loop | 8-week enterprise pilot (~5K queries/day): 58% lower inference cost at 91% response acceptance, p99 latency 1,847 ms to 387 ms; 40-85% cost reduction retaining 96-100% quality on a six-task benchmark | six-task finance/customer-service/legal benchmark; enterprise pilot | not stated | [guo-2026-routenlp] |
| Distilled executor | Scaling delegation backbone improves EM ~11 points vs ~2.6 for execution; a 1.7B distilled executor matches a frontier sub-agent with 37% fewer sub-agent tokens | five multi-hop QA benchmarks | frontier sub-agent | [cai-2026-think-big-search-small] |
| MERA | Deployed policy retains 88.3% pass at 60.8% of always-teacher cost on HumanEval+MBPP; Qwen2.5-Coder-1.5B improves 28.7% to 49.7% after four cycles | HumanEval+, MBPP | always-teacher | [yao-2026-mera] |
| SWE-Protege | Qwen2.5-Coder-7B reaches 42.4% Pass@1 on SWE-bench Verified (+25.4% over prior SLM SOTA) using ~4 expert calls per task and 11% of tokens | SWE-bench Verified | prior SLM SOTA | [kon-2026-swe-protege] |
| Harness adaptation | best small-model agent recovered 89.7% of LLM performance at 4% of the cost; optimized harnesses improved 16 of 21 task-SLM pairs, closing the gap in seven | seven business-oriented agentic tasks | LLM agent | [yang-2026-harness-adaptation-slm] |
| Intercom escalation classifier | Company-reported: custom ModernBERT handles 90% of cases with LLM fallback for 10%; >98% escalation accuracy; latency -0.5s; cost per resolution ~-3% | production A/B test | LLM escalation decision | [yarullin-2025-intercom-escalation] |
| Intercom fine-tuned SLM | Company-reported: fine-tuned Qwen3 14B matched GPT-4.1 answer rate (63.4%); 5% and 12.5% cost reductions on two components | internal issue detection and extraction tasks | GPT-4.1 | [joglekar-2025-intercom-small-llms] |
| LangSmith trace SFT | Engine agent: Kimi K3 + SFT 96.0 vs Kimi K3 90.0 vs GPT-5.6 Sol 87.0 on internal IssueBench subset; OpenSWE Review: Qwen-3.8-27B + SFT F1 48.9%->53.7% with 29.8% fewer model calls and 29.4% fewer tool requests per review (vendor-reported, internal agents) | IssueBench (internal), internal PR review set | base model; GPT-5.6 Sol | [gola-2026-langsmith-fine-tuning] |
| Progressive crystallization | Deterministic execution rose 0% to 45% over eight months; per-incident agent cost reduced >70% despite doubling incident volume (production cloud-networking AIOps) | production AIOps | not stated | [malik-2026-progressive-crystallization] |
| ADEPT | 41.5% reduction in LLM calls and 39.8% cost savings over two months in ITOps and cybersecurity | not stated | not stated | [selvaraj-2026-adept] |
| TAO | Brings Llama 3.1 8B and Llama 3.3 70B close to GPT-4o and o3-mini on FinanceBench, DB Enterprise Arena and BIRD-SQL without labels, at the base model's inference cost | FinanceBench, DB Enterprise Arena, BIRD-SQL (Databricks dialect) | GPT-4o, o3-mini | [databricks-2025-tao] |
| Distilled Flash cascade | Cascade (GLM-5.3 Flash first, escalate to GLM-5.3 when tests reject): 80.9% of DeepSWE tasks at $1.70 vs GLM-5.3 alone 69.0% at $3.99 (57% lower cost); Flash 17x cheaper per rollout (vendor-reported) | DeepSWE | GLM-5.3 alone | [together-2026-deepswe-cascade-glm-flash] |
| ToolOrchestra | On HLE 37.1% vs GPT-5 35.1% while 2.5x more efficient; on tau2-Bench and FRAMES surpasses GPT-5 using about 30% of the cost | HLE, tau2-Bench, FRAMES | GPT-5 | [su-2025-toolorchestra] |
| Fine-tuned SLM vs prompted LLM | Fine-tuning an SLM improves quality by 10% on average over prompting GPT-4o | low-code workflow generation (JSON) | prompted GPT-4o | [ayala-2025-slm-vs-llm-low-code-workflows] |
| LoRA Land | 4-bit LoRA fine-tuned models beat base models by 34 points and GPT-4 by 10 points on average; 25 LoRA Mistral-7B models served on one A100 | 31 tasks | base models, GPT-4 | [zhao-2024-lora-land] |

### Critical notes

- **Gains are concentrated on structured, repetitive tasks.** Fine-tuning wins on functional representation and SQL but stays behind GPT-4 on GSM8K [hakhamaneshi-2023-anyscale-finetuning-llama2]. Harness adaptation works best on repetitive workflows and with capable-enough SLMs [yang-2026-harness-adaptation-slm]. SLMs struggle with long trajectories even after role specialisation [bijoy-2025-prost-slm-multi-agent]. Skill frameworks need roughly 30B+ models [xu-2026-agent-skills-slm-industrial]. Dropping SLMs into LLM-designed frameworks gives resolution rates near zero [tripathy-2025-swenergy].
- **Distillation mostly costs consistency.** A distilled Flash model loses mainly first-try reliability and breaks passing tests slightly more often [together-2026-deepswe-cascade-glm-flash]. Small students learn poorly from teachers far above their capacity [li-2025-small-model-learnability-gap].
- **Self-improving loops can collapse.** Corrective fine-tuning on cheap-verifier rejections degraded the student while in-loop metrics looked flat [rajput-2026-cheap-verifiers-blind-spots]. Curation matters: a less selective first trace dataset lowered F1 [gola-2026-langsmith-fine-tuning].
- **Production savings can be modest.** Intercom reports only 5% and 12.5% cost reductions [joglekar-2025-intercom-small-llms] and about -3% cost per resolution [yarullin-2025-intercom-escalation]. Both are company-reported. Headline claims such as "two orders of magnitude" [dennis-2026-compiling-workflows-into-weights] and "10x-100x" [sharma-2025-slm-agentic-survey] come from a preprint title and a survey abstract respectively.
- **Multi-agent structure is not free for small models.** Single tool-using agents give a better cost balance than multi-agent setups at sub-10B scale [wang-2026-rethinking-scale-slm-agent-paradigms].
- **Provenance.** Much of the production evidence is vendor or blog material [databricks-2025-tao] [gola-2026-langsmith-fine-tuning] [aws-bedrock-model-distillation] [openpipe-2025-art]. [selvaraj-2026-adept] does not state a baseline.

### Open problems

- Safe continual distillation: admission gates that use independent ground truth rather than the cascade's own verifier [yao-2026-mera] [rajput-2026-cheap-verifiers-blind-spots].
- Curating traces from implicit feedback without bias [wang-2025-drift-dissatisfaction] [gola-2026-langsmith-fine-tuning].
- Deciding which pipeline role to distil first, beyond search agents [cai-2026-think-big-search-small] [apparaju-2026-compute-allocation-retrieval-agents].
- Re-distilling as teacher models change, and keeping many specialists served cheaply [sheng-2023-s-lora] [chen-2023-punica].
- Controlled comparisons of weight distillation, harness adaptation and crystallisation on the same workload. None exists in this cluster.

## C9. Reasoning and token budget control

### Key ideas

The cluster has five lines of work.

1. **Query-level think/no-think and effort routing.** These methods decide per query whether to reason and how much [wang-2025-when-to-reason] [zhang-2025-synapseroute] [lou-2025-adacot] [lee-2026-dart] [lu-2025-car-certainty-routing]. Some route jointly over model and budget [xue-2026-r2-router] [fernandez-2025-radar] [pan-2025-route-to-reason] [ding-2025-best-route] [damani-2024-learning-how-hard-to-think].
2. **Step- and turn-level effort for agents.** This includes per-step effort selection [yang-2026-ares-reasoning-effort], learned per-turn budgets under a global cap [jali-2026-turn-adaptive-budgets], step-level cognitive depth [yang-2026-cogrouter], learning when to plan [paglieri-2025-learning-when-to-plan], and stepwise small/large model routing [lee-2025-steer] [kapoor-2026-trim-stepwise-routing] [si-2026-policy-guided-stepwise-routing].
3. **Early exit and termination.** Methods stop reasoning on confidence or answer stability [yang-2025-deer-dynamic-early-exit] [fu-2024-dynasor-certaindex] [wang-2026-conformal-thinking] [dong-2026-learning-to-stop-cost-study], or stop agent runs that are unlikely to improve [guo-2026-eet-early-termination] [pham-2026-agentstop].
4. **Budget allocation across pipeline phases.** Approaches include knapsack allocation [hamri-2026-zebra-budget-allocation], per-subtask search [wang-2025-agenttts] [wang-2025-agent-reinforce] [lin-2025-plan-and-budget], and trajectory budgets exposed to the model [anthropic-task-budgets-docs].
5. **Training models to reason concisely.** These use RL or SFT for length control [aggarwal-2025-l1-lcpo] [he-2025-thinkdial] [zhang-2025-adaptthink] [munkhbat-2025-self-training-concise].

Measurement work frames all of this. [bai-2026-agents-spend-your-money] reports that coding agents use about 1000x more tokens than code chat and that runs on the same task vary up to 30x. [weinberger-2026-prompt-induced-waste-coding-agents] argues for cost per successful task as the metric.

### Strongest methods

- **Per-step effort routing for agents.** [yang-2026-ares-reasoning-effort] cuts reasoning tokens by up to 52.7% on three agent benchmarks with minimal degradation in task success. [jali-2026-turn-adaptive-budgets] gives easy turns small budgets and saves tokens for hard turns under a global cap. Both map directly onto a pipeline that already calls one frontier model with an effort knob.
- **Low effort with escalation on failure.** Vendor-measured results show low effort with failures re-run at high effort beats all-high effort on cost [anthropic-2026-optimizing-cost-and-intelligence]. Routine office agents complete work at low effort with no rule violations [xu-2026-low-effort-office-agents].
- **PRM-flagged step escalation.** [kapoor-2026-trim-stepwise-routing] escalates only the critical steps, and [lee-2025-steer] does so without a trained router.
- **Experience-driven early termination for coding agents.** [guo-2026-eet-early-termination] reports at most 0.2% resolution loss on SWE-bench Verified.
- **Production think/no-think triggering.** [lou-2025-adacot] was evaluated on a production traffic test set, and [lyu-2026-differential-reasoning-router] in a production e-commerce workflow.

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| Ares per-step effort | Reduces reasoning token usage by up to 52.7% vs fixed high-effort reasoning with minimal degradation in task success (TAU-Bench, BrowseComp-Plus, WebArena) | TAU-Bench, BrowseComp-Plus, WebArena | fixed high-effort reasoning | [yang-2026-ares-reasoning-effort] |
| Turn-adaptive budgets | Saves up to 35% tokens and reduces latency up to 30% while maintaining accuracy; plan-aware variant saves up to 40% tokens over baselines | diverse agentic benchmarks | not stated | [jali-2026-turn-adaptive-budgets] |
| Low effort, retry at high | on SWE-bench Pro, Opus 5.5 low effort with failures re-run at high reached 97% at ~$0.17/task vs 95.3% at $0.29 all-high | SWE-bench Pro | all-high effort | [anthropic-2026-optimizing-cost-and-intelligence] |
| Solo model vs orchestrator | on full BrowseComp, solo Fable 5 at lower effort was 22-30% cheaper than the orchestrator at same accuracy | BrowseComp | Fable 5 coordinator + Sonnet 5 worker orchestrator | [anthropic-2026-optimizing-cost-and-intelligence] |
| Orchestrator-assigned tiers/effort | DeepSWE v1.1: Replit Agent 72% at $2.11/task vs GPT-6 Astra (mini-swe-agent) 67% at $1.60 (low effort) and 74% at $4.43 (xhigh); sidekick architecture 61% at $1.34 | DeepSWE v1.1 | GPT-6 Astra single-model baselines | [furman-2026-replit-free-the-models] |
| Per-role subagent model swap | Librarian subagent moved from Sonnet 4.6 (medium) to GPT-5.5 (no reasoning): mean latency 237s->81s, F1 0.47->0.48, average cost $1.21->$0.69 (43% cheaper) on internal eval | Amp internal subagent evals | Sonnet 4.6 (medium) | [amp-2026-subagent-model-assignment] |
| ZEBRA phase budgets | At 0.5 of unconstrained spend, recovers 94.4% of unconstrained quality vs 88.1% for LLM-direct on APPS; beats LLM-direct by 14.3pp on 3-phase HotpotQA pipeline | APPS, HotpotQA | LLM-direct allocation | [hamri-2026-zebra-budget-allocation] |
| TRIM PRM step routing | On MATH-500 simplest threshold policy gives 5x higher cost efficiency than prior routing methods; advanced policies match the strong model using 80% fewer expensive-model tokens; up to 6x higher cost efficiency on AIME | MATH-500, AIME | prior routing methods; strong model | [kapoor-2026-trim-stepwise-routing] |
| STEER stepwise routing | Up to +20% accuracy with 48% fewer FLOPs vs using only the larger model on AIME | AIME | larger model only | [lee-2025-steer] |
| EET early termination | Total cost reduced 19%-55% (32% average) with at most 0.2% loss in resolution rate; API calls, input and output tokens down 21%, 30%, 25% | SWE-bench Verified | not stated | [guo-2026-eet-early-termination] |
| Learned stopping, cost-aware | Learned stopping gives +3.2 to +21.2 pp additional token saving on Qwen3 free-form math; the same policy saves 32% tokens with KV-cache forking but costs 121% extra under black-box repeated prefilling | GSM8K, MATH-500, MMLU-Pro, AIME-90, GPQA | not stated | [dong-2026-learning-to-stop-cost-study] |
| AdaCoT | On a production traffic test set, reduced CoT triggering rate to as low as 3.18% and average response tokens by 69.06% while maintaining performance on complex tasks | production traffic test set | not stated | [lou-2025-adacot] |
| Differential Reasoning Router | In a production e-commerce workflow, reaches accuracy parity with the strongest confidence-based router with more than 60% reasoning-token cost savings | production e-commerce annotation workflow | strongest confidence-based router | [lyu-2026-differential-reasoning-router] |
| CogRouter | With Qwen2.5-7B, 82.3% success on ALFWorld/ScienceWorld, beating GPT-4o (+40.3%) and o3 (+18.3%) while using 62% fewer tokens | ALFWorld, ScienceWorld | GPT-4o, o3 | [yang-2026-cogrouter] |
| Low effort for office agents | Low reasoning level used about 43% fewer output tokens and 20% less time than max level, with no rule violations across 840 runs of 14 office tasks | 14 routine office tasks (840 runs) | max reasoning level | [xu-2026-low-effort-office-agents] |

### Critical notes

- **Budget control can hurt hard cases.** No model thinks optimally, and methods that fix overthinking worsen underthinking [aggarwal-2025-optimalthinkingbench]. Under-reasoning yields incomplete answers [mia-2026-adaptive-reasoning-agentic-ai]. Concise CoT carried a 27.69% math penalty for GPT-3.5 [renze-2024-concise-cot]. Effort reduction modes trade accuracy, for example under 15% degradation at a 75% token cut [he-2025-thinkdial]. Inference-scaling gains diminish on hard tasks [balachandran-2025-inference-time-scaling-complex-tasks].
- **More compute can hurt.** Longer thinking can lower accuracy [gema-2025-inverse-scaling-test-time]. Parallel sampling beats extended thinking at equal budget [ghosal-2025-mirage-test-time-scaling]. Overthinking correlates with lower success in coding agents [cuadron-2025-danger-of-overthinking-agents]. Test-time scaling does not reliably help general agents [li-2026-test-time-scaling-agents].
- **Savings depend on the serving regime.** The same stopping policy saves 32% with KV-cache forking but costs 121% extra under black-box repeated prefilling [dong-2026-learning-to-stop-cost-study]. Switching models mid-trajectory causes cache misses that the Cursor router prices explicitly [okeefe-2026-cursor-router]. Input tokens dominate coding-agent cost [bai-2026-agents-spend-your-money], so savings on reasoning tokens alone can be small at the system level.
- **Multi-agent gains often reflect unaccounted compute.** At equal thinking budgets, single agents match or beat multi-agent systems on multi-hop reasoning [tran-2026-single-vs-multi-agent-equal-budget] [wang-2024-reasoning-token-economies]. A contrary compute-matched result exists [wunderlich-2026-multi-agent-pareto-tts]. A vendor also found delegation to cheaper workers losing to a solo model at lower effort [anthropic-2026-optimizing-cost-and-intelligence].
- **Effort is a model-specific API contract.** Defaults and costs differ across models [moon-2026-price-of-thinking]. No-think modes leak reasoning [wang-2025-demystifying-hybrid-thinking].
- **Provenance.** The Replit, Amp and Anthropic numbers are vendor-reported [furman-2026-replit-free-the-models] [amp-2026-subagent-model-assignment] [anthropic-2026-optimizing-cost-and-intelligence]. The vLLM router discloses no benchmark or baseline [vllm-2025-semantic-router]. Most of the training-based methods are math-centric arXiv preprints.

### Open problems

- Jointly controlling per-step effort, model tier and trajectory budget in multi-agent pipelines. Current work handles one or two of these [anthropic-task-budgets-docs] [jali-2026-turn-adaptive-budgets] [hamri-2026-zebra-budget-allocation].
- Effort routers that guard hard cases with explicit risk bounds [wang-2026-conformal-thinking] [farzaneh-2026-tsds-edge-deferral].
- Predicting token spend before a run [bai-2026-agents-spend-your-money] [smekal-2026-task-spec-token-spend].
- Evaluation on enterprise agent workloads beyond math and coding [xu-2026-low-effort-office-agents].
- Settling whether external routers or the orchestrator itself should assign effort [furman-2026-replit-free-the-models] [okeefe-2026-cursor-router].

## C10. Caching, context and prompt efficiency

### Key ideas

Six connected lines of work cut what each agent step costs without changing which model runs it.

1. **Provider prefix caching and cache-friendly context layout.** Providers bill cached input at a fraction of the normal rate: cache reads at 0.1x with priced writes [anthropic-2024-prompt-caching], up to 95% off cached input [openai-2024-prompt-caching], and 75% off implicit hits [kilpatrick-2025-gemini-implicit-caching]. Practitioners treat KV-cache hit rate as the main cost metric. They keep prefixes stable, keep context append-only, and mask tools instead of removing them [ji-2025-manus-context-engineering]. Where cache breakpoints sit decides whether caching helps [lumer-2026-dont-break-the-cache]. Follow-up work designs compaction [xu-2026-tokenpilot], tool discovery [zha-2026-cacherouter] and keepalive pings [khailo-2026-keepalive-economics] so they do not break the cached prefix.
2. **Semantic and agent-level result caching.** The simplest form keys an embedding-similarity cache on the query [bang-2023-gptcache]. Agent work moves the cache key: plan templates adapted by a small model [zhang-2025-agentic-plan-caching], structured intermediate artifacts [chillara-2026-semanticalli], response templates for structurally similar prompts [chakraborty-2025-gencache], canonicalized structured intent [basu-2026-w5h2-intent-caching] and tool results keyed on the whole call prefix [kumar-2026-tvcache]. Some caches add verification: per-entry thresholds learned online under an error bound [schroeder-2025-vcache], or LLM-judge checks run off the critical path [singh-2026-krites].
3. **Context compression and trajectory pruning.** Prompt compressors drop tokens before the expensive call [jiang-2023-llmlingua; pan-2024-llmlingua-2]. Agent-specific methods prune trajectories [xiao-2025-agentdiet-trajectory-reduction], learn compression guidelines from failures [kang-2025-acon], fold sub-trajectories [sun-2025-context-folding], or prune tool output with small models [kovacs-2026-squeez; wang-2026-swe-pruner; chen-2026-coact]. Simply masking old observations is a strong baseline [lindenbauer-2025-complexity-trap].
4. **Tool-context reduction.** Tools can be registered lazily [zhang-2024-ecoact], loaded through deferred tool search [anthropic-2025-advanced-tool-use], pulled into dynamic context [katz-2026-cursor-dynamic-context-discovery] or pre-routed deterministically [singla-2026-agentweave].
5. **Communication efficiency in multi-agent systems (MAS).** Methods prune redundant messages [zhang-2024-agentprune], route context by role [liu-2025-rcr-router], schedule when agents are active [dubey-2026-psmas-phase-scheduled] and attribute memory-injection cost [singh-2026-total-cost-of-agency]. Homogeneous multi-agent workflows collapse into a single agent that reuses its KV cache [xu-2026-oneflow-single-agent-baseline].
6. **Experience reuse so cheaper models can take over.** A strong model's resolutions can be stored for a weak model to reuse [wu-2025-inter-cascade]. Other routes are hierarchical teacher memory [kim-2026-agent-memory-distillation], record-and-replay [feng-2025-agentrr], compiling repeated trajectories into code [lenton-2026-distilling-repeated-work-into-code] and adapting the harness so small models suffice [yang-2026-harness-adaptation-slm]. Harness design alone can move cost more than the choice of model [ali-2026-harness-effect; katz-2026-cursor-token-efficiency].

### Strongest methods

- **Cache-aware prompt layout.** This is the best-supported lever. It was tested over 500+ sessions on several providers, comparing three caching strategies [lumer-2026-dont-break-the-cache]. A 99.3% hit rate in real telemetry made the API cheaper per token than on-prem [peng-2026-inference-economics-coding-agents]. Production harness changes aimed at cache reuse are reported from production traffic [katz-2026-cursor-token-efficiency].
- **Observation masking.** It is simple and reproducible on SWE-bench Verified, matches LLM summarization at about half the cost of the raw agent, and should be the default baseline for any compression method [lindenbauer-2025-complexity-trap].
- **Trajectory pruning (AgentDiet).** It reduces cost on two coding-agent benchmarks while keeping the resolve rate [xiao-2025-agentdiet-trajectory-reduction].
- **Cache-compatible compaction.** It replaces volatile values with stable placeholders and evicts in batches, so compression and caching compound instead of fighting [xu-2026-tokenpilot].
- **Verified semantic caching.** vCache is the only semantic cache here with a user-set error bound [schroeder-2025-vcache].
- **Plan caching.** It is the clearest agent-specific reuse method, and it lets a small model replace frontier planning on repeated task types [zhang-2025-agentic-plan-caching].

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| Prompt-cache block placement | prompt caching reduced API cost by 41-80% and improved time-to-first-token by 13-31% across providers; naive full-context caching can paradoxically increase latency | DeepResearch Bench | not stated | [lumer-2026-dont-break-the-cache] |
| Context compression, cost audit | Largest compression cut tool-output tokens by 38.4% but increased billed cost by 6.8%; token reduction correlated weakly with cost reduction (r=0.15) | SWE-bench Go subset, controlled coding tasks | not stated | [weinberger-2026-token-reduction-not-cost] |
| Observation masking | simple observation masking roughly halves cost relative to the raw agent while matching (sometimes slightly exceeding) the solve rate of LLM summarization on SWE-bench Verified | SWE-bench Verified | raw agent; LLM summarization | [lindenbauer-2025-complexity-trap] |
| AgentDiet trajectory reduction | input tokens reduced 39.9%-59.7% and total computational cost reduced 21.1%-35.9% while maintaining agent performance | SWE-bench Verified, Multi-SWE-bench | not stated | [xiao-2025-agentdiet-trajectory-reduction] |
| Agentic Plan Caching | 50.31% average cost reduction and 27.28% average latency reduction while maintaining performance across real-world agent applications | multiple real-world agent applications | not stated | [zhang-2025-agentic-plan-caching] |
| TokenPilot | costs reduced 61% and 56% in isolated mode and 61% and 87% in continuous mode on PinchBench and Claw-Eval with competitive performance | PinchBench, Claw-Eval | not stated | [xu-2026-tokenpilot] |
| CacheRouter | Cache hit rates 90.99% and 95.2%, cutting input cost to ~12.0% and ~8.0% of a no-cache baseline under DeepSeek pricing (55 queries and a 30-turn dialogue) | not stated | no-cache baseline | [zha-2026-cacherouter] |
| vCache | up to 12.5x higher cache hit rate and 26x lower error rate than static-threshold and fine-tuned-embedding baselines while meeting the user-specified error bound | four released semantic-cache benchmarks | static-threshold and fine-tuned-embedding baselines | [schroeder-2025-vcache] |
| Semantic-cache eviction study | raw semantic-cache hit rates of 51-60% on LMSYS and QQP fall to quality-adjusted rates of 1.1-2.2% when only answer-substitutable hits are counted | LMSYS, QQP | not stated | [kulkarni-2026-cache-eviction-study] |
| Prompt caching, real telemetry | Prompt caching (99.3% hit rate) cut realized API cost 88.6% to $0.57 per million tokens | production monorepo, two 28-day periods | not stated | [peng-2026-inference-economics-coding-agents] |
| Compression gateway attribution | Tool-schema filtering removes ~21K-57K tokens per turn; content compression saves only ~2% of cache-priced prefix per turn (real Claude Code/Codex sessions) | not stated | not stated | [chen-2026-compression-gateway-cost] |
| Harness swap | Harness cuts blended cost per task 41% ($0.21->$0.12), wall-clock 44%, tokens 38% at quality parity (0.78->0.81) across six models on 22 tasks | 22 locked enterprise evaluation tasks | conventional production loop | [ali-2026-harness-effect] |
| ACON | peak token usage reduced by 26-54% while improving task success over compression baselines | AppWorld, OfficeBench, Multi-objective QA | compression baselines | [kang-2025-acon] |
| AgentPrune | Comparable results to SOTA topologies at $5.6 vs $43.7; 28.1%-72.8% token reduction when plugged into existing MAS | six benchmarks | SOTA topologies | [zhang-2024-agentprune] |

### Critical notes

- **Semantic-cache false hits are the central risk.**
  - Most hits at standard thresholds cannot actually substitute for a fresh answer: the quality-adjusted hit rate is 1.1-2.2% [kulkarni-2026-cache-eviction-study].
  - Surface-similarity caches miss the user's intent. GPTCache matched intent 37.9% of the time against 91.1% for intent canonicalization [basu-2026-w5h2-intent-caching].
  - Semantic caching breaks on queries with many parameters [merchant-2026-temporal-semantic-cache-mcp].
  - Caches can be poisoned through key collisions [zhang-2026-cacheattack; zhang-2026-semantic-cache-poisoning].
  - Semantic-cache models should be judged at the thresholds used in deployment, not by offline ranking [baral-2026-semantic-cache-operational-gap].
- **Fewer tokens does not mean a smaller bill.**
  - Compression set off extra retrieval and extra turns, and billed cost rose [weinberger-2026-token-reduction-not-cost].
  - Once prompt caching is on, content compression saves only about 2% of the cache-priced prefix [chen-2026-compression-gateway-cost].
  - Policies using about a third of the tokens ran 20-80% longer [satish-2026-beyond-token-savings].
  - LLM summarization can lengthen trajectories [lindenbauer-2025-complexity-trap].
- **Switching models invalidates prompt caches.**
  - A per-turn router lowered cache hit rate from 97.8% to 94.2% [fireworks-2026-firerouter-with-opus].
  - On long tool-heavy sessions, the highest-priced model can cost less than the next tier down because of cache rebuilds [kwartler-2026-harness-tokenomics].
  - Cursor deliberately limits switching models for subagents [katz-2026-cursor-token-efficiency].
  - Removing tools mid-session breaks the cached prefix, so Manus masks them instead [ji-2025-manus-context-engineering].
- **Caching everything can backfire.** Naive full-context caching can increase latency [lumer-2026-dont-break-the-cache].
- **Compaction can silently erase safety constraints** [chen-2026-governance-decay].
- **Memory and skill modules often do not pay for themselves.**
  - A plain actor given the same token budget matches or beats them [hajimiri-2026-skill-memory-worth-tokens].
  - Persistent memory barely changed budget use, which is vendor-reported [lenton-2026-continual-arc].
  - Agent-written skills can raise cost [hu-2026-cost-inefficient-coding-agents].
- **Some vendor figures are illustrative or from internal tests:**
  - 98.7% saving on a single example workflow [anthropic-2025-code-execution-with-mcp]
  - internal tool-search results [anthropic-2025-advanced-tool-use]
  - an 84% token cut from context editing [anthropic-2025-context-management]
- **Prompt caching can leak information across users** through timing [gu-2025-auditing-prompt-caching].

### Open problems

- There is no cache-validity model for agent steps whose outputs depend on changing external state. Plan caching argues that chatbot-style caches fail here [zhang-2025-agentic-plan-caching], and only tool-value caching keyed on the full call prefix addresses state correctness [kumar-2026-tvcache].
- No method optimizes compression and caching together against billed cost per solved task, the metric that weinberger-2026-token-reduction-not-cost asks for.
- Verified semantic caches have not been tested on multi-step agent traffic. vCache uses query-level benchmarks [schroeder-2025-vcache].
- Compaction has no guarantees that it keeps constraints [chen-2026-governance-decay].
- The cost of a cache rebuild when switching models mid-session has been derived only for coding harnesses [kwartler-2026-harness-tokenomics].

## C11. Serving and systems for agentic workloads

### Key ideas

1. **KV-centric serving for agent loops.** Systems manage the KV cache across tool-call pauses [abhyankar-2024-infercept; li-2025-continuum; bian-2025-tokencake; xia-2026-mori]. Others evict and prefetch based on the workflow structure [pan-2025-kvflow; zhang-2026-cachescout; zheng-2026-pbkv; ni-2026-topas]. A third group shares KV across agents [ye-2025-kvcomm; bian-2026-tokendance] and across LoRA agents or specialized models [jeon-2026-lragent; wang-2026-forkkv; woo-2026-prefillshare]. All of this builds on automatic prefix reuse [zheng-2023-sglang] and paged KV memory [kwon-2023-vllm].
2. **Program-level scheduling.** The whole agent program, not the single call, becomes the scheduling unit [luo-2025-autellix; lin-2024-parrot; ni-2025-astraea; kang-2026-thunderagent].
3. **Workflow-level model selection and placement under SLOs.** This line connects most directly to step-level routing. It covers:
   - choosing a model per stage with a MILP [gravara-2026-atlas]
   - profile-guided mapping of components to models and hardware [chaudhry-2025-murakkab]
   - just-in-time reassignment driven by load [dai-2025-aragog]
   - switching among Pareto-optimal configurations [gravara-2026-compass]
   - runtime model selection driven by SLOs [gravara-2026-plaiground]
   - switching between physical plans [qian-2026-dyserve]
   - joint per-stage routing and load balancing [ni-2026-chimera; kabir-2026-inframind]
   - session-aware cascading inside an agent session [ren-2025-agserve]
   - compile-time profiling of each sub-agent configuration [li-2026-flowcompile]
4. **Speculation above the token level.** Speculation is applied to actions [ye-2025-speculative-actions], plans [guan-2025-dynamic-speculative-planning], verification [ro-2025-sherlock], specific action types [zhong-2026-dualspec] and tool execution [sui-2026-paste].
5. **Local/remote splitting.** The frontier model decomposes the task and a small model executes it [narayan-2025-minions; dong-2025-hybridflow].
6. **Cheaper serving tiers.** Batch APIs cost 50% less [anthropic-message-batches; openai-batch-api; google-gemini-batch-api]. An internal platform routes offline work to batch APIs [baranowski-2025-instacart-maple]. Other options are self-hosting economics [ivanovici-2025-intercom-cost-of-serving] and multi-LoRA serving [zhao-2024-lora-land; sheng-2023-s-lora].
7. **Workload characterization.** Production traces show long autonomous loops with idle gaps [liu-2026-copilot-traces], heavy cross-request redundancy [chang-2026-agentic-workload-characterization] and real KV hit-rate ceilings [wang-2025-kvcache-in-the-wild].

### Strongest methods

- **AgentCARD.** It directly measures how assigning a model per role, and choosing where each is hosted, moves the cost-accuracy frontier. It uses Shapley diagnostics to find which role is the bottleneck [jiang-2026-agentcard].
- **Aragog.** It is the clearest serving-side analogue of step routing. It finds the configurations that preserve accuracy once, then picks among them per stage based on load [dai-2025-aragog].
- **Continuum.** Its KV time-to-live is evaluated on SWE-Bench, BFCL and OpenHands agents, which are realistic tool-calling loops [li-2025-continuum].
- **KVFlow and CacheScout.** They are well-specified KV policies aware of multi-agent structure. CacheScout learns agent transitions online instead of needing a predefined graph [pan-2025-kvflow; zhang-2026-cachescout].
- **Minions.** It reports both quality retention and remote cost, which is rare in this cluster [narayan-2025-minions].
- **Production-scale traces.** The Copilot traces cover 761M LLM calls [liu-2026-copilot-traces], and chang-2026-agentic-workload-characterization adds traces from three production agent applications. Both anchor design assumptions in real data.

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| AgentCARD role/deployment mix | Heterogeneous teams improve accuracy by up to 44% over cost-equivalent homogeneous teams, or match the strongest homogeneous team at up to 12x lower per-task cost via hybrid deployment | not stated | cost-equivalent homogeneous teams | [jiang-2026-agentcard] |
| Atlas | Selects plans within 0.03 of oracle accuracy while reducing deployment cost by up to 42% via heterogeneous placement | not stated | oracle accuracy | [gravara-2026-atlas] |
| Murakkab | Reduces GPU usage by up to 2.8x, energy by 3.7x and cost by 4.3x while maintaining SLOs | diverse agentic workflows | not stated | [chaudhry-2025-murakkab] |
| Aragog | maximum serving throughput increased 42.8-217.0% and median latency reduced 32.5-86.1% at peak load while keeping accuracy comparable to the most expensive configurations | diverse agentic workflows | most expensive configurations | [dai-2025-aragog] |
| AgServe | Comparable response quality to GPT-4o at 16.5% of the cost | not stated | GPT-4o | [ren-2025-agserve] |
| MinionS | MinionS: 5.7x lower cost at 97.9% of remote-only performance | financial, medical and scientific long-document reasoning tasks | remote-only | [narayan-2025-minions] |
| Continuum | average job completion time improved by over 8x with higher throughput on SWE-Bench, BFCL and OpenHands agents | SWE-Bench, BFCL, OpenHands | not stated | [li-2025-continuum] |
| KVFlow | up to 1.83x speedup for single workflows with large prompts and up to 2.19x with many concurrent workflows vs SGLang with hierarchical radix cache | not stated | SGLang with hierarchical radix cache | [pan-2025-kvflow] |
| KVCOMM | over 70% KV reuse rate across multi-agent workloads without quality degradation; up to 7.8x speedup (TTFT ~430 ms to ~55 ms) in a five-agent setting | RAG, math reasoning, collaborative coding | not stated | [ye-2025-kvcomm] |
| CacheScout | KV hit rate +10-18 points, mean TTFT -18-45%, per-turn latency -29-38%, peak throughput up to +57% (TTFT up to -54% on larger models) | real-world multi-agent workloads | not stated | [zhang-2026-cachescout] |
| Autellix | program throughput improved 4-15x at the same latency vs vLLM across diverse LLMs and agentic workloads | not stated | vLLM | [luo-2025-autellix] |
| TokenCake | end-to-end latency reduced by up to 47.06% in memory-constrained settings and effective GPU utilization improved 16.9 points vs vLLM | multi-agent benchmarks | vLLM | [bian-2025-tokencake] |
| Chimera | End-to-end latency reduced 1.2-2.4x and task performance +8.0-9.5 pp on average vs baselines incl. vLLM | code generation and math reasoning workflows | baselines incl. vLLM | [ni-2026-chimera] |
| LLMCompiler | up to 3.7x latency speedup, up to 6.7x cost savings and up to ~9% accuracy improvement vs ReAct | function-calling tasks with varied dependency patterns | ReAct | [kim-2023-llmcompiler] |

### Critical notes

- **Most results are about latency and throughput, not dollars per task or quality.** That holds for KVFlow, CacheScout, TokenCake, Autellix and Continuum [pan-2025-kvflow; zhang-2026-cachescout; bian-2025-tokencake; luo-2025-autellix; li-2025-continuum]. Exceptions that report cost or quality include [jiang-2026-agentcard; narayan-2025-minions; ren-2025-agserve; gravara-2026-atlas].
- **Nearly all of these systems assume self-hosted engines.** A pipeline calling a frontier API cannot change KV eviction. Its only levers are request layout, batch tiers and keepalives [anthropic-message-batches; khailo-2026-keepalive-economics].
- **Mixing models costs KV reuse.** Homogeneous workflows can share one agent's KV cache [xu-2026-oneflow-single-agent-baseline]. Across different models, sharing has been shown only through a shared prefill module [woo-2026-prefillshare] or for LoRA variants of one base model [jeon-2026-lragent; wang-2026-forkkv].
- **Some headline numbers need context.**
  - The 14.6x-106x serving-cost advantage is project-reported and compares against API list prices [vllm-2026-agentx-agentic-serving].
  - The Dynamo figures are vendor- or partner-reported [nvidia-dynamo-repo].
  - Self-hosting is not always cheaper: a Llama-70B distill cost about 1.7x GPT-4.1 [ivanovici-2025-intercom-cost-of-serving], and dedicated on-prem reservation cost 43.8% more [peng-2026-inference-economics-coding-agents].
- **Workflow-atomic scheduling gives up peak throughput** (about 30% lower) to meet SLOs [guo-2026-saga].

### Open problems

- No serving system jointly optimizes routing each step to a different model and keeping KV warm for each model in a multi-model pipeline. INFRAMIND and Chimera route with awareness of load but do not price cross-model cache loss [kabir-2026-inframind; ni-2026-chimera].
- KV reuse across models with different weights is barely explored [woo-2026-prefillshare].
- Idle-gap prediction from production traces [liu-2026-copilot-traces] has not been tied to API-level cache TTL decisions [khailo-2026-keepalive-economics].
- Quality-aware evaluation of serving-side model swaps under load is missing. Aragog and Compass assume configuration sets that preserve accuracy, profiled offline [dai-2025-aragog; gravara-2026-compass].

## C12. Architecture-level cost reduction and when multi-agent is worth it

### Key ideas

**When multi-agent pays.** Google's 260 controlled configurations show multi-agent gains ranging from strongly positive on decomposable tasks to strongly negative on sequential planning. Gains saturate as single-agent capability rises, and tool-heavy tasks pay coordination overhead [kim-2025-science-of-scaling-agent-systems]. MASBENCH finds that gains depend on depth, horizon, breadth, parallelism and robustness [ke-2026-mas-orchestra]. Anthropic's Research system shows a heterogeneous lead-plus-subagents design that buys quality with roughly 15x chat tokens. It is therefore justified only for high-value, parallelizable work [hadfield-2025-multi-agent-research-system]. Anthropic's guidance limits multi-agent designs to context isolation, parallelism or specialization [phillips-2026-when-to-use-multi-agent-systems]. LangChain finds that decomposition helps mainly as context grows [fu-hinthorn-2025-benchmarking-multi-agent-architectures]. Coordination itself has a measurable cost, which falls as capability rises [sun-2026-collaboration-tax].

**Equal-budget critiques.** At equal call budgets, an evolved planner-executor-critic team does not significantly beat a single agent [dylan-2026-equal-inference-cost-mas]. Under equal thinking-token budgets, single agents match or beat MAS on multi-hop reasoning [tran-2026-single-vs-multi-agent-equal-budget]. Self-Refine, Best-of-N and Debate give modest gains at 2-4x tokens, and those gains are no larger on harder items [leins-2026-when-orchestration-pays-off]. Homogeneous debate loses to isolated self-correction [bertalanic-2026-cost-of-consensus], [zhang-2025-stop-overvaluing-mad], and well-prompted single agents match multi-agent discussion [wang-2024-rethinking-bounds-multi-agent-discussion]. Memory and skill modules for web agents lose to a token-matched plain actor [hajimiri-2026-skill-memory-worth-tokens]. One compute-matched study is more favourable: at equal budget, debate and MoA beat self-consistency by small margins [wunderlich-2026-multi-agent-pareto-tts].

**Collapse to one agent, or make heterogeneity the point.** A homogeneous MAS can be simulated by one agent that reuses its KV cache, so heterogeneity is where value remains [xu-2026-oneflow-single-agent-baseline]. Related approaches include a single agent with a skill library [li-2026-single-agent-skills-vs-mas], role steering vectors [zeng-2026-mixture-of-roles-single-agent] and MAS distillation [luo-2026-agentark], [li-2025-chain-of-agents-afm].

**Harness and deterministic offload.** Changing only the harness cut cost per task 41% across six fixed models [ali-2026-harness-effect]. Cursor's small context changes compounded to 7% production savings [katz-2026-cursor-token-efficiency]. Adapting the harness lets SLMs approach LLM performance on routine tasks [yang-2026-harness-adaptation-slm]. Other approaches move work out of the model altogether:
- Promoting validated behaviours to deterministic workflows [malik-2026-progressive-crystallization].
- Deterministic tool-graph routing [bholani-2026-self-healing-router].
- Code nodes inside workflows [xu-2026-hyevo].
- Fixed pipelines such as Agentless [xia-2024-agentless].
- Terminal-only agents [bechard-2026-terminal-agents-suffice].
- In-context self-orchestration [dennis-2026-in-context-obsoletes-orchestration].

**Adaptive architecture per query.** These methods decide per query how much architecture to use:
- Pick single-agent or hierarchical topology by predicted difficulty [hong-2026-dats-difficulty-aware-topology].
- Trigger debate only when it is predicted to fix the answer [fan-2025-imad].
- Cascade between single-agent and MAS [gao-2025-single-or-multi-agent-why-not-both].
- Choose plain LLM versus full agent [wang-2026-boundaryrouter].
- Sample the architecture from a supernet [zhang-2025-maas-agentic-supernet].

**Cost accounting.** Coding agents use about 1000x the tokens of code chat, and repeated runs on the same task vary up to 30x [bai-2026-agents-spend-your-money]. Proposed metrics include cost-of-pass [erol-2025-cost-of-pass], Pareto reporting [kapoor-2024-ai-agents-that-matter], [kapoor-2025-hal-holistic-agent-leaderboard] and cost per successful task [weinberger-2026-prompt-induced-waste-coding-agents].

### Strongest methods

- **Difficulty-gated architecture** [hong-2026-dats-difficulty-aware-topology], [fan-2025-imad]. These are evaluated under matched budgets and use interpretable difficulty or self-critique features. They work because the advantage of multi-agent collaboration grows with difficulty.
- **Heterogeneous planner/worker with measured token shares** [lin-2026-cursor-agent-swarm-economics], [cai-2026-think-big-search-small]. Savings come from the workers, who carry most tokens. Planner choice still affects how many tokens workers generate.
- **Harness-first optimization** [ali-2026-harness-effect], [yang-2026-harness-adaptation-slm], [katz-2026-cursor-token-efficiency]. This holds the model fixed and isolates the effect of the harness.
- **Crystallization into deterministic code** [malik-2026-progressive-crystallization]. It has eight months of production evidence and a demote-on-regression safeguard.

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| Evolved MAS team vs single agent, equal calls | Team 0.769 vs single agent 0.754 on ALFWorld (p=0.80) despite 1.8x more evaluation calls | ALFWorld | single frozen agent | [dylan-2026-equal-inference-cost-mas] |
| Orchestration (Self-Refine, BoN, Debate) | Largest gain 4.6 points over optimized CoT while needing roughly 2-4x the tokens of task-only inference | competitive programming, chess puzzles, mathematics | optimized CoT | [leins-2026-when-orchestration-pays-off] |
| DATS topology selection | At 40% of always-hierarchical cost, 77.7% pass@1 vs 73.6% always-hierarchical and 74.3% strongest learned competitor; hierarchical advantage grows from 2.4 to 21.1 points with difficulty at ~10x tokens | APPS, HumanEval+, LiveCodeBench, 400 math problems | always-hierarchical | [hong-2026-dats-difficulty-aware-topology] |
| Scaling agent systems (260 configs) | Relative change vs single agent ranges from +80.8% (decomposable financial reasoning) to -70.0% (sequential planning) | six agentic benchmarks | single agent | [kim-2025-science-of-scaling-agent-systems] |
| Claude Research (Opus lead + Sonnet subagents) | Opus 4 lead + Sonnet 4 subagents outperformed single-agent Opus 4 by 90.2% on an internal research eval; agents use ~4x chat tokens and multi-agent systems ~15x | internal research eval, BrowseComp | single-agent Opus 4 | [hadfield-2025-multi-agent-research-system] |
| Writer harness swap | Harness cuts blended cost per task 41% ($0.21->$0.12), wall-clock 44%, tokens 38% at quality parity (0.78->0.81) across six models on 22 tasks; quality per dollar +82% | 22 locked enterprise evaluation tasks | conventional production loop | [ali-2026-harness-effect] |
| Automated harness adaptation for SLMs | best small-model agent recovered 89.7% of LLM performance at 4% of the cost | seven business-oriented agentic tasks | LLM agent | [yang-2026-harness-adaptation-slm] |
| Progressive crystallization | Deterministic execution rose 0% to 45% over eight months; per-incident agent cost reduced >70% despite doubling incident volume (production cloud-networking AIOps) | production AIOps incidents | not stated | [malik-2026-progressive-crystallization] |
| Homogeneous debate vs self-correction | Debate uses 2.1-3.4x more tokens (up to 28,631 per problem) than isolated self-correction for equal or lower accuracy | GSM-Hard, MMLU-Hard | isolated self-correction | [bertalanic-2026-cost-of-consensus] |
| Orchestration patterns (financial docs) | Reflexive loop F1 0.943 at 2.3x sequential cost; hierarchical supervisor-worker F1 0.921 at 1.4x cost; hybrid configs recover 89% of reflexive gains at 1.15x cost | 10,000 SEC filings | sequential pipeline | [kulkarni-2026-financial-mas-orchestration-benchmark] |
| Single/multi-agent cascading | Hybrid cascading between MAS and single-agent improves accuracy 1.1-12% while reducing deployment cost up to 20% | popular agentic applications | not stated | [gao-2025-single-or-multi-agent-why-not-both] |
| Cloud vs on-prem coding agent | Prompt caching (99.3% hit rate) cut realized API cost 88.6% to $0.57 per million tokens; on-prem saves 40.1% TCO under shared GPUs but dedicated reservation costs 43.8% more; local config had fix-commit ratio 74.9% vs 45.9% | production monorepo, two 28-day periods | not stated | [peng-2026-inference-economics-coding-agents] |

### Critical notes

- **Pruning gains are probably overstated.** Under controlled, MAS-demanding evaluation, many reported pruning and topology gains turn out to come from method-specific prompts, structural collapse or disabled tools [zhang-2026-rethinking-mas-efficiency-evaluation]. This applies directly to [zhang-2024-agentprune], [zhang-2024-g-designer], [wang-2025-agentdropout] and [li-2025-adaptive-graph-pruning]. Their headline token cuts are measured against a dense MAS, not against a single agent.
- **Most positive MAS results are not compute-matched.** [hadfield-2025-multi-agent-research-system] states explicitly that its result is not a cost reduction. Compute-matched studies are mostly null or negative [dylan-2026-equal-inference-cost-mas], [tran-2026-single-vs-multi-agent-equal-budget], [leins-2026-when-orchestration-pays-off]. [tran-2026-single-vs-multi-agent-equal-budget] also flags API budget-control artefacts.
- **Small samples and high variance.** [dylan-2026-equal-inference-cost-mas] uses a frozen 7B model, and [peng-2026-inference-economics-coding-agents] is a single-developer case. Run-to-run variance is large and often unreported [hajimiri-2026-skill-memory-worth-tokens], [bai-2026-agents-spend-your-money].
- **Failure analysis.** MAS failures are dominated by design and verification issues, not model capacity [cemri-2025-mast-why-mas-fail]. Automated agent design rarely pays back its own cost [el-2025-inefficiencies-meta-agents].
- **Preprint share.** About 47% of C12 (74 of 158 records) are 2026 arXiv preprints. Production evidence comes almost entirely from vendor blogs [hadfield-2025-multi-agent-research-system], [katz-2026-cursor-token-efficiency], [phillips-2026-when-to-use-multi-agent-systems], plus one industrial preprint [malik-2026-progressive-crystallization].

### Open problems

- A predictive, task-level test for whether decomposition will pay, before building it. LLMs predict failure well (0.88 AUROC) but struggle to predict which protocol helps [yang-2026-protocol-routing-cost-aware].
- Compute-matched comparisons of *heterogeneous* MAS against a single frontier agent at reduced effort [xu-2026-oneflow-single-agent-baseline], [anthropic-2026-optimizing-cost-and-intelligence].
- When to crystallize or distill rather than route [malik-2026-progressive-crystallization], [dennis-2026-compiling-workflows-into-weights].

## C13. Benchmarks and evaluation for cost-quality tradeoffs

### Key ideas

1. **Offline router benchmarks built on precomputed outcomes.** Examples are RouterBench [hu-2024-routerbench], RouterEval [huang-2025-routereval], SPROUT with CARROT [somerstep-2025-carrot], EmbedLLM [zhuang-2024-embedllm], RouterArena [lu-2025-routerarena] and LLMRouterBench [li-2026-llmrouterbench]. A growing critique finds that simple or fixed baselines are hard to beat:
   - many routers, including commercial ones, do no better than simple baselines [li-2026-llmrouterbench]
   - well-tuned kNN is competitive with learned routers [li-2025-knn-routing]
   - gains often track how the tiers are composed, not the specific task [kumar-2026-task-session-routing-eval]
   - most of the routing gap is explained by task type [lee-2026-routing-gap-task-type]
   - failures shared by all models cap what any router can gain [chen-2026-co-failure-ceiling]
   - routers collapse toward the strongest model as budgets rise [lai-2026-routing-collapse]
2. **Evaluating step-level and in-agent switching.**
   - TwinRouterBench has a static track and a live SWE-bench track [yang-2026-twinrouterbench].
   - Interactive benchmarks with simulated users capture session length and cache expiry [notdiamond-2026-interactive-benchmarks-routing].
   - Live forked rollouts show that replaying logged trajectories mispredicts what a model swap does [gonuguntla-2026-replay-gap].
   - Handing a conversation to a different model mid-way causes drift that depends on the model pair [khraishi-2026-model-switching-drift].
   - Other benchmarks cover delegation headroom [gao-2026-decisionbench], which steps need which model tier [karmakar-2026-agentfloor] and routing per tool call [agarwal-2026-switchcraft].
3. **Cost-aware metrics.** These include the cost-accuracy Pareto frontier [kapoor-2024-ai-agents-that-matter], cost-of-pass [erol-2025-cost-of-pass; wang-2025-efficient-agents], CLEAR [mehta-2025-clear], dollar-denominated economic evaluation [zellinger-2025-economic-evaluation], the cheapest oversight policy that meets a reliability target [chatrath-2026-ready] and cost per successful task [weinberger-2026-prompt-induced-waste-coding-agents].
4. **When MAS pays off.** Coordination helps or hurts depending on the task [kim-2025-science-of-scaling-agent-systems]. Under equal thinking-token budgets, single agents match or beat MAS [tran-2026-single-vs-multi-agent-equal-budget]. Orchestration gains are modest and cost 2-4x the tokens [leins-2026-when-orchestration-pays-off]. Many MAS efficiency gains vanish under controlled evaluation [zhang-2026-rethinking-mas-efficiency-evaluation]. Heterogeneous model assignment is a reliable improvement [ye-2025-x-mas; zhang-2025-stop-overvaluing-mad; jiang-2026-agentcard], but larger model pools often hurt [marjanovic-2026-model-pools-mas].
5. **Reliability and enterprise realism.** τ-bench introduced pass^k [yao-2024-tau-bench], and reliability profiles go further [rabanser-2026-agent-reliability]. Enterprise suites include WorkArena and WorkArena++ [drouin-2024-workarena; boisvert-2024-workarena-plus-plus], ITBench [jha-2025-itbench], CRMArena-Pro [huang-2025-crmarena-pro] and EnterpriseClawBench [zhong-2026-enterpriseclawbench].
6. **Reasoning effort.** Reasoning models overthink in agent loops [cuadron-2025-danger-of-overthinking-agents]. Routine office tasks need only low effort [xu-2026-low-effort-office-agents]. Effort defaults work like contract terms that differ per model [moon-2026-price-of-thinking].

### Strongest methods

- **HAL.** It is a standardized harness with cost accounting and Pareto analysis, and it releases 2.5B tokens of logs [kapoor-2025-hal-holistic-agent-leaderboard].
- **LLMRouterBench and the four-router common-interface evaluation.** These are the most rigorous controls for router claims. Any step router should be compared against fixed tiers matched on traffic share [li-2026-llmrouterbench; kumar-2026-task-session-routing-eval].
- **The Replay Gap.** It is the key methodological result for step routing: offline replay of model swaps is invalid [gonuguntla-2026-replay-gap].
- **TwinRouterBench.** Its tier labels are verified by execution, and its dynamic track avoids online judges [yang-2026-twinrouterbench].
- **Cost-of-pass, plus pass^k for consistency.** Together they are the most defensible metrics for agents [erol-2025-cost-of-pass; yao-2024-tau-bench].
- **AgentCARD.** It is the only benchmark here that varies the model per role and the hosting mode [jiang-2026-agentcard].
- **Towards a Science of Scaling Agent Systems.** Its 260 controlled configurations give a predictive model of when coordination helps [kim-2025-science-of-scaling-agent-systems].

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| Switchcraft per-tool-call router | 82.9% accuracy, matching/exceeding best single model, with 84% lower inference cost (over $3,600 saved per million queries) | five function-calling benchmarks (incl. BFCL) | best single model | [agarwal-2026-switchcraft] |
| Harness-native step router | LightGBM router preserves task quality while cutting realized cost ~90% on PinchBench and ~43% on DRACO vs fixed strong model | DRACO, PinchBench | fixed strong model | [liu-2026-harness-native-routing] |
| Efficient Agents | Efficient Agents retains 96.7% of OWL's performance while reducing cost from $0.398 to $0.228, a 28.4% improvement in cost-of-pass on GAIA | GAIA | OWL | [wang-2025-efficient-agents] |
| Orchestration vs single call | Largest gain 4.6 points over optimized CoT while needing roughly 2-4x the tokens of task-only inference | competitive programming, chess puzzles, mathematics | optimized CoT | [leins-2026-when-orchestration-pays-off] |
| Agent-system scaling | Relative change vs single agent ranges from +80.8% (decomposable financial reasoning) to -70.0% (sequential planning) | six agentic benchmarks | single agent | [kim-2025-science-of-scaling-agent-systems] |
| CLEAR | Accuracy-only optimization yields agents 4.4-10.8x more expensive than cost-aware alternatives with comparable performance (300 enterprise tasks) | 300 enterprise tasks | cost-aware alternatives | [mehta-2025-clear] |
| Overthinking selection | Selecting the solution with lower overthinking score improves performance by almost 30% while reducing computational costs by 43% (SWE-Bench Verified) | SWE-Bench Verified (4018 trajectories) | not stated | [cuadron-2025-danger-of-overthinking-agents] |
| Test-gated cascade (vendor-reported) | Cascade (DeepSeek V4 Pro first, escalate to GPT-5.6 Sol on test failure): 83.0% of DeepSWE tasks at $3.35/task vs Sol alone 72.7% at $8.37 (10 pts better, 60% cheaper); 904 rollouts (vendor-reported) | DeepSWE | Sol alone | [together-2026-deepswe-cascade-deepseek-gpt] |
| Web-agent observation-mode routing | Sending only tasks no mode solves to the cheapest mode cuts cost 9.5-30.6% at unchanged success; no tested routing policy robustly beats one well-chosen fixed mode | VisualWebArena, WebArena | one well-chosen fixed mode | [wei-2026-routing-least-learnable] |
| Learned early stopping | the same policy saves 32% tokens with KV-cache forking but costs 121% extra under black-box repeated prefilling | GSM8K, MATH-500, MMLU-Pro, AIME-90, GPQA | not stated | [dong-2026-learning-to-stop-cost-study] |
| HAL leaderboard | Reports ~$40,000 total cost for 21,730 rollouts across 9 models x 9 benchmarks; finds higher reasoning effort reduced accuracy in most runs | 9 benchmarks incl. coding, web navigation, science, customer service (e.g., tau-bench, GAIA, SWE-bench) | not stated | [kapoor-2025-hal-holistic-agent-leaderboard] |
| Leader-model orchestration | API cost spans over 100x across leader models while overall score spans under 4x | ClawArena-Team (41 scenarios, 258 rounds) | not stated | [xiong-2026-clawarena-team] |
| Low reasoning effort | Low reasoning level used about 43% fewer output tokens and 20% less time than max level, with no rule violations across 840 runs of 14 office tasks | 14 routine office tasks (840 runs) | max level | [xu-2026-low-effort-office-agents] |
| Delegation headroom | Counterfactual perfect delegation is 15-31 points above measured performance | GAIA, tau-bench, BFCL multi-turn | measured performance | [gao-2026-decisionbench] |

### Critical notes

- **Oracle headroom is inflated by noise.**
  - 5.37% of model-question pairs flip between identical runs [lee-2026-routing-gap-task-type].
  - Run-to-run noise of 12-14% inflates oracle gains for web agents [wei-2026-routing-least-learnable].
  - Stochastic decoding overstates the ceiling for routers that commit to one model [chen-2026-routing-oracle-stochastic].
  - Even a vendor states that oracle upside is biased upward [fireworks-2026-frontier-is-a-router].
- **Static evaluation scores the wrong world for model switching.** Swaps rewrite 61-94% of subsequent actions [gonuguntla-2026-replay-gap]. A single handoff moves strict success by -8 to +13 pp [khraishi-2026-model-switching-drift].
- **Static benchmarks miss cache expiry,** which raises input cost 10-20x [notdiamond-2026-interactive-benchmarks-routing]. Savings from early exit flip sign depending on the serving cost model [dong-2026-learning-to-stop-cost-study].
- **MAS gains are often unaccounted compute.** They shrink under equal token budgets [tran-2026-single-vs-multi-agent-equal-budget; wang-2024-reasoning-token-economies] and can come from structural collapse or disabled tools [zhang-2026-rethinking-mas-efficiency-evaluation].
- **Most enterprise benchmarks do not report cost.** WorkArena, ITBench and CRMArena-Pro list no cost figures in these digests [drouin-2024-workarena; jha-2025-itbench; huang-2025-crmarena-pro]. EnterpriseClawBench explicitly asks for cost and runtime to be reported [zhong-2026-enterpriseclawbench].
- **Judges and benchmarks have known flaws.** LLM judges for deep-research agents are unreliable [wang-2026-reflect-judges-research-agents]. Benchmark flaws can misestimate performance by up to 100% relative [zhu-2025-agentic-benchmark-checklist]. Prompt optimization is often a coin flip [zhang-2026-po-coin-flip].
- **Several cascade results are vendor-run.** They should be read as single-source [together-2026-deepswe-cascade-deepseek-gpt; together-2026-deepswe-cascade-glm-flash].

### Open problems

- No public benchmark puts step-level routing, prompt-cache pricing and heterogeneous multi-agent roles together. AgentCARD has roles without cache dynamics [jiang-2026-agentcard]. The interactive benchmarks have cache dynamics but cover single-agent coding [notdiamond-2026-interactive-benchmarks-routing].
- Off-policy evaluation for agent model switching is still unvalidated against live forks [liu-2026-adwm-ope-llm-agents; gonuguntla-2026-replay-gap].
- Enterprise benchmarks need cost-of-pass and pass^k reporting [erol-2025-cost-of-pass; yao-2024-tau-bench].
- Routing supervision is scarcest exactly where routing matters most [wei-2026-routing-least-learnable].
- Statistical certification of router gain before shipping is only starting [sun-2026-routeguard; khosravi-2026-which-llm-which-work].

## C14. Industry products, gateways and production case studies

### Key ideas

1. **Production turn- and step-level routers in coding agents.**
   - Fireworks routes per turn and prices the cache loss from switching [fireworks-2026-firerouter-with-opus].
   - Cursor learns its router from implicit feedback in production logs [okeefe-2026-cursor-router].
   - GitHub Copilot switches models only at cache boundaries [github-copilot-auto-model-selection].
   - Not Diamond Code predicts reward and cost per step [notdiamond-2026-interactive-benchmarks-routing].
   - Replit lets the frontier agent assign tiers to its own subagents [furman-2026-replit-free-the-models].
   - Amp swaps the model behind each role independently [amp-2026-subagent-model-assignment].
2. **Executor-advisor pairing and effort levers.** A cheap executor consults an expensive advisor at hard points [anthropic-2026-advisor-strategy]. Vendor guidance covers effort, task budgets and delegation [anthropic-2026-optimizing-cost-and-intelligence; anthropic-task-budgets-docs].
3. **Managed routers and gateways.** These include Bedrock routing [feng-2025-ipr; aws-2025-bedrock-intelligent-prompt-routing], the Microsoft Foundry router with session affinity [microsoft-2025-foundry-model-router], OpenRouter's router driven by market spend [openrouter-2025-auto-router], LiteLLM with session pinning [litellm-2026-auto-routing], and gateways [portkey-ai-gateway; cloudflare-ai-gateway; wang-2024-uber-genai-gateway].
4. **Replacing a pipeline step with a smaller model trained on production traces.**
   - escalation classifier [yarullin-2025-intercom-escalation]
   - fine-tuned small models [joglekar-2025-intercom-small-llms; gola-2026-langsmith-fine-tuning; ayala-2025-slm-vs-llm-low-code-workflows]
   - progressive delegation from LLM to SLM [selvaraj-2026-adept]
   - crystallizing repeated behavior into deterministic workflows [malik-2026-progressive-crystallization]
   - covering the whole request mix with one self-hosted model [tsymboi-2026-production-traffic-post-training]
   - prompt re-optimization for cheaper models [wang-2026-dropbox-dash-dspy; singhvi-2025-databricks-gepa-90x]
5. **Closed-loop production cascades.** Examples are [guo-2026-routenlp; lyu-2026-differential-reasoning-router; liskowski-2026-compositional-online-learning; rokon-2026-autorelannotator].
6. **Guidance on what multi-agent setups cost.** MAS spend 3-10x more tokens [phillips-2026-when-to-use-multi-agent-systems]. A heterogeneous research system uses about 15x chat tokens [hadfield-2025-multi-agent-research-system]. Practitioners advise starting simple [anthropic-2024-building-effective-agents; openai-2025-practical-guide-building-agents; yan-2025-dont-build-multi-agents].

### Strongest methods (most credible production evidence)

- **FireRouter.** Its A/B test randomized sessions for more than a month, reports an uncertainty band (57% +/- 19 ppts) and reports the drop in cache hit rate. It is still vendor-reported [fireworks-2026-firerouter-with-opus].
- **Cursor Router.** It trains on hundreds of thousands of production turns, validates on held-out logs and then on live traffic, and only upgrades to a frontier model at 75% one-sided confidence. It is vendor-reported, and no confidence intervals are given in the digest [okeefe-2026-cursor-router].
- **Intercom escalation classifier.** A production A/B test reports modest, plausible gains: cost per resolution down about 3% [yarullin-2025-intercom-escalation].
- **Snowflake Cortex AISQL.** This production case study breaks down per-row cost by component, and it reports the gap between the upper bound and realized savings (11.4x to about 8x) [liskowski-2026-compositional-online-learning].
- **Long deployments.** An eight-month AIOps deployment [malik-2026-progressive-crystallization] and an 8-week enterprise pilot [guo-2026-routenlp] are among the longest production records here.
- **Inference economics from telemetry.** Peng uses Git history and telemetry, but from a single developer [peng-2026-inference-economics-coding-agents].
- **Enterprise agent routing on real traffic.** Routing to specialized agents degrades as the catalog grows, measured on 1,435 human-labeled production utterances [gillespie-2026-enterprise-agent-routing-scaling].

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| FireRouter per-turn cache-aware router (vendor) | Internal A/B test (>1 month, sessions randomised to router vs Opus-only): cost per session $15.36 -> $6.63 (57% +/- 19 ppts lower); graded-turn accuracy 78.7% vs 80.2% (98.1% of Opus-only); cache hit rate 94.2% vs 97.8% (vendor-reported) | internal coding traffic A/B test | Opus-only | [fireworks-2026-firerouter-with-opus] |
| Cursor Router (vendor) | Auto Intelligence: above Fable-level user satisfaction at 68% lower cost; Auto Balance: outperforms Opus 4.8 at 41% lower cost with +3% satisfaction (relative to frontier models, measured on live traffic; vendor-reported) | live Cursor traffic, held-out offline split of production turns | frontier models | [okeefe-2026-cursor-router] |
| Advisor strategy (vendor) | Sonnet executor + Opus advisor: +2.7 points on SWE-bench Multilingual over Sonnet alone with 11.9% lower cost per agentic task | SWE-bench Multilingual | Sonnet alone | [anthropic-2026-advisor-strategy] |
| Orchestrator-worker vs solo (vendor) | on full BrowseComp, solo Fable 5 at lower effort was 22-30% cheaper than the orchestrator at same accuracy | BrowseComp | orchestrator (Fable 5 coordinator + Sonnet 5 worker) | [anthropic-2026-optimizing-cost-and-intelligence] |
| Replit self-delegation (vendor) | DeepSWE v1.1: Replit Agent 72% at $2.11/task vs GPT-6 Astra (mini-swe-agent) 67% at $1.60 (low effort) and 74% at $4.43 (xhigh); sidekick architecture 61% at $1.34 | DeepSWE v1.1 | GPT-6 Astra (mini-swe-agent) | [furman-2026-replit-free-the-models] |
| Escalation step distilled to ModernBERT (vendor) | Company-reported: custom ModernBERT handles 90% of cases with LLM fallback for 10%; >98% escalation accuracy; latency -0.5s; cost per resolution ~-3% | production A/B test | LLM escalation decision | [yarullin-2025-intercom-escalation] |
| RouteNLP closed loop | 8-week enterprise pilot (~5K queries/day): 58% lower inference cost at 91% response acceptance, p99 latency 1,847 ms to 387 ms | six-task finance/customer-service/legal benchmark | not stated | [guo-2026-routenlp] |
| Progressive crystallization | Deterministic execution rose 0% to 45% over eight months; per-incident agent cost reduced >70% despite doubling incident volume (production cloud-networking AIOps) | not stated | not stated | [malik-2026-progressive-crystallization] |
| IPR (Bedrock) (vendor-authored) | Deployed on a major cloud platform: 43.9% cost reduction at quality parity with the strongest Claude-family model, sub-150ms latency (abstract) | IPRBench (1.5M prompts, 11 LLMs) | strongest Claude-family model | [feng-2025-ipr] |
| RLM-Cascade response-level draft/verify | On a Claude Code agentic workload: 88.8% draft-use over 125 production requests, 45.8% lower API cost vs direct Opus, p50 latency 2,026 ms vs 3,698 ms | 125 production Claude Code requests | direct Opus | [wu-2026-rlm-cascade] |
| Harness context changes (vendor) | Harness changes cut user token costs 7% with no quality loss | Cursor production traffic | not stated | [katz-2026-cursor-token-efficiency] |
| Router cost-tradeoff knob (vendor) | Enabling the cost-tradeoff parameter on Not Diamond's consumer chatbot cut inference cost 51% on average (~$750,000 annualised) | live chatbot traffic before/after | quality-maximising setting | [notdiamond-2024-reduce-own-inference-costs] |
| Cloud API vs on-prem | on-prem saves 40.1% TCO under shared GPUs but dedicated reservation costs 43.8% more; local config had fix-commit ratio 74.9% vs 45.9% | production monorepo, two 28-day periods | API Claude Opus in Claude Code | [peng-2026-inference-economics-coding-agents] |
| Customer-specific routers (vendor) | Customer help chat: 52.4% lower error rate and 92% lower cost | not stated | not stated | [wood-2025-martian-up-and-to-the-left] |

### Critical notes

- **Some vendor claims come with no methodology.**
  - Martian's customer results are explicitly reported with "no methodology" [wood-2025-martian-up-and-to-the-left].
  - vLLM Semantic Router's trial gains have "no benchmark or baseline disclosed" [vllm-2025-semantic-router].
  - Not Diamond's product claims are headline numbers only [notdiamond-product].
  - Klarna's figures are business-impact estimates [klarna-2024-ai-assistant].
- **Before/after comparisons are not A/B tests.** Not Diamond's 51% saving compares live traffic before and after the change, not randomized groups. Week-over-week growth also changed at the same time [notdiamond-2024-reduce-own-inference-costs].
- **Production quality proxies are implicit and can be gamed.** Cursor labels quality by whether the user moves on or corrects the agent [okeefe-2026-cursor-router], and Fireworks scores whether the user corrected the next turn [fireworks-2026-firerouter-with-opus]. Signals from the next user turn are open to proxy hacking [pang-2023-implicit-feedback-deployment-dialogue]. Sparse thumbs-up/down feedback is selection-biased [morandi-2026-selection-bias-sparse-feedback].
- **Routing reduces cache hits.** The cache hit rate fell under routing [fireworks-2026-firerouter-with-opus]. Copilot and Foundry limit switching to cache boundaries or session affinity [github-copilot-auto-model-selection; microsoft-2025-foundry-model-router].
- **Delegating to cheaper workers is not always a win.**
  - Solo models at lower effort beat orchestrator-worker setups on full BrowseComp [anthropic-2026-optimizing-cost-and-intelligence].
  - Advisor gains shrink as the executor approaches the advisor's capability [anthropic-2026-optimizing-cost-and-intelligence].
  - Replit argues that an external router is less capable than the model it routes for [furman-2026-replit-free-the-models].
- **Managed routers have narrow scope.** Bedrock routing is pairwise, works only within one model family, and cannot adapt to application data [aws-2025-bedrock-intelligent-prompt-routing]. The Foundry page gives no quantitative savings [microsoft-2025-foundry-model-router].
- **Honest production gains are small.** Examples are about 3% [yarullin-2025-intercom-escalation], 5% and 12.5% [joglekar-2025-intercom-small-llms] and 7% [katz-2026-cursor-token-efficiency]. The 50-90% headlines come from vendors or from small samples [wu-2026-rlm-cascade].
- **Hosted open-weight endpoints drift.** Their quality is not predicted by price [he-2026-market-aware-routing; li-2026-same-model-not-same-service].

### Open problems

- All randomized production A/B evidence for step-level routing comes from coding agents [fireworks-2026-firerouter-with-opus; okeefe-2026-cursor-router]. None comes from enterprise multi-agent workflows such as ITSM or CRM.
- No production report gives per-role cost attribution together with quality for heterogeneous subagents. Amp's figures come from small internal evals [amp-2026-subagent-model-assignment].
- Prompts need re-optimizing whenever a step is downgraded to a cheaper model [wang-2026-dropbox-dash-dspy; notdiamond-2025-prompt-adaptation-weaker-models]. This cost is rarely included in claimed savings.
- Certifying a budgeted assignment of models to workloads under uncertain evaluation is new and has not been replicated [khosravi-2026-which-llm-which-work].

## C15. Surveys and theory

### Key ideas

- **Optimality theory.** Cascade routing unifies routing and cascading with optimality proofs [dekoninck-2024-cascade-routing]. Separate cost and accuracy estimators are minimax rate-optimal [somerstep-2025-carrot]. Duality gives the cascade frontier [bouchard-2026-escalation-decision-theory]. Earlier work asks when confidence-based deferral is enough [jitkrittum-2023-confidence-cascade-deferral]. Distribution-free conditional guarantees have limits [zeng-2025-pac-efficient-routing].
- **Ceilings and value of information.** A co-failure ceiling bounds any single-answer policy [chen-2026-co-failure-ceiling]. Pandora's Box search decides when a better routing estimate is worth paying for [fisch-2026-pandoras-router], [belloni-2026-online-pandoras-box-cascading].
- **Online and budgeted guarantees.** These include SLA virtual queues [woisetschlager-2025-mess-plus], competitive ratios [wu-2025-training-free-online-routing], surrogate-reward bandits [sridhar-2026-correlated-bandits-routing], offline RL with a global budget [xu-2026-seqroute], and batch Lagrangian allocation [mei-2025-omnirouter].
- **Economic metrics.** Cost-of-pass [erol-2025-cost-of-pass], [wang-2025-efficient-agents] and dollar-denominated evaluation [zellinger-2025-economic-evaluation].
- **Causal framing of logs.** Routing from logs is a causal problem [frauen-2026-causal-methods-llm]. Prediction-powered inference corrects proxy labels with a small labelled set [angelopoulos-2023-ppi], [angelopoulos-2023-ppi-plus-plus].
- **Position papers and surveys.** SLM-default agents [belcak-2025-slms-future-agentic-ai], [sharma-2025-slm-agentic-survey]; routing and cascading taxonomies [moslem-2026-routing-cascading-survey], [varangot-reille-2025-doing-more-with-less]; test-time compute as an alternative to a larger model [snell-2024-test-time-compute-optimal], [brown-2024-large-language-monkeys].

### Strongest methods

The most useful theory for practitioners is [dekoninck-2024-cascade-routing], which shows that quality-estimator accuracy is the bottleneck, [chen-2026-co-failure-ceiling], which gives a certificate on achievable gain, and [somerstep-2025-carrot], which justifies simple estimators. [moslem-2026-routing-cascading-survey] is the most current taxonomy.

### Reported cost/quality tradeoffs

| Method | Result as reported | Benchmark | Baseline | Record |
|---|---|---|---|---|
| Efficient Agents | retains 96.7% of OWL's performance while reducing cost from $0.398 to $0.228, a 28.4% improvement in cost-of-pass | GAIA | OWL | [wang-2025-efficient-agents] |
| Cost-of-pass frontier | Frontier cost-of-pass for complex quantitative tasks roughly halved every few months over the year studied | complex quantitative task suites | not stated | [erol-2025-cost-of-pass] |
| Compute-optimal TTC | improves test-time compute efficiency by more than 4x vs best-of-N | not stated | best-of-N | [snell-2024-test-time-compute-optimal] |
| Repeated sampling | DeepSeek-V2-Coder solves 56% with 250 samples vs 15.9% with one | SWE-bench Lite | single sample | [brown-2024-large-language-monkeys] |
| Escalation decision theory | A lightweight pre-generation router beats the best cascade policy on 4 of 5 datasets | MATH, MMLU, TriviaQA, SimpleQA, LiveCodeBench | best cascade policy | [bouchard-2026-escalation-decision-theory] |
| SeqRoute | Reduces operational costs by 6.0-73.5% while maintaining or improving quality; bankruptcy under 1% | 10,000 raw sessions expanded to 2.38M transitions | not stated | [xu-2026-seqroute] |
| MESS+ | Average 2x cost savings vs existing LLM routing techniques | LLM benchmarks | existing LLM routing techniques | [woisetschlager-2025-mess-plus] |
| Training-free online routing | Average 3.55x improvement in overall performance, 1.85x in cost efficiency and ~4.25x throughput vs 8 baselines | 3 benchmark datasets | 8 baselines | [wu-2025-training-free-online-routing] |
| OmniRouter | Up to 6.30% accuracy improvement while reducing computational costs by at least 10.15% vs router baselines | not stated | router baselines | [mei-2025-omnirouter] |

### Critical notes

- The SLM survey's 10x-100x claim is synthesised from the literature, not measured [sharma-2025-slm-agentic-survey]. [belcak-2025-slms-future-agentic-ai] is a position paper.
- Regret and competitive-ratio guarantees assume i.i.d. or bounded-drift streams and reliable rewards. Uncertainty quantification for agents needs reassessment [kirchhof-2025-uq-agents-position].
- Many 2026 theory papers are unreviewed preprints [chen-2026-co-failure-ceiling], [fisch-2026-pandoras-router], [bouchard-2026-escalation-decision-theory].

### Open problems

- Theory for placing strong models within agent swarms [itkin-2026-budgeted-strong-correctors] and for when multi-agent systems help [yu-2026-when-mas-help-information-bottleneck], [fan-2026-science-of-collective-ai].
- Using orchestration traces as the learning unit [zhang-2026-rl-orchestration-traces] and tracking cost against utility along agent trajectories [liu-2026-cost-utility-alignment-trajectories].

# Part 2: Approaches applicable to step-level routing in multi-agent pipelines

This part merges the step-level notes from every cluster. The target setting has one frontier model at every subagent and step, and the goal is to move steps to cheaper models, or cheaper effort settings, without lowering task success.

## 2.1 Choose the decision points before the router

Routing can happen at four places:
- **at session or task admission** [raj-2026-trace-router] [kwartler-2026-harness-tokenomics];
- **at subagent launch or plan time** [singh-2026-planner-as-router] [furman-2026-replit-free-the-models];
- **per turn** [okeefe-2026-cursor-router] [fireworks-2026-firerouter-with-opus];
- **per step or tool call** [paul-2026-agentrouter-step-tiers] [agarwal-2026-switchcraft].

The cheapest safe points are those where context is fresh anyway: subagent launch, side lanes and new sessions. A switch in the middle of a trajectory costs a cache rebuild and a handoff tax [kwartler-2026-harness-tokenomics] [ganz-2026-handoff-tax].

Pinning one model per task can beat per-call routing when rewards are delayed, because it avoids credit-assignment problems [raj-2026-trace-router]. Downshifting after the hard reasoning is done is more favourable than escalating midway [ganz-2026-handoff-tax].

**Default design:** assign a tier per role or subagent at launch. Allow switches mid-trajectory only at safe boundaries:
- outside tool loops [liu-2026-vllm-sr-session-aware-agentic-routing];
- at cache boundaries [github-copilot-auto-model-selection].

## 2.2 Order the levers by friction

**1. Change effort before changing model.** Per-step reasoning effort keeps the model and its prompt cache. It cut reasoning tokens by up to 52.7% against fixed high effort on agent benchmarks [yang-2026-ares-reasoning-effort]. Related approaches:
- turn-adaptive thinking budgets under a global cap [jali-2026-turn-adaptive-budgets];
- vendor documentation of effort, caching and delegation trade-offs [anthropic-2026-optimizing-cost-and-intelligence].

The same vendor documentation records a negative result to keep in mind: on the full task set, delegating to cheaper workers lost to a lower-effort solo model [anthropic-2026-optimizing-cost-and-intelligence]. A compute-matched single agent is often competitive with a multi-agent system [tran-2026-single-vs-multi-agent-equal-budget] [xu-2026-oneflow-single-agent-baseline].

**2. Assign models per role.**
- Measure each role's share of tokens first. Workers carried at least 69% of tokens in a vendor agent-swarm study, so downsizing executors gives the most saving [lin-2026-cursor-agent-swarm-economics].
- Keep capacity on the decomposer or delegator. Scaling it improved exact match about four times more than scaling the executor [cai-2026-think-big-search-small].
- Search the role-to-model map offline:
  - bandit-style search over role assignments [hua-2026-agentopt];
  - quality-constrained per-module selection [huang-2026-scope];
  - per-module model choice as a decision variable [chen-2025-llmselector].
- The value of an upgrade depends on the role, so generic tier labels do not transfer between roles [wang-2026-beyond-tier-labels].
- Routing over (model, role) or (model, collaboration mode) pairs generalises this to multi-agent systems [yue-2025-masrouter] [ye-2025-x-mas].

**3. Add calibrated escalation only where a cheap signal exists.**
- Make the cheap model the default executor. Escalate on calibrated risk, stalls or verifier failure [hemadri-2026-r2v-agent] [piatrashyn-2026-redact] [wei-2026-step-level-cua-cascade] [farzaneh-2026-tsds-edge-deferral].
- Where tests exist, verifier-gated cascades transfer directly [together-2026-deepswe-cascade-deepseek-gpt].
- Hand off only the remaining part of the task, under a residual contract, instead of restarting [deng-2026-cfrc-residual-completion-handoffs].

**4. Train small step routers.**
- A small classifier per step can beat RouteLLM and FrugalGPT applied per step. It reported 72% lower cost at 97.3% of frontier quality [paul-2026-agentrouter-step-tiers].
- Per-turn routers learned from logged trajectories are a second option [zhang-2026-mtrouter]. Per-tool-call routing is a third [agarwal-2026-switchcraft].
- An RL-trained small orchestrator is a heavier alternative that treats models and tools as callable actions [su-2025-toolorchestra].

## 2.3 Signals available at each point

| When | Signal | Records |
|---|---|---|
| Before a step | Subtask text and embeddings, task-taxonomy labels, plan structure, difficulty features | [agarwal-2026-switchcraft] [okeefe-2026-cursor-router] [singh-2026-planner-as-router] [hong-2026-dats-difficulty-aware-topology] |
| During a step | Harness and conversation state; small-model uncertainty or logits; cheap-model hidden-state probes; draft-model scoring of a black-box agent | [liu-2026-harness-native-routing] [lee-2025-steer] [piatrashyn-2026-redact] [bhola-2026-scrouting-superscout] [grotov-2026-speculative-uncertainty] |
| During a step | Process verifiers or step PRMs as escalation triggers; progress or stall monitors | [hemadri-2026-r2v-agent] [kapoor-2026-trim-stepwise-routing] [chae-2025-web-shepherd] [wei-2026-step-level-cua-cascade] |
| After a step | Tests, compile and lint results | [together-2026-deepswe-cascade-deepseek-gpt] [abdollahi-2026-hdlforge] |
| System | Cache warmth and provider-state portability; serving load | [liu-2026-vllm-sr-session-aware-agentic-routing] [dai-2025-aragog] [kabir-2026-inframind] |

Cheap verifiers carry a monitoring risk. A cascade judged by its own cheap verifier can look healthy while it degrades [rajput-2026-cheap-verifiers-blind-spots]. Keep a gold-labelled audit stream.

## 2.4 How to label and train

- **Implicit user feedback on production turns.** Labels come from whether the user moved on or corrected the agent. A frontier model is admitted for a task label only when confidence passes a gate [okeefe-2026-cursor-router] (vendor-reported).
- **Execution-verified tier labels** [yang-2026-twinrouterbench]. **Minimum-sufficient-effort labels** [yang-2026-ares-reasoning-effort].
- **Distil first, then route.** Train the router on the cheap model's residual failures [hemadri-2026-r2v-agent], and keep improving the cheap model behind it [yao-2026-mera].
- **When to hand off.** Learn handoff timing from censored teacher annotations [lei-2026-tacit-switch]. For session-level budgets, use offline RL with hindsight budget relabeling [xu-2026-seqroute].

## 2.5 Cache and switching costs

- Put the expected cache loss into the routing objective [fireworks-2026-firerouter-with-opus] [liu-2026-vllm-sr-session-aware-agentic-routing] [notdiamond-2026-interactive-benchmarks-routing].
- On long, tool-heavy sessions, the most expensive model can cost less than the next tier down, because cache hits dominate the bill [kwartler-2026-harness-tokenomics].
- Keep the main model's prefix stable:
  - offload volatile side work such as tool search to a sub-model [zha-2026-cacherouter];
  - keep context append-only [ji-2025-manus-context-engineering].
- Measure billed cost, not tokens. Compressing tool output cut tokens by 38.4% but raised billed cost by 6.8% [weinberger-2026-token-reduction-not-cost].
- In self-hosted serving, mixing models loses single-model KV reuse [xu-2026-oneflow-single-agent-baseline]. A shared prefill module across specialized models is one remedy [woo-2026-prefillshare]. Multi-LoRA KV sharing is another, when the cheap tiers are adapters of one base model [jeon-2026-lragent].

## 2.6 How to evaluate a step router

1. **Replay is not enough.** Replaying logged trajectories mispredicts the outcome of a mid-trajectory swap [gonuguntla-2026-replay-gap]. Use instead:
   - forked live rollouts;
   - dynamic harness tracks [yang-2026-twinrouterbench];
   - interactive simulated-user sessions that capture cache expiry [notdiamond-2026-interactive-benchmarks-routing].
2. **Compare against a share-matched fixed-tier control.** Many routers emit near-constant tiers [kumar-2026-task-session-routing-eval].
3. **Include a budget-matched single-frontier baseline at tuned effort** [anthropic-2026-optimizing-cost-and-intelligence].
4. **Report cost per solved task with run-to-run variance** [weinberger-2026-token-reduction-not-cost] [wei-2026-routing-least-learnable]. For reliability, use pass^k [yao-2024-tau-bench].
5. **Finish with a randomized session-level A/B** that scores per-session cost and next-turn correction [fireworks-2026-firerouter-with-opus].

## 2.7 Suggested sequence for the target pipeline

1. Baseline the current frontier-everywhere pipeline on cost per solved task. Tune caching and effort first [anthropic-2026-optimizing-cost-and-intelligence] [lumer-2026-dont-break-the-cache] [yang-2026-ares-reasoning-effort].
2. Measure each role's token share. Downsize high-volume executor and worker roles first, and keep the planner strong [lin-2026-cursor-agent-swarm-economics] [cai-2026-think-big-search-small].
3. Search role-to-model maps offline on re-run samples [hua-2026-agentopt] [huang-2026-scope].
4. Add calibrated escalation at steps that have a verifier or a reliable risk signal [hemadri-2026-r2v-agent] [farzaneh-2026-tsds-edge-deferral].
5. Make the routers cache-aware, and restrict switches to safe boundaries [liu-2026-vllm-sr-session-aware-agentic-routing] [kwartler-2026-harness-tokenomics].
6. Validate with forked rollouts, then shadow mode, then a randomized A/B, before trusting any offline estimate of savings [gonuguntla-2026-replay-gap] [fireworks-2026-firerouter-with-opus].

# Part 3: Approaches for learning routing/config policies from offline logs vs. online traffic

## 3.1 What the logs actually contain

Each step of a frontier-everywhere system logs one model's output and its cost. Nothing is logged about what any other model would have done.

Quality signals in those logs come in three kinds:
- **Proxy or judge scores** [khosravi-2026-which-llm-which-work] [zhang-2025-meta-router].
- **Implicit behaviour.** The user moves on or corrects the agent [okeefe-2026-cursor-router], retries [bae-2026-contextual-queueing-bandits], switches model [openai-2025-gpt5-system-card], or rejects the output [unell-2026-deployment-centered-rejection-risk].
- **Sparse explicit feedback** such as thumbs or complaints [woisetschlager-2026-slarouter].

This has three consequences:
1. **Single-model logs hold no counterfactual outcomes.** Some re-running or exploration is required before any router can be trained.
2. **Routed logs are biased.** Once a router is deployed, logged outcomes are confounded by its choices. Supervised training or naive replay on them is biased [tsiourvas-2025-causal-routing] [frauen-2026-causal-methods-llm].
3. **Off-policy evaluation needs logged propensities and overlap** [li-2026-counterfactual-tool-ranking]. Implicit signals carry selection bias and can be gamed [morandi-2026-selection-bias-sparse-feedback] [pang-2023-implicit-feedback-deployment-dialogue] [liu-2025-user-feedback-noisy-signal].

## 3.2 Learning from logs (offline)

| Family | What it needs | Records |
|---|---|---|
| Supervised routers on full-information data | Preference pairs or per-model correctness matrices, built by re-running a sample of logged inputs on candidate models | [ong-2024-routellm] [zhuang-2024-embedllm] [song-2025-irt-router] [somerstep-2025-carrot]; kNN is a strong baseline [li-2025-knn-routing] |
| Step-level supervised routers | Annotated steps or logged trajectories | [paul-2026-agentrouter-step-tiers] [zhang-2026-mtrouter] |
| Counterfactual / causal routing | Single-outcome logs with some variation across models; or judge labels debiased with a small gold set | [tsiourvas-2025-causal-routing] [zhang-2025-meta-router] |
| Off-policy evaluation | Logged propensities; doubly robust, model-based or optimal-transport estimators | [bhargava-2024-ope-logged-human-feedback] [xiang-2026-otrope] [dudik-2011-doubly-robust]; world-model OPE for multi-turn agents [liu-2026-adwm-ope-llm-agents] |
| Valid estimates from cheap labels | Many judge labels plus a few human labels | [angelopoulos-2023-ppi] [boyeau-2024-autoeval-done-right] |
| Offline RL on traces | Session trajectories with costs and outcomes | [xu-2026-seqroute] [bhatti-2026-proteus] [si-2025-ccpo] [lei-2026-tacit-switch] |
| Performance prediction | Per-query or per-trajectory success labels | [hasanaliyev-2026-expected-reward-prediction] [xiao-2025-generalized-correctness-models] [unell-2026-deployment-centered-rejection-risk] |
| Distillation from traces | Curated production traces | [gola-2026-langsmith-fine-tuning] [tsymboi-2026-production-traffic-post-training] [yao-2026-mera] |

**Practical notes.**
- **Estimator choice.** Direct regression beat doubly robust estimation in-distribution, and doubly robust was better under shift [li-2026-counterfactual-tool-ranking].
- **Labelling budget.** Supervision cost can be cut by acquiring labels sparsely [lai-2026-saverouter], or by spending evaluation only where it changes the decision. On paid software tasks, better information about model quality saved more than further optimising the assignment [khosravi-2026-which-llm-which-work].
- **Prompt before judging.** Re-optimise prompts for each cheap candidate model before evaluating it. Dropbox moved an LLM judge from o3 to cheaper models this way [wang-2026-dropbox-dash-dspy].

## 3.3 Learning online (live traffic)

- **Contextual bandits that respect partial feedback** [wei-2025-barp] [wang-2025-mixllm].
- **Budget pacing**, using primal-dual or knapsack-style controllers [taberner-miller-2026-paretobandit] [panda-2025-pilot] [li-2026-wiserouter].
- **Drift handling**, through forgetting and audit windows [taberner-miller-2026-paretobandit] [lee-2026-drift-aware-routing]. Provider quality also drifts [he-2026-market-aware-routing].
- **Adding new models** without retraining, using fingerprints computed on anchor prompts [jitkrittum-2025-unirouter].
- **Safe exploration with guarantees:**
  - SLA-guaranteed learning from one-sided complaints [woisetschlager-2026-slarouter];
  - fail-safe anchors [he-2026-market-aware-routing];
  - online conformal risk control under bandit feedback [lee-2025-online-conformal-abstention].
- **Surrogate rewards from judges**, with regret guarantees that hold even when the judge is wrong [sridhar-2026-correlated-bandits-routing].
- **Implicit feedback as reward:**
  - user retrials [bae-2026-contextual-queueing-bandits];
  - a reward model trained on user reactions and validated by live A/B tests [han-2025-rluf];
  - online RL from accept signals [jackson-2025-cursor-tab-online-rl].
- **Production-scale online learning.** Compositional online cascade and memoisation learners run in a production SQL engine and report about 8x lower realistic per-row cost [liskowski-2026-compositional-online-learning].

**Caveat.** Most online-routing evidence is simulation, in which benchmark data is streamed to the bandit. Real-traffic bandit results are rare.

## 3.4 Hybrid designs and how to evaluate them

**Warm-start, then online.**
- PILOT initialises from human preference data, then runs LinUCB [panda-2025-pilot].
- OrcaRouter fits ridge regressors offline, then deploys them as LinUCB [bao-2026-orcarouter].
- Compass finds Pareto configurations offline and switches between them online according to load [gravara-2026-compass].
- Cursor's router is tuned offline by cross-validation, checked on held-out logs, and validated on live traffic, with confidence-gated upgrades [okeefe-2026-cursor-router] (vendor-reported).
- GPT-5's router is trained continuously on live signals [openai-2025-gpt5-system-card].

**Evaluation ladder.**
1. **Offline replay.** Valid for query-level routing only when propensities or full information are available [li-2010-unbiased-offline-eval]. Invalid for step-level swaps [gonuguntla-2026-replay-gap].
2. **Shadow mode.** Measures cost and agreement, but cannot observe how users react to the cheaper output [manoharan-2026-t2mo].
3. **Randomized A/B at session level**, with cache-aware cost accounting [fireworks-2026-firerouter-with-opus] [kohavi-2013-online-experiments]. Pair it with offline evals and trace clustering when swapping models [furman-2026-replit-closing-the-loop-evals]. Watch for drift between the eval set and production traffic [levit-2026-proxydrift].

## 3.5 Recommended staged path for a team with production logs

1. **Instrument.** Log per-step cost, full harness state, and the propensity of every routing decision [yuan-2026-clawtrace] [liu-2026-harness-native-routing].
2. **Build a partial full-information matrix.** Re-run a stratified sample of logged inputs on candidate models and score with a judge. Calibrate the judge against a small human set using prediction-powered inference or causal correction [zhang-2025-meta-router] [angelopoulos-2023-ppi]. Spend labels where they change the decision [khosravi-2026-which-llm-which-work] [lai-2026-saverouter].
3. **Re-optimise prompts for each cheap candidate** before judging it [wang-2026-dropbox-dash-dspy].
4. **Fit simple routers first:** kNN or CARROT-style [li-2025-knn-routing] [somerstep-2025-carrot]. Then search per-role assignments [huang-2026-scope] [hua-2026-agentopt] and train step routers [paul-2026-agentrouter-step-tiers]. Check the gain against share-matched controls [kumar-2026-task-session-routing-eval].
5. **Validate step-level policies** with forked live rollouts, then shadow mode, then a randomized session A/B [gonuguntla-2026-replay-gap] [fireworks-2026-firerouter-with-opus].
6. **Deploy with guardrails:** confidence-gated upgrades and a budget pacer [okeefe-2026-cursor-router] [taberner-miller-2026-paretobandit]. Refine with online bandits under SLA constraints, logging propensities [woisetschlager-2026-slarouter] [bao-2026-orcarouter].
7. **Close the loop.** Distil from routed failures and production traces [yao-2026-mera] [gola-2026-langsmith-fine-tuning]. Audit with gold labels, because cheap-verifier dashboards can hide degradation [rajput-2026-cheap-verifiers-blind-spots]. Monitor drift [levit-2026-proxydrift] [he-2026-market-aware-routing].

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
