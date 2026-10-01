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
