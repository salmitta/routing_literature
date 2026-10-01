# Synthesis: cutting the cost of production multi-agent LLM systems

**Scope.** This synthesis looks at how to lower the cost of an agentic pipeline in which every subagent and step calls one frontier model, without losing task quality. The cost levers covered are:
- query-level routing;
- step- and subagent-level routing or downscaling;
- learning configurations from offline logs or live traffic;
- supporting levers: caching, context management, distillation, reasoning budgets and serving.

**Source.** Everything here is written from the merged database only: `literature_db.json`, __N__ unique records. Claims cite record IDs in square brackets. Every number in the tradeoff tables appears in the cited record's own fields, and an automated check confirmed this for every table row. Records whose arXiv versions were withdrawn have been removed.

**How to read the numbers.**
- Cost/quality results are copied as reported. They can be compared only within the same benchmark, baseline and price sheet.
- Vendor blog and product numbers are marked as vendor-reported. They come from internal traffic and are not independently replicated.
- A large share of the step-level and online-learning work is unreviewed 2026 preprints. Treat these as hypotheses to test, not settled results.

**Coverage.** __COVERAGE__

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
