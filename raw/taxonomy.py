"""Write taxonomy.md with per-node counts from literature_db.json."""
import json, os
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db = json.load(open(os.path.join(ROOT, "literature_db.json")))
NODES = [
 ("C1", "Query-level LLM routing", "Pick one model per incoming request from a pool, using predicted quality and cost (classifiers, kNN, matrix factorization, preference-trained and RL routers)."),
 ("C2", "Cascades, deferral and selective escalation", "Run a cheap model first and escalate on low confidence, failed verification or disagreement; includes learning-to-defer and abstention."),
 ("C3", "Step-, role- and subagent-level model assignment", "Choose models per turn, per action, per role or per subagent inside agent trajectories and multi-agent systems; includes topology and communication pruning."),
 ("C4", "Automated optimization of compound AI system configurations", "Search over per-module model choice, prompts, workflow structure and hyperparameters, often jointly for cost and quality."),
 ("C5", "Online learning from live traffic", "Bandits, online RL, budget pacing, drift handling, A/B tests and implicit feedback for routing and config choice."),
 ("C6", "Offline learning from logs and customer data", "Off-policy evaluation, counterfactual learning from logged outcomes, performance prediction and reward models for routing."),
 ("C7", "Quality signals without ground truth", "LLM judges, process and outcome reward models, verifiers, uncertainty estimates, probes and agent failure attribution."),
 ("C8", "Distillation and specialization", "Training small models on agent traces or production data, SFT and RL for small agents, task-specific specialists."),
 ("C9", "Reasoning and token budget control", "Adaptive reasoning effort, thinking budgets, overthinking, think/no-think routing and test-time compute allocation."),
 ("C10", "Caching, context and prompt efficiency", "Prompt and KV caching, semantic caches, plan caching, context compression and agent memory or context management."),
 ("C11", "Serving and systems for agentic workloads", "Agent- and program-aware scheduling, prefix sharing, speculative decoding, multi-model and multi-adapter serving."),
 ("C12", "Architecture-level cost reduction and when multi-agent is worth it", "Single- vs multi-agent comparisons, orchestration patterns, token-consumption analyses and cost-aware agent evaluation."),
 ("C13", "Benchmarks and evaluation for cost-quality tradeoffs", "Routing benchmarks, agent benchmarks with cost reporting, reliability metrics (pass^k) and Pareto evaluation."),
 ("C14", "Industry products, gateways and production case studies", "Routers and gateways sold as products, provider docs, and company reports of cost reduction on real traffic."),
 ("C15", "Surveys and theory", "Surveys of routing, efficient inference, efficient agents and reasoning; formal treatments of routing as optimization or stopping."),
]
cc = Counter(c for r in db for c in r["topic_clusters"])
prim = Counter(r["topic_clusters"][0] for r in db if r["topic_clusters"])
L = ["# Taxonomy", "",
     f"Final topic taxonomy for `literature_db.json` ({len(db):,} unique records). Records can carry several cluster tags, so the per-node counts add up to more than the total.", "",
     "| ID | Node | Records tagged | Relevance 4–5 | Production evidence | Scope |", "|---|---|---:|---:|---:|---|"]
for c, name, scope in NODES:
    rs = [r for r in db if c in r["topic_clusters"]]
    L.append(f"| {c} | {name} | {len(rs)} | {sum(r['relevance_score'] >= 4 for r in rs)} | {sum(bool(r['production_evidence']) for r in rs)} | {scope} |")
L += ["", "## Cross-cutting facets", "", "Every record is also classified along three facets that cut across clusters.", ""]
for f, title in [("routing_granularity", "Routing granularity"), ("learning_regime", "Learning regime"), ("source_type", "Source type")]:
    L += [f"### {title}", "", "| Value | Records |", "|---|---:|"]
    for k, v in Counter(r[f] or "null" for r in db).most_common():
        L.append(f"| {k} | {v} |")
    L.append("")
L += ["### Relevance score", "", "| Score | Records |", "|---|---:|"]
for k, v in sorted(Counter(r["relevance_score"] for r in db).items(), reverse=True):
    L.append(f"| {k} | {v} |")
L += ["", "### Step-level and subagent-level work by learning regime", "",
      "This facet matters most for the premise, so it is broken out here.", "", "| Learning regime | step | subagent |", "|---|---:|---:|"]
for k in ["training-free", "supervised-offline", "RL-offline", "bandit-online", "RL-online", "hybrid", "n/a"]:
    L.append(f"| {k} | {sum(r['routing_granularity']=='step' and r['learning_regime']==k for r in db)} | {sum(r['routing_granularity']=='subagent' and r['learning_regime']==k for r in db)} |")
L += ["", "### Year", "", "| Year | Records |", "|---|---:|"]
for k, v in sorted(Counter(r["year"] or "n.d." for r in db).items(), key=lambda x: str(x[0]), reverse=True):
    L.append(f"| {k} | {v} |")
open(os.path.join(ROOT, "taxonomy.md"), "w").write("\n".join(L) + "\n")
print("taxonomy.md written")
