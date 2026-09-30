"""Emit compact per-cluster digests of literature_db.json for synthesis writers."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "raw", "digests")
os.makedirs(OUT, exist_ok=True)
db = json.load(open(os.path.join(ROOT, "literature_db.json")))


def line(r, full):
    parts = [f"[{r['id']}] ({r['year']}, {r['venue_or_source']}, rel={r['relevance_score']}, gran={r['routing_granularity']}, "
             f"regime={r['learning_regime']}, prod={'Y' if r['production_evidence'] else 'N'}, clusters={','.join(r['topic_clusters'])})",
             f"  title: {r['title']}", f"  key_idea: {r['key_idea']}"]
    if r.get("cost_reduction_reported"):
        parts.append(f"  cost: {r['cost_reduction_reported']}")
    if full:
        parts.append(f"  summary: {r['summary']}")
        if r.get("benchmarks"):
            parts.append(f"  benchmarks: {', '.join(map(str, r['benchmarks']))[:200]}")
        if r.get("signals_used"):
            parts.append(f"  signals: {', '.join(map(str, r['signals_used']))[:200]}")
    return "\n".join(parts)


for c in [f"C{i}" for i in range(1, 16)]:
    rs = [r for r in db if c in r["topic_clusters"]]
    rs.sort(key=lambda r: (-r["relevance_score"], -(r["year"] or 0)))
    with open(os.path.join(OUT, f"{c}.txt"), "w") as f:
        f.write(f"# {c}: {len(rs)} records (full detail for rel>=4; one-liners for rel<=3)\n\n")
        for r in rs:
            f.write(line(r, r["relevance_score"] >= 4) + "\n\n")
print({c: sum(c in r["topic_clusters"] for r in db) for c in [f"C{i}" for i in range(1, 16)]})
