"""Assemble synthesis.md from raw/synth/: lead front matter, cluster sections C1-C15 from S1-S5, lead cross-cutting and back matter."""
import json, os, re
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SY = os.path.join(ROOT, "raw", "synth")
db = json.load(open(os.path.join(ROOT, "literature_db.json")))
sections = {}
for f in ["S1.md", "S2.md", "S3.md", "S4.md", "S5.md"]:
    txt = open(os.path.join(SY, f)).read()
    for m in re.finditer(r"^## (C\d+)\..*?(?=^## |\Z)", txt, re.S | re.M):
        sections[m.group(1)] = m.group(0).rstrip() + "\n"
missing = [f"C{i}" for i in range(1, 16) if f"C{i}" not in sections]
assert not missing, missing
cc = Counter(c for r in db for c in r["topic_clusters"])
cov = (f"{len(db):,} records: {sum(r['relevance_score'] >= 4 for r in db)} at relevance 4–5, "
       f"{sum(bool(r['production_evidence']) for r in db)} with production evidence, "
       f"{sum(r['routing_granularity'] == 'step' for r in db)} step-level and {sum(r['routing_granularity'] == 'subagent' for r in db)} subagent-level. "
       "Records per cluster: " + ", ".join(f"C{i} {cc[f'C{i}']}" for i in range(1, 16)) + ". Full counts are in `taxonomy.md`.")
front = open(os.path.join(SY, "lead_front.md")).read().replace("__N__", f"{len(db):,}").replace("__COVERAGE__", cov)
body = "# Part 1: Topic clusters\n\n" + "\n".join(sections[f"C{i}"] for i in range(1, 16))
out = "\n".join([front.rstrip() + "\n", body, open(os.path.join(SY, "lead_cross.md")).read().rstrip() + "\n", open(os.path.join(SY, "lead_back.md")).read()])
open(os.path.join(ROOT, "synthesis.md"), "w").write(out)
print(f"synthesis.md: {len(out.split()):,} words")
