"""Write UNVERIFIED.md from the subagents' unverified notes and the exclusion list."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
u = json.load(open(os.path.join(ROOT, "raw", "unverified_merged.json")))
excl = json.load(open(os.path.join(ROOT, "raw", "exclude.json")))
COVERAGE = ("coverage", "blog", "posts", "case studies", "skipped", "product page", "model optimizer", "announcement", "engine behind", "s2 cited-by")
import re
def name(x): return str(x.get("_unverified") or x.get("title") or "")
def is_cov(x): return any(k in name(x).lower() for k in COVERAGE)
norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
titles = {norm(r["title"]) for r in json.load(open(os.path.join(ROOT, "literature_db.json")))}
later = [x for x in u if norm(name(x).split(" (")[0]) and any(t.startswith(norm(name(x).split(" (")[0])) for t in titles)]
u = [x for x in u if x not in later]
print("now verified and in DB:", [name(x) for x in later])
L = ["# Unverified and excluded items", "",
     "These items were seen during search but are **not** in `literature_db.json`. Each one either couldn't be verified to the standard the brief requires (a resolving source with title, authors, year and an abstract or full text), or was removed after verification. They are listed so they can be chased by hand.", "",
     "## Seeds", "",
     "Every seed in the brief was found and verified, with one exception: SPROUT has no paper of its own. It is the dataset released with the CARROT paper, which is in the database as `somerstep-2025-carrot`.", "",
     "## Removed after verification", "", "| arXiv ID | Reason |", "|---|---|"]
L += [f"| [{k}](https://arxiv.org/abs/{k}) | {v} |" for k, v in excl.items()]
L += ["", "## Candidate papers that could not be verified", "", "| Item | What was found | Found by |", "|---|---|---|"]
L += [f"| {name(x)} | {str(x.get('note','')).replace('|','/')} | {x['_file'].split('_')[0]} |" for x in u if not is_cov(x)]
L += ["", "## Sources that couldn't be reached, and coverage notes", "",
      "These notes record where a source was unreachable or the session's web-search quota ran out. They show where coverage is thinner than the brief asked for.", "",
      "| Source | Note | Found by |", "|---|---|---|"]
L += [f"| {name(x)} | {str(x.get('note','')).replace('|','/')} | {x['_file'].split('_')[0]} |" for x in u if is_cov(x)]
open(os.path.join(ROOT, "UNVERIFIED.md"), "w").write("\n".join(L) + "\n")
print(sum(not is_cov(x) for x in u), "candidates;", sum(is_cov(x) for x in u), "coverage notes")
