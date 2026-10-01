"""For each markdown table row ending in a record id, check that every number in the row's result cell
appears somewhere in that record's text (cost, summary, key idea, title, benchmarks, rationale)."""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "literature_db.json")))}
num = re.compile(r"\d+(?:[.,]\d+)*")
def text(r):
    return " ".join(str(r.get(k) or "") for k in ("cost_reduction_reported", "summary", "key_idea", "title", "relevance_rationale")) + " " + " ".join(map(str, (r.get("benchmarks") or []) + (r.get("models_studied") or [])))
bad = 0; rows = 0
for p in sys.argv[1:]:
    for line in open(p):
        if not line.startswith("|") or "---" in line: continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        ids = re.findall(r"\[([a-z0-9][a-z0-9.\-]+-\d{4}[a-z0-9.\-]*)\]", cells[-1]) if cells else []
        if len(cells) < 3 or not ids: continue
        rows += 1
        hay = " ".join(text(db[i]) for i in ids if i in db).replace(",", "")
        missing = [n for n in num.findall(cells[1]) if n.replace(",", "") not in hay]
        if missing:
            bad += 1; print(f"{os.path.basename(p)} {ids[0]}: {missing} | {cells[1][:110]}")
print(f"{rows} table rows checked, {bad} with numbers not found in the record")
