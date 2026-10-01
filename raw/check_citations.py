"""Report bracketed record-id citations in markdown files that don't exist in literature_db.json."""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ids = {r["id"] for r in json.load(open(os.path.join(ROOT, "literature_db.json")))}
bad_total = 0
for p in sys.argv[1:]:
    txt = open(p).read()
    cites = re.findall(r"\[([a-z0-9][a-z0-9.\-]+-\d{4}[a-z0-9.\-]*)\]", txt)
    cites += [c for grp in re.findall(r"\[([a-z0-9.\-]+-\d{4}[^\]]*;[^\]]+)\]", txt) for c in re.split(r";\s*", grp)]
    bad = sorted({c for c in cites if c not in ids})
    bad_total += len(bad)
    print(f"{os.path.basename(p)}: {len(set(cites))} distinct ids cited, {len(bad)} unknown", bad[:40])
sys.exit(1 if bad_total else 0)
