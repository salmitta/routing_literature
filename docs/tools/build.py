"""Build the GitHub Pages site in docs/ from literature_db.json.

Usage: python3 docs/tools/build.py [--no-embed]
Embeddings need Node plus @huggingface/transformers@3.8.1. Set TFJS_DIR to a folder where it is npm-installed
(default: docs/tools); embed.mjs is copied there and run, since ESM imports resolve from the script's folder.
"""
import datetime, json, os, shutil, subprocess, sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(TOOLS)
ROOT = os.path.dirname(SITE)
TFJS = "3.8.1"  # must match the version used by embed.mjs
CLUSTERS = {
    "C1": "Query-level LLM routing", "C2": "Cascades, deferral & escalation",
    "C3": "Step / subagent-level model assignment", "C4": "Compound AI config optimization",
    "C5": "Online learning from live traffic", "C6": "Offline learning from logs",
    "C7": "Quality signals without ground truth", "C8": "Distillation & specialization",
    "C9": "Reasoning & token budgets", "C10": "Caching, context & prompts",
    "C11": "Serving & systems", "C12": "Architecture & when MAS is worth it",
    "C13": "Benchmarks & cost-aware evaluation", "C14": "Industry products & case studies",
    "C15": "Surveys & theory",
}


def s(x):
    return "" if x is None else str(x)


def lst(x):
    return [str(v) for v in (x or []) if v is not None]


def main():
    db = json.load(open(os.path.join(ROOT, "literature_db.json")))
    rows = [[s(r["id"]), s(r["title"]), lst(r["authors"]), r["year"], s(r["venue_or_source"]), s(r["source_type"]),
             s(r["url"]), s(r["code_url"]), lst(r["topic_clusters"]), s(r["routing_granularity"]), s(r["learning_regime"]),
             lst(r["signals_used"]), s(r["cost_reduction_reported"]), lst(r["benchmarks"]), lst(r["models_studied"]),
             bool(r["production_evidence"]), s(r["summary"]), s(r["key_idea"]), r["relevance_score"] or 0,
             s(r["relevance_rationale"]), s(r["found_via"]), lst(r["cites_or_cited_by"])] for r in db]
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    extras = []
    for name in ("synthesis.md", "taxonomy.md", "UNVERIFIED.md"):
        if os.path.exists(os.path.join(ROOT, name)):
            shutil.copy(os.path.join(ROOT, name), os.path.join(SITE, name))
            extras.append(f' · <a href="{name}">{name.split(".")[0].lower()}</a>')
    shutil.copy(os.path.join(ROOT, "literature_db.json"), os.path.join(SITE, "literature_db.json"))
    shutil.copy(os.path.join(ROOT, "literature_db.csv"), os.path.join(SITE, "literature_db.csv"))
    with open(os.path.join(SITE, "literature_db.jsonl"), "w") as f:
        for r in db:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    open(os.path.join(SITE, ".nojekyll"), "w").close()

    html = open(os.path.join(TOOLS, "template.html")).read()
    for k, v in {"__DATA__": data, "__CLUSTERS__": json.dumps(CLUSTERS), "__TFJS__": TFJS,
                 "__SNAPSHOT__": datetime.date.today().isoformat(), "__EXTRA_LINKS__": "".join(extras)}.items():
        html = html.replace(k, v)
    open(os.path.join(SITE, "index.html"), "w").write(html)
    print(f"index.html: {len(db)} records, {len(html) / 1e6:.2f} MB")

    if "--no-embed" in sys.argv:
        return
    texts = [f"{r['title']}. {r['key_idea'] or ''} {r['summary'] or ''}" for r in db]
    tpath = os.path.join(TOOLS, "texts.json")
    json.dump(texts, open(tpath, "w"))
    tdir = os.environ.get("TFJS_DIR", TOOLS)
    script = os.path.join(tdir, "embed.mjs")
    if os.path.abspath(tdir) != TOOLS:
        shutil.copy(os.path.join(TOOLS, "embed.mjs"), script)
    subprocess.run(["node", script, tpath, os.path.join(SITE, "embeddings.bin")], check=True)
    os.remove(tpath)


if __name__ == "__main__":
    main()
