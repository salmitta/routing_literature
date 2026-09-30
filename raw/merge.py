"""Merge subagent outputs into literature_db.json / .csv and emit taxonomy counts."""
import csv, glob, json, os, re, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
FIELDS = ["id", "title", "authors", "year", "venue_or_source", "source_type", "url", "code_url",
          "topic_clusters", "routing_granularity", "learning_regime", "signals_used",
          "cost_reduction_reported", "benchmarks", "models_studied", "production_evidence",
          "summary", "key_idea", "relevance_score", "relevance_rationale", "found_via",
          "cites_or_cited_by"]
LIST_FIELDS = {"authors", "topic_clusters", "signals_used", "benchmarks", "models_studied", "cites_or_cited_by"}
CLUSTERS = {
    "C1": "Query-level LLM routing", "C2": "Cascades, deferral and selective escalation",
    "C3": "Step/role/subagent-level model assignment in agents and MAS",
    "C4": "Automated optimization of compound AI system configurations",
    "C5": "Online learning from live traffic", "C6": "Offline learning from logs and customer data",
    "C7": "Quality signals without ground truth", "C8": "Distillation and specialization",
    "C9": "Reasoning and token budget control", "C10": "Caching, context and prompt efficiency",
    "C11": "Serving and systems for agentic workloads",
    "C12": "Architecture-level cost reduction / when multi-agent is worth it",
    "C13": "Benchmarks and evaluation for cost-quality tradeoffs",
    "C14": "Industry products, gateways and production case studies", "C15": "Surveys and theory",
}


def arxiv_id(r):
    for f in ("url", "code_url"):
        m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", str(r.get(f) or ""))
        if m:
            return m.group(1)
    return None


def doi(r):
    m = re.search(r"(10\.\d{4,9}/[^\s\"?#]+)", str(r.get("url") or ""))
    return m.group(1).lower().rstrip(".") if m else None


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def load():
    recs, unverified = [], []
    skip = [x for x in os.environ.get("MERGE_SKIP", "").split(",") if x]  # e.g. MERGE_SKIP=g9_,g10_ for in-progress files
    for p in sorted(glob.glob(os.path.join(RAW, "g*.json"))):
        if any(os.path.basename(p).startswith(x) for x in skip):
            print(f"skipping {os.path.basename(p)}"); continue
        try:
            data = json.load(open(p))
        except Exception as e:
            print(f"!! cannot parse {p}: {e}", file=sys.stderr)
            continue
        if "unverified" in os.path.basename(p):
            unverified += [dict(d, _file=os.path.basename(p)) for d in data]
            continue
        for d in data:
            if isinstance(d, dict) and "_unverified" in d:
                unverified.append(dict(d, _file=os.path.basename(p)))
            elif isinstance(d, dict) and d.get("title"):
                d["_file"] = os.path.basename(p)
                recs.append(d)
    return recs, unverified


def merge_two(a, b):
    """Merge b into a (a is kept as canonical)."""
    for f in FIELDS:
        va, vb = a.get(f), b.get(f)
        if f in LIST_FIELDS:
            seen, out = set(), []
            for x in (va or []) + (vb or []):
                k = json.dumps(x, sort_keys=True).lower() if not isinstance(x, str) else x.lower()
                if k not in seen:
                    seen.add(k); out.append(x)
            a[f] = out
        elif f == "relevance_score":
            a[f] = max(va or 0, vb or 0)
        elif f == "production_evidence":
            a[f] = bool(va) or bool(vb)
        elif f in ("summary", "relevance_rationale", "cost_reduction_reported"):
            if not va or (vb and len(str(vb)) > len(str(va)) and f == "cost_reduction_reported"):
                a[f] = vb
        elif va in (None, "", 0, []):
            a[f] = vb
    fv = [x for x in [a.get("found_via"), b.get("found_via")] if x]
    a["found_via"] = " | ".join(dict.fromkeys(fv)) if fv else None
    a["_alias_ids"] = list(set(a.get("_alias_ids", [a["id"]]) + b.get("_alias_ids", [b.get("id")])))
    return a


EXCLUDE_PATH = os.path.join(RAW, "exclude.json")  # {arxiv_id_or_record_id: reason}


def main():
    recs, unverified = load()
    excl = json.load(open(EXCLUDE_PATH)) if os.path.exists(EXCLUDE_PATH) else {}
    dropped = [r for r in recs if r.get("id") in excl or arxiv_id(r) in excl]
    recs = [r for r in recs if r not in dropped]
    print(f"excluded {len(dropped)} records:", sorted({r['id'] for r in dropped}))
    print(f"loaded {len(recs)} raw records, {len(unverified)} unverified notes")
    canon, key2idx = [], {}
    for r in recs:
        r = {**{f: None for f in FIELDS}, **r}
        r["_alias_ids"] = [r["id"]]
        keys = [k for k in (("ax", arxiv_id(r)), ("doi", doi(r)), ("t", norm_title(r["title"])),
                            ("u", re.sub(r"^https?://(www\.)?|/$", "", (r.get("url") or "").lower()))) if k[1]]
        hit = next((key2idx[k] for k in keys if k in key2idx), None)
        if hit is None:
            canon.append(r); idx = len(canon) - 1
        else:
            merge_two(canon[hit], r); idx = hit
        for k in keys + [("ax", arxiv_id(canon[idx])), ("t", norm_title(canon[idx]["title"]))]:
            if k[1]:
                key2idx[k] = idx
    # unique ids
    seen = Counter()
    for r in canon:
        base = r["id"] or norm_title(r["title"])[:40]
        seen[base] += 1
        r["id"] = base if seen[base] == 1 else f"{base}-{'xbcdefgh'[seen[base]-1]}"
    alias = {}
    for r in canon:
        for a in r["_alias_ids"]:
            alias.setdefault(a, r["id"])
    # resolve and symmetrize citations
    links = defaultdict(set)
    for r in canon:
        for c in r.get("cites_or_cited_by") or []:
            t = alias.get(c)
            if t and t != r["id"]:
                links[r["id"]].add(t); links[t].add(r["id"])
    for r in canon:
        r["cites_or_cited_by"] = sorted(links[r["id"]])
        r["topic_clusters"] = sorted(set(r["topic_clusters"] or []), key=lambda c: int(c[1:]) if c[1:].isdigit() else 99)
    chk_path = os.path.join(RAW, "arxiv_check.json")
    chk = json.load(open(chk_path)) if os.path.exists(chk_path) else {}
    for r in canon:
        c = chk.get(arxiv_id(r) or "")
        if c and c["status"] == "title_mismatch":
            r["title"] = c["arxiv_title"]  # renamed between arXiv versions; use current title
        if c and c["status"] in ("withdrawn", "not_found"):
            r["relevance_score"] = 0
    canon = [r for r in canon if (r.get("relevance_score") or 0) >= 2]
    canon.sort(key=lambda r: (-(r.get("relevance_score") or 0), -(r.get("year") or 0), r["id"]))
    out = [{f: r.get(f) for f in FIELDS} for r in canon]
    json.dump(out, open(os.path.join(ROOT, "literature_db.json"), "w"), indent=2, ensure_ascii=False)
    with open(os.path.join(ROOT, "literature_db.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow({k: ("; ".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items()})
    json.dump(unverified, open(os.path.join(ROOT, "raw", "unverified_merged.json"), "w"), indent=2)
    # stats
    cc = Counter(c for r in out for c in r["topic_clusters"])
    print(f"unique records: {len(out)}")
    for c in CLUSTERS:
        print(f"  {c:4} {cc.get(c,0):4}  {CLUSTERS[c]}")
    print("relevance:", dict(sorted(Counter(r["relevance_score"] for r in out).items())))
    print("granularity:", dict(Counter(r["routing_granularity"] for r in out)))
    print("regime:", dict(Counter(r["learning_regime"] for r in out)))
    print("source_type:", dict(Counter(r["source_type"] for r in out)))
    print("production_evidence:", sum(1 for r in out if r["production_evidence"]))
    bad = [r["id"] for r in out if not r.get("url")]
    print("missing url:", len(bad), bad[:10])


if __name__ == "__main__":
    main()
