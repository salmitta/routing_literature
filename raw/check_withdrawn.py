"""Scan arXiv abs pages for the "This paper has been withdrawn" banner (the API comment field doesn't reliably say so).

Checks every arXiv id found in raw/g*.json; results are cached in raw/withdrawn_check.json and
withdrawn ids are added to raw/exclude.json.
"""
import glob, json, os, re, sys, time, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

RAW = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(RAW, "withdrawn_check.json")
EXCL = os.path.join(RAW, "exclude.json")


def ids():
    out = set()
    for p in glob.glob(os.path.join(RAW, "g*.json")):
        try:
            for r in json.load(open(p)):
                m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", str(r.get("url") or "")) if isinstance(r, dict) else None
                if m:
                    out.add(m.group(1))
        except Exception as e:
            print("skip", p, e, file=sys.stderr)
    return sorted(out)


def check(aid):
    err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(f"https://arxiv.org/abs/{aid}", headers={"User-Agent": "lit-db-verify/1.0"})
            html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
            time.sleep(0.5)
            return aid, ("withdrawn" if "has been withdrawn" in html else "ok")
        except Exception as e:
            err = f"error: {e}"
            time.sleep(15 * (attempt + 1))
    return aid, err


def main():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    excl = json.load(open(EXCL)) if os.path.exists(EXCL) else {}
    todo = [i for i in ids() if cache.get(i) not in ("ok", "withdrawn")]
    print(f"{len(todo)} abs pages to check", flush=True)
    with ThreadPoolExecutor(4) as ex:
        for n, (aid, status) in enumerate(ex.map(check, todo)):
            cache[aid] = status
            if status == "withdrawn":
                excl[aid] = "withdrawn on arXiv (abs-page banner)"
                print("WITHDRAWN", aid, flush=True)
            if n % 50 == 0:
                json.dump(cache, open(CACHE, "w"), indent=0)
                json.dump(excl, open(EXCL, "w"), indent=1)
                print(f"  {n}/{len(todo)}", flush=True)
    json.dump(cache, open(CACHE, "w"), indent=0)
    json.dump(excl, open(EXCL, "w"), indent=1)
    print(Counter(v if not v.startswith("error") else "error" for v in cache.values()))


if __name__ == "__main__":
    main()
