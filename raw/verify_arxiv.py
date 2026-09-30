"""Batch-check every arXiv URL in literature_db.json against the arXiv API.

Flags: id not found, title mismatch, withdrawn. Writes raw/arxiv_check.json.
"""
import json, os, re, sys, time, urllib.request, difflib
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "raw", "arxiv_check.json")
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (t or "").lower())).strip()


def fetch(ids):
    url = "http://export.arxiv.org/api/query?max_results=%d&id_list=%s" % (len(ids), ",".join(ids))
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "lit-db-verify/1.0"}), timeout=60) as r:
                return r.read()
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"  retry {attempt} after {e}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None


def main():
    db = json.load(open(os.path.join(ROOT, "literature_db.json")))
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = {}
    for r in db:
        m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", r.get("url") or "")
        if m and m.group(1) not in done:
            todo[m.group(1)] = r["title"]
    ids = list(todo)
    print(f"{len(ids)} arXiv ids to check ({len(done)} cached)")
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        xml = fetch(batch)
        if xml is None:
            print("giving up on batch", i); continue
        got = {}
        for e in ET.fromstring(xml).findall("a:entry", NS):
            eid = re.search(r"(\d{4}\.\d{4,5})", e.findtext("a:id", "", NS) or "")
            if not eid:
                continue
            got[eid.group(1)] = {
                "title": " ".join((e.findtext("a:title", "", NS) or "").split()),
                "comment": e.findtext("x:comment", "", NS) or "",
                "summary": (e.findtext("a:summary", "", NS) or "")[:300],
                "published": e.findtext("a:published", "", NS),
            }
        for aid in batch:
            g = got.get(aid)
            if not g:
                done[aid] = {"status": "not_found", "db_title": todo[aid]}; continue
            sim = difflib.SequenceMatcher(None, norm(g["title"]), norm(todo[aid])).ratio()
            wd = bool(re.search(r"withdrawn", g["comment"] + " " + g["summary"], re.I))
            done[aid] = {"status": "withdrawn" if wd else ("ok" if sim >= 0.85 else "title_mismatch"),
                         "sim": round(sim, 3), "db_title": todo[aid], "arxiv_title": g["title"],
                         "published": g["published"]}
        json.dump(done, open(OUT, "w"), indent=1)
        print(f"  {min(i+50, len(ids))}/{len(ids)}")
        time.sleep(3.5)
    from collections import Counter
    print(Counter(v["status"] for v in done.values()))
    for k, v in done.items():
        if v["status"] != "ok":
            print(k, v["status"], v.get("sim"), "| DB:", v["db_title"][:70], "| AX:", (v.get("arxiv_title") or "")[:70])


if __name__ == "__main__":
    main()
