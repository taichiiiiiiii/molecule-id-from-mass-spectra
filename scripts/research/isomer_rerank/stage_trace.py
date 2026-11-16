#!/usr/bin/env python3
"""(c) の分子が、score だけ・score+prior の段で何位だったかを数える（読むだけ）。"""
import csv, json, sys
from collections import Counter
from pathlib import Path
ROOT, CODE, H, MISS = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
sys.path.insert(0, str(CODE / "scripts/validate"))
from mrr25 import to_key, read_truth
truth = read_truth(ROOT / f"outputs/holdouts/{H}/truth.csv")
def load(tag):
    return {r["molecule_id"]: r["smiles"].split(";") for r in csv.DictReader(open(ROOT / f"outputs/local_eval/{H}/submission_{tag}.csv", newline=""))}
per = {p["molecule_id"]: p for p in json.loads(MISS.read_text())["per"]}
out = {}
for tag in ("score", "score+prior"):
    P = load(tag); c = Counter()
    for mid, p in per.items():
        tk = to_key(truth[mid]); ks = [to_key(s) for s in P.get(mid, [])[:25]]
        r = next((i + 1 for i, k in enumerate(ks) if k == tk), None)
        c[(p["cls"], "top1" if r == 1 else ("in25" if r else "out"))] += 1
    out[tag] = {f"{a}:{b}": v for (a, b), v in sorted(c.items())}
print(json.dumps(dict(holdout=H, **out), ensure_ascii=False))
