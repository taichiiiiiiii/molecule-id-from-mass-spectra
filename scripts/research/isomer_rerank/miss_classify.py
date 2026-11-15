#!/usr/bin/env python3
"""#4 (2026-11-15 昼): 1 位が外れた分子を (a)/(b)/(c) に分けて数える（読むだけ）。

(a) 正解が上位 25 のリストに無い
(b) 正解はリストにあるが、1 位の分子式が正解と違う
(c) 正解はリストにあり、1 位が正解と同じ分子式の異性体
同一判定は scripts/validate/mrr25.py の to_key（tautomer 正準化 → InChIKey14）。
分子式は RDKit CalcMolFormula（電荷は式に含めない: 中性化はしない。差が出たら別に数える）。
(a) は補足として「リストに正解の分子式がある/無い」に分ける。
上限の見積もり: (c) の分子の正解を 1 位にしたときの MRR（= 同式の並べ替えが完璧な場合の上限）。
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1])          # 元のフォルダ（読むだけ）
CODE = Path(sys.argv[2])          # クローン（mrr25.py を読む）
HOLDOUT = sys.argv[3]
TAG = sys.argv[4] if len(sys.argv) > 4 else "score+prior+frag"
sys.path.insert(0, str(CODE / "scripts/validate"))
from mrr25 import to_key, read_truth  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def formula(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return CalcMolFormula(m) if m is not None else None


truth_rows = {r["molecule_id"]: r for r in csv.DictReader(open(ROOT / f"outputs/holdouts/{HOLDOUT}/truth.csv"))}
truth = read_truth(ROOT / f"outputs/holdouts/{HOLDOUT}/truth.csv")
preds = {}
with open(ROOT / f"outputs/local_eval/{HOLDOUT}/submission_{TAG}.csv", newline="") as f:
    for r in csv.DictReader(f):
        preds[r["molecule_id"]] = [s for s in r["smiles"].split(";") if s]

cnt = Counter()
rr_now, rr_ub = [], []
rank_c, ncand_same_c, a_formula_in_list = [], [], Counter()
per = []
for mid, t in truth.items():
    tk = to_key(t)
    c = preds.get(mid, [])[:25]
    keys = [to_key(s) for s in c]
    rank = next((i + 1 for i, k in enumerate(keys) if k == tk), None)
    rr = 1.0 / rank if rank else 0.0
    rr_now.append(rr)
    tsmi = truth_rows[mid].get("smiles") or ""
    tf = formula(tsmi) if tsmi else None
    fl = [formula(s) for s in c]
    if rank == 1:
        cls = "hit"
    elif rank is None:
        cls = "a"
        a_formula_in_list["truth_formula_in_list" if tf in fl else "truth_formula_not_in_list"] += 1
    elif fl[0] != tf:
        cls = "b"
    else:
        cls = "c"
        rank_c.append(rank)
        ncand_same_c.append(sum(1 for x in fl if x == tf))
    cnt[cls] += 1
    rr_ub.append(1.0 if cls == "c" else rr)
    per.append(dict(molecule_id=mid, cls=cls, rank=rank, truth_formula=tf, top1_formula=fl[0] if fl else None,
                    n_same_formula_in_top25=sum(1 for x in fl if x == tf)))

n = len(truth)
miss = n - cnt["hit"]
out = dict(holdout=HOLDOUT, tag=TAG, n=n, mrr25_recomputed=round(sum(rr_now) / n, 4),
           top1_hits=cnt["hit"], top1_miss=miss, a=cnt["a"], b=cnt["b"], c=cnt["c"],
           a_check_sum=cnt["a"] + cnt["b"] + cnt["c"] == miss,
           a_split=dict(a_formula_in_list),
           c_truth_rank_counts=dict(sorted(Counter(rank_c).items())),
           c_median_same_formula_in_top25=(sorted(ncand_same_c)[len(ncand_same_c) // 2] if ncand_same_c else None),
           b_truth_rank_counts=dict(sorted(Counter(p["rank"] for p in per if p["cls"] == "b").items())),
           mrr25_upper_if_c_fixed=round(sum(rr_ub) / n, 4),
           truth_formula_none=sum(1 for p in per if p["truth_formula"] is None))
print(json.dumps(out, ensure_ascii=False))
outp = Path(sys.argv[5]) if len(sys.argv) > 5 else None
if outp:
    outp.write_text(json.dumps(dict(summary=out, per=per), ensure_ascii=False, indent=1))
