#!/usr/bin/env python3
"""
ランカーの追加点検（Issue #29、採否の条件には入れない）: 保存済みモデル outputs/ranker/model.txt を別のホールドアウトに当て、
run.py の段の出力（submission_<tag>.csv）と MRR@25・差の 95% 区間を比べる。

    python scripts/ranker/apply_eval.py --holdout enveda_v1 --baselines score+prior,score+prior+lib+pubchem
出力: outputs/ranker/apply_<holdout>.json、outputs/local_eval/<holdout>/submission_ranker.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))
import train_eval as T  # noqa: E402


def main() -> None:
    import lightgbm as lgb
    from mrr25 import read_truth
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--baselines", default="score+prior,score+prior+lib+pubchem,score+joint")
    args = ap.parse_args()
    model = lgb.Booster(model_file=str(ROOT / "outputs/ranker/model.txt"))
    ev = T.load([args.holdout])
    ev["ranker"] = model.predict(ev[T.FEATURES])
    truth = read_truth(ROOT / f"outputs/holdouts/{args.holdout}/truth.csv")
    preds = T.ranked_lists(ev, "ranker")
    with open(ROOT / f"outputs/local_eval/{args.holdout}/submission_ranker.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["molecule_id", "smiles"])
        for m in truth:
            w.writerow([m, ";".join(preds.get(m, []))])
    rr = T.per_molecule_rr(truth, preds)
    top1_src = ev.sort_values("ranker", ascending=False).groupby("molecule_id").src_pubchem.first()
    res = dict(holdout=args.holdout, n=len(truth), ranker=round(float(np.mean(list(rr.values()))), 4),
               top1_from_pubchem=int(top1_src.sum()), truth_in_pool=int(ev[(ev.label == 1) & (ev.src_pubchem == 0)].molecule_id.nunique()),
               truth_in_pubchem=int(ev[(ev.label == 1) & (ev.src_pubchem == 1)].molecule_id.nunique()))
    for tag in args.baselines.split(","):
        p = ROOT / f"outputs/local_eval/{args.holdout}/submission_{tag}.csv"
        if not p.is_file():
            res[tag] = None
            continue
        rb = T.per_molecule_rr(truth, T.read_sub(p))
        res[tag] = round(float(np.mean(list(rb.values()))), 4)
        res[f"ranker_minus_{tag}"] = T.bootstrap_diff(rr, rb)
    (ROOT / f"outputs/ranker/apply_{args.holdout}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
