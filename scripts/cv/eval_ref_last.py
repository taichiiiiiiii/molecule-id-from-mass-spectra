#!/usr/bin/env python3
"""
ref_last をホールドアウトの最終出力（Kaggle の 0.479 パイプライン出力）に掛けて MRR@25 を比べる（Issue #22）。

    python scripts/cv/eval_ref_last.py --holdout np_v2 [--run np_v2]

学習データの参照集合は「train.parquet の inchikey14 − ホールドアウトの drop_keys」（Kaggle 上の hold_comp と同じ）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate")); sys.path.insert(0, str(ROOT / "scripts/submit"))
import mrr25  # noqa: E402
import ref_last as RL  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = args.run or args.holdout
    d = ROOT / f"outputs/holdout_eval/{run}"
    truth = mrr25.read_truth(ROOT / f"outputs/holdouts/{args.holdout}/truth.csv")
    drop = set((ROOT / f"data/holdouts/{args.holdout}/drop_keys.txt").read_text().split())
    ref = set(pq.read_table(ROOT / "data/train.parquet", columns=["inchikey14"]).column(0).to_pylist()) - drop
    lib = pd.read_parquet(d / "neural_candidate_features.parquet", columns=["molecule_id", "library_max"])
    lib = {str(m): float(v) for m, v in lib.groupby("molecule_id").library_max.max().items()}
    sub = {r["molecule_id"]: mrr25.parse_candidates(r["smiles"]) for r in csv.DictReader(open(d / "submission.csv"))}
    keys_of = RL.make_keys_of()
    new, n_changed, n_top1 = {}, 0, 0
    for m, c in sub.items():
        new[m], ch = RL.ref_last(c, keys_of, ref, lib.get(m))
        n_changed += ch; n_top1 += bool(c) and new[m][0] != c[0]
    rr0 = {m: mrr25.reciprocal_rank(mrr25.to_key(t), sub.get(m, [])) for m, t in truth.items()}
    rr1 = {m: mrr25.reciprocal_rank(mrr25.to_key(t), new.get(m, [])) for m, t in truth.items()}
    out = dict(holdout=args.holdout, run=run, n=len(truth), lib_tau=RL.LIB_TAU,
               before=round(sum(rr0.values()) / len(truth), 4), after=round(sum(rr1.values()) / len(truth), 4),
               lists_changed=n_changed, top1_changed=n_top1,
               better=sum(rr1[m] > rr0[m] for m in truth), worse=sum(rr1[m] < rr0[m] for m in truth))
    print(json.dumps(out))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / f"results/ref_last_{run}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
