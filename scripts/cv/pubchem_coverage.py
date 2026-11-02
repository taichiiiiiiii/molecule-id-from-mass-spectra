#!/usr/bin/env python3
"""
PubChem チャネル（プール外の提案）の正解カバレッジと順位を、ホールドアウトで測る。

入力: outputs/holdout_eval/<run>/pc_lists.json（pc_keys は順位順の scorer key）、pubchem_promotion_audit.csv
    python scripts/cv/pubchem_coverage.py --run np_v2_pcopen --holdout np_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--holdout", required=True)
    args = ap.parse_args()
    d = Path(f"outputs/holdout_eval/{args.run}")
    truth = pd.read_csv(f"outputs/holdouts/{args.holdout}/truth.csv", dtype=str)
    pc = json.load(open(d / "pc_lists.json"))
    rows = []
    for m, k in zip(truth.molecule_id, truth.inchikey14):
        p = pc.get(m, {})
        keys = p.get("pc_keys", [])
        r = next((i + 1 for i, x in enumerate(keys) if x == k), 0)
        fz = p.get("pc_fz", [])
        rows.append(dict(molecule_id=m, rank=r, n=len(keys), S=p.get("S"), fz_top=fz[0] if fz else np.nan,
                         fz_truth=fz[r - 1] if r else np.nan, best_pool_fz=p.get("best_pool_fz")))
    df = pd.DataFrame(rows)
    rr = np.where(df["rank"] > 0, 1 / df["rank"].clip(lower=1), 0.0)
    out = dict(run=args.run, n=len(df), coverage25=round(float((df["rank"] > 0).mean()), 3),
               top1=round(float((df["rank"] == 1).mean()), 3), mrr25_pc_only=round(float(rr.mean()), 4),
               rank_dist=df["rank"].clip(upper=26).value_counts().sort_index().to_dict())
    aud = d / "pubchem_promotion_audit.csv"
    if aud.is_file():
        a = pd.read_csv(aud, dtype={"molecule_id": str}).merge(df[["molecule_id", "rank"]], on="molecule_id", how="left")
        a["pc_top_is_truth"] = a["rank"] == 1
        out["promotion"] = dict(n_audited=int(len(a)), promoted=int(a.new_promote.sum()),
                                promoted_and_correct=int((a.new_promote & a.pc_top_is_truth).sum()),
                                correct_not_promoted=int((~a.new_promote & a.pc_top_is_truth).sum()),
                                promoted_but_wrong=int((a.new_promote & ~a.pc_top_is_truth).sum()))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    Path("results").mkdir(exist_ok=True)
    Path(f"results/pubchem_coverage_{args.run}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    df.to_csv(f"results/pubchem_coverage_{args.run}.csv", index=False)


if __name__ == "__main__":
    main()
