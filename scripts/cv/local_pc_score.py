#!/usr/bin/env python3
"""
PubChem チャネルの FPNet ロジット（pc_logits.npz）で、任意の候補集合をローカル採点する。

0.479 版の PubChem チャネルは score = fp[bits] @ z（fp は raw_fingerprint、bits は fp_bits.npy の選択ビット）。
ここでは候補プール（pool_meta）の ±10 ppm 窓を候補集合にして、
  (a) クラス 2: 正解がプールに居るときの FPNet 単独の順位
  (b) クラス 3 の近さ: 正解をプールから外したとき、FPNet 上位 25 の中で正解に最も近い構造（Tanimoto, 同一組成式）
を測る。(b) は「新規構造を既知の類似体から編集で作れるか」の見積り。

    python scripts/cv/local_pc_score.py --run np_v2_pcopen_z --holdout np_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "data/ext/v4g/casmi26-v3-models/code")
from casmi import chem  # noqa: E402

PPM = 10.0
K = 25


def fp_sel(smiles: str, bits: np.ndarray):
    fp = chem.raw_fingerprint(smiles)
    return None if fp is None else fp[bits].astype(np.float32)


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a * b).sum())
    return inter / max(float(a.sum() + b.sum() - inter), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--pool", default="data/ext/v4g/casmi26-v2-pool/pool_meta.parquet")
    ap.add_argument("--bits", default="data/ext/v4g/casmi26-v2-pool/fp_bits.npy")
    args = ap.parse_args()

    bits = np.load(args.bits)
    z = np.load(f"outputs/holdout_eval/{args.run}/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in z["__targets__"]}
    truth = pd.read_csv(f"outputs/holdouts/{args.holdout}/truth.csv", dtype=str)
    pool = pd.read_parquet(args.pool, columns=["key", "smiles", "formula", "mass"]).sort_values("mass").reset_index(drop=True)
    mass = pool.mass.values
    rows = []
    for mid, k14, tsmi in zip(truth.molecule_id, truth.inchikey14, truth.smiles):
        if mid not in targets:
            continue
        zz = z[mid].astype(np.float32)
        t = targets[mid]
        tol = t * PPM * 1e-6
        a, b = np.searchsorted(mass, t - tol, "left"), np.searchsorted(mass, t + tol, "right")
        win = pool.iloc[a:b]
        tfp = fp_sel(tsmi, bits)
        tform = chem.mol_props(tsmi)[0] if tfp is not None else None
        fz_t = float(tfp @ zz) if tfp is not None else np.nan
        cand = []
        for key, smi, form in zip(win.key, win.smiles, win.formula):
            fp = fp_sel(smi, bits)
            if fp is None:
                continue
            cand.append((float(fp @ zz), key, form, fp))
        cand.sort(key=lambda x: -x[0])
        # (a) クラス 2: 正解（プール内の同じキー）の順位、同キー重複は 1 つに
        seen, rank, r = set(), 0, 0
        for fz, key, _, _ in cand:
            if key in seen:
                continue
            seen.add(key); r += 1
            if key == k14:
                rank = r; break
        # (b) クラス 3: 正解キーを外した上位 K の近さ
        top = [c for c in cand if c[1] != k14][:K]
        best_tani = max((tanimoto(tfp, c[3]) for c in top), default=0.0) if tfp is not None else np.nan
        same_formula = sum(c[2] == tform for c in top)
        rows.append(dict(molecule_id=mid, n_window=len(win), rank_class2=rank, fz_truth=fz_t,
                         fz_top=cand[0][0] if cand else np.nan, best_tanimoto_top25=best_tani,
                         same_formula_top25=same_formula, truth_in_pool=bool((win.key == k14).any())))
    df = pd.DataFrame(rows)
    rr = np.where(df.rank_class2 > 0, 1 / df.rank_class2.clip(lower=1), 0.0)
    out = dict(run=args.run, n=len(df), window_median=int(df.n_window.median()),
               class2_fpnet_only=dict(mrr25=round(float(np.where(df.rank_class2 <= K, rr, 0).mean()), 4),
                                      top1=round(float((df.rank_class2 == 1).mean()), 3),
                                      in_top25=round(float(((df.rank_class2 > 0) & (df.rank_class2 <= K)).mean()), 3)),
               class3_proxy=dict(best_tanimoto_median=round(float(df.best_tanimoto_top25.median()), 3),
                                 best_tanimoto_ge_0_7=round(float((df.best_tanimoto_top25 >= 0.7).mean()), 3),
                                 best_tanimoto_ge_0_85=round(float((df.best_tanimoto_top25 >= 0.85).mean()), 3),
                                 same_formula_in_top25=round(float((df.same_formula_top25 > 0).mean()), 3)))
    print(json.dumps(out, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path(f"results/local_pc_score_{args.run}.json").write_text(json.dumps(out, indent=2))
    df.to_csv(f"results/local_pc_score_{args.run}.csv", index=False)


if __name__ == "__main__":
    main()
