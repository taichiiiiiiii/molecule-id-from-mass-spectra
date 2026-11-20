#!/usr/bin/env python3
"""
Issue #169: 「家族（Murcko 骨格）に特有の断片・中性損失」の証拠の段（rep、docs/research/diagnostic-reporters.md §6）が
v3 の 4 集合で届きうる範囲を、採点せずに数える（記録だけ。採否には使わない）。

1. 並べ替えの上限（家族の単位）: 対照 score+prior+frag の最終 25 件で、正解の上にある候補のうち、正解と同じ Murcko 骨格の
   候補は家族の証拠では正解と区別できない。家族の違う候補を全部正解の下へ回せた場合の正解の順位は
   1 + （正解の上にある同じ骨格の候補の数）。その伸び Σ(1/best − 1/r) / n が上限（25 件の外の正解は 0 とする）
2. 家族の代表の有無: 正解と同じ Murcko 骨格を持つ train の構造（正解自身・その集合の drop_keys・集合の全正解を
   inchikey14・key・rawkey のどれかで除いた後）の数。0 なら、その正解の家族の証拠は train から作れない

    .venv/bin/python scripts/cv/reporter_reach.py [--out outputs/local_eval/reporter_reach.json]

読むのは truth.csv・対照の submission・train_structs.parquet・drop_keys.txt だけ（スペクトルは読まない）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

HOLDOUTS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
CTRL = "score+prior+frag"
K = 25


def scaffold(smiles: str) -> str | None:
    """Murcko 骨格（RDKit の GetScaffoldForMol、正準 SMILES）。環の無い分子は ""。読めなければ None。"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))


def best_rank(above: list, truth_fam) -> int:
    """正解の上にある候補の骨格の列 above と正解の骨格から、家族の違う候補を全部下へ回した場合の正解の順位。
    骨格が読めない候補（None）と、正解の骨格が読めない場合は、区別できない（同じ家族）とみなす（上限を大きくしない向き）。"""
    if truth_fam is None:
        return len(above) + 1
    return 1 + sum(1 for f in above if f is None or f == truth_fam)


def summarize_bound(rows: list, n: int) -> dict:
    """rows = [(r, best, top1_same)]（r は対照の正解の順位、圏外は r=0）。n は集合の分子数。"""
    inside = [(r, b, s) for r, b, s in rows if r >= 2]
    gain = sum(1.0 / b - 1.0 / r for r, b, _ in inside)
    return dict(n=n, rank1=sum(1 for r, _, _ in rows if r == 1), rank2_25=len(inside),
                absent=sum(1 for r, _, _ in rows if r == 0),
                rank2_25_top1_same_scaffold=sum(1 for _, _, s in inside if s),
                rank2_25_reach_rank1=sum(1 for _, b, _ in inside if b == 1),
                gain_bound=round(gain / n, 4) if n else None)


def coverage_summary(counts: list) -> dict:
    """counts = 各正解について、同じ骨格の train の構造の数（除いた後）。None は骨格が読めなかった正解。"""
    ok = [c for c in counts if c is not None]
    return dict(n=len(counts), unreadable=sum(1 for c in counts if c is None),
                ge1=sum(1 for c in ok if c >= 1), ge5=sum(1 for c in ok if c >= 5),
                zero=sum(1 for c in ok if c == 0),
                median=statistics.median(ok) if ok else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/reporter_reach.json"))
    args = ap.parse_args()
    import mrr25
    import pandas as pd
    import library as L
    t0 = time.time()
    st = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet",
                         columns=["inchikey14", "key", "rawkey", "smiles"])
    st_fam = [scaffold(s) for s in st.smiles.values]
    t_scaf = round(time.time() - t0, 1)
    res = dict(issue=169, ctrl=CTRL, n_train_structs=len(st), train_scaffold_unreadable=sum(f is None for f in st_fam),
               train_scaffold_secs=t_scaf, rdkit=__import__("rdkit").__version__)
    for h in HOLDOUTS:
        truth = mrr25.read_truth(ROOT / f"outputs/holdouts/{h}/truth.csv")
        tk = pd.read_csv(ROOT / f"outputs/holdouts/{h}/truth.csv", dtype=str)
        excl = set(L.drop_keys(h)) | set(tk.inchikey14)
        keep = ~(st.inchikey14.isin(excl) | st.key.isin(excl) | st.rawkey.isin(excl))
        fam_count: dict = {}
        for f, k in zip(st_fam, keep.values):
            if k and f is not None:
                fam_count[f] = fam_count.get(f, 0) + 1
        pred = mrr25.read_submission(ROOT / f"outputs/local_eval/{h}/submission_{CTRL}.csv")
        rows, cov = [], []
        for mid, smi in truth.items():
            tf = scaffold(smi)
            cov.append(None if tf is None else fam_count.get(tf, 0))
            cands = pred.get(mid, [])[:K]
            rr = mrr25.reciprocal_rank(mrr25.to_key(smi), cands)
            r = round(1 / rr) if rr > 0 else 0
            if r >= 2:
                above = [scaffold(c) for c in cands[:r - 1]]
                rows.append((r, best_rank(above, tf), tf is not None and above[0] == tf))
            else:
                rows.append((r, r, False))
        res[h] = dict(bound=summarize_bound(rows, len(truth)), coverage=coverage_summary(cov),
                      n_excluded_keys=len(excl), n_train_kept=int(keep.sum()))
        print(h, res[h], flush=True)
    res["secs"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
