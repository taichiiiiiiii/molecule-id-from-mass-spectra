#!/usr/bin/env python3
"""H2 の補助の数（Issue #287）。採否に使わない（この結果で関門・列の選び方を変えない）。定義は数える前に固定:

- 対象の行 = 評価の分子（meta の eval_molecules）の行。ラベル = lambdarank_oof.make_labels（metric のキー）
1. 分位: 5 型ごと・正解の行 / 偽物の行ごとに lib_max_sim_xi・lib_max_sim_xi_sa の 10・50・90% 分位（numpy の既定の線形補間）、
   has_ref_xi = 1 の割合。あわせて正解の行の無い lib_max_sim_xi の範囲 = (正解の行の最大, 偽物の行の最大]（偽物の最大 >
   正解の最大のときだけ。その範囲の偽物の行の数も）
2. 消える信号: 5 型ごとに、偽物の行のうち lib_max_sim（入力の表 --old-table、行の順で対応）− lib_max_sim_xi > 0 の割合。
   分母 2 つ: 偽物の行すべて、偽物の行のうち has_ref = 1
3. 反事実: 型 a の評価の分子で、正解の行の lib_max_sim_xi を 0.816（#280、文書 55 行の「問い合わせ timsTOF → 参照 timsTOF」の
   群の中央値）に置き、同じ分子の lib_max_sim_xi の相対の 5 列を relativize.add_relative で作り直す（相対の列は lib_max_sim_xi
   だけの関数なので）。has_ref_xi・lib_max_sim_xi_sa・他の列は変えない。exp019 の fold ごとのモデル（各分子はその分子の
   query_fold k を抜いた model_fold{k}.txt）で点を出し直す。まず変えない特徴で出し直して oof.parquet の点と一致
   （|差| ≤ 1e-9）を確かめる。1 位の割合（rr = 1）の前後を、分母 2 つ（型 a の評価の分子すべて、そのうち正解の行が候補に
   ある分子）で。前後の MRR@25 も

    python scripts/ranker/h2_aux.py --exp-dir outputs/local_eval/ranker/lambdarank/exp019 \\
        --old-table outputs/local_eval/features/dev_b8_cc_gl2_n1.parquet --out docs/experiments/h2_287/h2_aux.json
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts/validate", "scripts/ranker", "scripts/features"):
    sys.path.insert(0, str(ROOT / p))
from lambdarank_oof import ID_COLS, FOLDS, group_ids, load_truth, make_labels, read_molecule_list  # noqa: E402
from n1_counterfactual import predict_by_fold, rr_by_molecule  # noqa: E402

XI, SA, HX = "lib_max_sim_xi", "lib_max_sim_xi_sa", "has_ref_xi"
CF_VALUE = 0.816
QS = (0.1, 0.5, 0.9)
TYPES = ("a", "b", "c", "d", "e")


def quantiles(df: pd.DataFrame, y: np.ndarray) -> dict:
    out = {}
    for t in TYPES:
        m = df["type"].to_numpy() == t
        r = {}
        for name, sel in (("truth", m & (y == 1)), ("decoy", m & (y == 0))):
            row = dict(n_rows=int(sel.sum()))
            for c in (XI, SA):
                v = df[c].to_numpy(np.float64)[sel]
                row[c] = {f"q{int(q * 100)}": (float(np.quantile(v, q)) if len(v) else None) for q in QS}
            row["has_ref_xi_rate"] = float(df[HX].to_numpy()[sel].mean()) if sel.any() else None
            r[name] = row
        vt = df[XI].to_numpy(np.float64)[m & (y == 1)]
        vd = df[XI].to_numpy(np.float64)[m & (y == 0)]
        if len(vt) and len(vd) and vd.max() > vt.max():
            r["no_truth_range"] = dict(lo_exclusive=float(vt.max()), hi=float(vd.max()), n_decoy_rows=int((vd > vt.max()).sum()))
        else:
            r["no_truth_range"] = None
        out[t] = r
    return out


def lost_signal(df: pd.DataFrame, y: np.ndarray, old_lms: np.ndarray, old_has: np.ndarray) -> dict:
    out = {}
    d = old_lms - df[XI].to_numpy(np.float64)
    for t in TYPES:
        dec = (df["type"].to_numpy() == t) & (y == 0)
        dh = dec & (old_has == 1)
        out[t] = dict(n_decoy=int(dec.sum()), n_decoy_has_ref=int(dh.sum()),
                      frac_all=float((d[dec] > 0).mean()) if dec.any() else None,
                      frac_has_ref=float((d[dh] > 0).mean()) if dh.any() else None)
    return out


def set_truth_xi(sub: pd.DataFrame, ys: np.ndarray, value: float = CF_VALUE) -> tuple[pd.DataFrame, int]:
    """正解の行の lib_max_sim_xi を value に置き、相対の 5 列を分子ごとに作り直した写し。"""
    import relativize as RZ
    out = sub.copy()
    v = out[XI].to_numpy(np.float64).copy()
    v[ys == 1] = value
    out[XI] = v
    rel = RZ.add_relative(out[["set", "molecule_id", XI]], ["set", "molecule_id"], [XI])
    for s in RZ.SUFFIXES:
        c = f"{XI}_{s}"
        if c in out.columns:
            out[c] = rel[c].to_numpy()
    return out, int((ys == 1).sum())


def main(argv=None) -> int:
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--old-table", required=True, help="置き換え前の表（lib_max_sim・has_ref、行の順が同じ）")
    ap.add_argument("--eval-root", default=str(ROOT / "outputs/local_eval"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.exp_dir)
    meta = json.loads((d / "meta.json").read_text())
    feats = meta["features"] if isinstance(meta["features"], list) else ast.literal_eval(meta["features"])
    if XI not in feats:
        raise RuntimeError(f"{XI} が特徴に無い")
    df = pd.read_parquet(meta["table"], columns=list(dict.fromkeys(list(ID_COLS) + list(feats))))
    if meta["rows"] != "all" or len(df) != meta["n_rows"]:
        raise RuntimeError("rows all の表の行数と合わない")
    old = pd.read_parquet(a.old_table, columns=["set", "molecule_id", "cand_key", "lib_max_sim", "has_ref"])
    if not old[["set", "molecule_id", "cand_key"]].equals(df[["set", "molecule_id", "cand_key"]]):
        raise RuntimeError("置き換え前の表と行の鍵・順が違う")
    models = {k: lgb.Booster(model_file=str(d / f"model_fold{k}.txt")) for k in FOLDS}
    for k, b in models.items():
        if list(b.feature_name()) != list(feats):
            raise RuntimeError(f"fold {k}: モデルの特徴の一覧が meta と違う")
    truth = load_truth(df, Path(a.eval_root))
    y = make_labels(df, truth)
    eval_ids, rec = read_molecule_list(Path(meta["eval_molecules"]["path"]))
    if rec["sha256"] != meta["eval_molecules"]["sha256"]:
        raise RuntimeError("評価の一覧が当てはめの時と違う")
    g = group_ids(df)
    in_eval = pd.Series(g).isin(eval_ids).to_numpy()
    base = predict_by_fold(df, models, feats)
    oof = pd.read_parquet(d / "oof.parquet", columns=["row", "set", "molecule_id", "score"])
    if not ((oof["set"].to_numpy() == df["set"].to_numpy()).all() and np.allclose(oof["score"], base, atol=1e-9, rtol=0)):
        raise RuntimeError("出し直した点が oof.parquet と一致しない")
    ev = df[in_eval].reset_index(drop=True)
    yev = y[in_eval]
    target = in_eval & (df["type"].to_numpy() == "a")
    sub = df[target].reset_index(drop=True)
    ys = y[target]
    cf, n_rep = set_truth_xi(sub, ys)
    s0 = base[target]
    s1 = predict_by_fold(cf, models, feats)
    rr0, rr1 = rr_by_molecule(sub, s0, truth), rr_by_molecule(sub, s1, truth)
    gid = group_ids(sub)
    has_truth = set(pd.unique(gid[ys == 1]))
    mols = sorted(rr0)
    with_t = [m for m in mols if f"{m[0]}\t{m[1]}" in has_truth]
    top = lambda rr, ms: float(np.mean([rr[m] == 1.0 for m in ms]))  # noqa: E731
    mrr = lambda rr, ms: float(np.mean([rr[m] for m in ms]))  # noqa: E731
    res = dict(issue=287, exp_dir=str(d), table=meta["table"], table_sha256=meta["table_sha256"],
               features_sha256=meta["features_sha256"], old_table=a.old_table, oof_reproduced=True, cf_value=CF_VALUE,
               quantiles_eval_rows=quantiles(ev, yev),
               lost_signal_eval_rows=lost_signal(ev, yev, old["lib_max_sim"].to_numpy(np.float64)[in_eval],
                                                 old["has_ref"].to_numpy()[in_eval]),
               counterfactual_type_a=dict(n_molecules=len(mols), n_with_truth_row=len(with_t), n_truth_rows_set=n_rep,
                                          top1_all_before=top(rr0, mols), top1_all_after=top(rr1, mols),
                                          top1_with_truth_before=top(rr0, with_t), top1_with_truth_after=top(rr1, with_t),
                                          mrr_all_before=mrr(rr0, mols), mrr_all_after=mrr(rr1, mols),
                                          n_top1_lost=int(sum(rr0[m] == 1.0 and rr1[m] != 1.0 for m in mols)),
                                          n_top1_gained=int(sum(rr0[m] != 1.0 and rr1[m] == 1.0 for m in mols))),
               note="補助の数。採否に使わない（#287 の事前の決まり）")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
