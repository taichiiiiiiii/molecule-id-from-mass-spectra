#!/usr/bin/env python3
"""N1 の反事実の点検（Issue #282、#261 の残り）。リスクの記録で関門ではない（この結果で関門・列の選び方を変えない）。

定義（数える前に固定）:
- 対象の当てはめ = exp016（候補 seed 0）。各分子の点は、その分子の query_fold k を抜いて学習した model_fold{k}.txt で出す
  （OOF と同じ）。まず変えない特徴で点を出し直し、oof.parquet の点と一致する（|差| ≤ 1e-9）ことを確かめる
- 反事実: 評価の分子（meta の eval_molecules）のうち型 a の各分子で、正解の行（ラベル 1、`lambdarank_oof.make_labels`）の
  nbr_ms_mean を、同じ分子の偽物の行（ラベル 0）の nbr_ms_mean の中央値に置き換える。他の列・他の行は変えない。
  偽物の行が無い分子は置き換えない（数を記録）
- 1 位の割合 = rr = 1（evaluate.py と同じ並び・同じ metric のキー）の分子の割合。分母 2 つ: (i) 評価の型 a の分子すべて、
  (ii) そのうち正解の行が候補にある分子。前後の MRR@25 も記録
- 分位: 評価の分子の行で、5 型ごとに正解の行と偽物の行の nbr_ms_mean の 10・50・90% 分位（numpy の既定の線形補間）

    python scripts/ranker/n1_counterfactual.py --exp-dir outputs/local_eval/ranker/lambdarank/exp016 \\
        --out docs/experiments/n1_counterfactual_282.json
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
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))
from lambdarank_oof import ID_COLS, FOLDS, group_ids, load_truth, make_labels, read_molecule_list  # noqa: E402

COL = "nbr_ms_mean"
QS = (0.1, 0.5, 0.9)
TYPES = ("a", "b", "c", "d", "e")


def replace_truth_with_decoy_median(df: pd.DataFrame, y: np.ndarray, target: np.ndarray, col: str = COL):
    """target（行の真偽、分子の単位で決めたもの）の分子で、正解の行の col を同じ分子の偽物の行の中央値に置き換えた写し。
    戻り値: (写し, 置き換えた正解の行の数, 偽物の行が無くて置き換えなかった分子の数)。"""
    out = df.copy()
    g = group_ids(df)
    vals = out[col].to_numpy(dtype=np.float64).copy()
    n_rep, n_skip = 0, 0
    for gid in pd.unique(g[target]):
        m = g == gid
        pos, neg = m & (y == 1), m & (y == 0)
        if not pos.any():
            continue
        if not neg.any():
            n_skip += 1
            continue
        vals[pos] = float(np.median(vals[neg]))
        n_rep += int(pos.sum())
    out[col] = vals
    return out, n_rep, n_skip


def predict_by_fold(df: pd.DataFrame, models: dict, features) -> np.ndarray:
    X = df[list(features)].to_numpy(dtype=np.float64)
    qf = df["query_fold"].to_numpy()
    s = np.full(len(df), np.nan)
    for k, bst in models.items():
        m = qf == k
        if m.any():
            s[m] = bst.predict(X[m])
    if np.isnan(s).any():
        raise RuntimeError("点の無い行がある")
    return s


def rr_by_molecule(df: pd.DataFrame, score: np.ndarray, truth: dict) -> dict:
    from evaluate import per_molecule_rr, ranked_lists
    oof = df[["set", "molecule_id", "cand_key", "cand_smiles"]].copy()
    oof["row"] = np.arange(len(df))
    oof["score"] = score
    lists = ranked_lists(oof)
    tt = {(s, m): truth[f"{s}\t{m}"] for s, m in lists}
    return per_molecule_rr(lists, tt)


def quantiles(df: pd.DataFrame, y: np.ndarray, col: str = COL) -> dict:
    out = {}
    for t in TYPES:
        m = df["type"].to_numpy() == t
        r = {}
        for name, sel in (("truth", m & (y == 1)), ("decoy", m & (y == 0))):
            v = df[col].to_numpy(dtype=np.float64)[sel]
            r[name] = dict(n_rows=int(sel.sum()), **{f"q{int(q * 100)}": float(np.quantile(v, q)) for q in QS})
        out[t] = r
    return out


def main(argv=None) -> int:
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--eval-root", default=str(ROOT / "outputs/local_eval"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.exp_dir)
    meta = json.loads((d / "meta.json").read_text())
    feats = meta["features"] if isinstance(meta["features"], list) else ast.literal_eval(meta["features"])
    if COL not in feats:
        raise RuntimeError(f"{COL} が特徴に無い")
    df = pd.read_parquet(meta["table"], columns=list(dict.fromkeys(list(ID_COLS) + list(feats))))
    if meta["rows"] != "all" or len(df) != meta["n_rows"]:
        raise RuntimeError("rows all の表の行数と合わない")
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
    target = in_eval & (df["type"].to_numpy() == "a")
    sub = df[target].reset_index(drop=True)
    ys = y[target]
    cf, n_rep, n_skip = replace_truth_with_decoy_median(sub, ys, np.ones(len(sub), bool))
    s0 = base[target]
    s1 = predict_by_fold(cf, models, feats)
    rr0, rr1 = rr_by_molecule(sub, s0, truth), rr_by_molecule(sub, s1, truth)
    gid = group_ids(sub)
    has_truth = {k for k in pd.unique(gid[ys == 1])}
    mols = sorted(rr0)
    n_all = len({x for x in eval_ids if x.split("\t")[0] in set(sub["set"])})
    if n_all != len(mols):
        raise RuntimeError(f"型 a の評価の分子 {n_all} と行のある分子 {len(mols)} が違う（rows all で一致するはず）")
    with_t = [m for m in mols if f"{m[0]}\t{m[1]}" in has_truth]
    top = lambda rr, ms: float(np.mean([rr[m] == 1.0 for m in ms]))  # noqa: E731
    mrr = lambda rr, ms: float(np.mean([rr[m] for m in ms]))  # noqa: E731
    res = dict(issue=282, exp_dir=str(d), table=meta["table"], table_sha256=meta["table_sha256"],
               features_sha256=meta["features_sha256"], col=COL, oof_reproduced=True,
               type_a=dict(n_molecules=len(mols), n_with_truth_row=len(with_t), n_truth_rows_replaced=n_rep,
                           n_molecules_without_decoy=n_skip,
                           top1_all_before=top(rr0, mols), top1_all_after=top(rr1, mols),
                           top1_with_truth_before=top(rr0, with_t), top1_with_truth_after=top(rr1, with_t),
                           mrr_all_before=mrr(rr0, mols), mrr_all_after=mrr(rr1, mols),
                           n_top1_lost=int(sum(rr0[m] == 1.0 and rr1[m] != 1.0 for m in mols)),
                           n_top1_gained=int(sum(rr0[m] != 1.0 and rr1[m] == 1.0 for m in mols))),
               quantiles_eval_rows=quantiles(df[in_eval].reset_index(drop=True), y[in_eval]),
               note="リスクの記録で関門ではない。この結果で関門・列の選び方を変えない（#282 の事前の決まり）")
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
