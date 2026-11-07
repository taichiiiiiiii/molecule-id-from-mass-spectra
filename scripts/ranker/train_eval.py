#!/usr/bin/env python3
"""
ランカーの学習と評価（Issue #29）。学習グループ rk_f{1..4}_c{1,2} の特徴量表で LightGBM lambdarank を学習し（設定は Issue #29 に
事前固定。v3 を見て変えない。early stopping なし、sample_weight なし）、v3_c1 / v3_c2 で MRR@25 を出す。
比較対象: 0.479 版規則（outputs/local_eval/v3_c*/submission_score+prior+lib+pubchem.csv）と joint（submission_score+joint.csv）。
差は分子ブートストラップ 2,000 回（seed 0）の 95% 区間。MRR はすべて scripts/validate/mrr25.py（公式と同じ）で計算する。

    python scripts/ranker/train_eval.py
出力: outputs/ranker/result.json、outputs/ranker/model.txt、outputs/local_eval/v3_c*/submission_ranker.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))
from ranker_features import FEATURES  # noqa: E402

PARAMS = dict(objective="lambdarank", num_leaves=31, learning_rate=0.05, min_data_in_leaf=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, seed=0, lambdarank_truncation_level=25, verbose=-1)
N_ESTIMATORS = 500
TRAIN = [f"rk_f{k}_c{c}" for k in range(1, 5) for c in (1, 2)]
EVAL = ["v3_c1", "v3_c2"]
BASELINES = {"rule479": "score+prior+lib+pubchem", "joint": "score+joint"}
K, N_BOOT = 25, 2000


def load(names: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(ROOT / f"outputs/ranker/features/{n}.parquet") for n in names], ignore_index=True)


def ranked_lists(df: pd.DataFrame, score_col: str) -> dict[str, list[str]]:
    """分子ごとに score_col 降順（同点は元の並び）、キーで重複除去した上位 25 の SMILES。"""
    out = {}
    for mid, g in df.groupby("molecule_id", sort=False):
        g = g.iloc[np.argsort(-g[score_col].to_numpy(), kind="stable")]
        seen, lst = set(), []
        for k, s in zip(g.key, g.smiles):
            if k in seen:
                continue
            seen.add(k); lst.append(s)
            if len(lst) >= K:
                break
        out[mid] = lst
    return out


def per_molecule_rr(truth: dict, preds: dict) -> dict[str, float]:
    from mrr25 import reciprocal_rank, to_key
    return {m: reciprocal_rank(to_key(t), preds.get(m, []), K) for m, t in truth.items()}


def bootstrap_diff(a: dict, b: dict, n: int = N_BOOT, seed: int = 0) -> dict:
    """分子を再抽出して mean(a) − mean(b) の 95% 区間。"""
    mids = sorted(a)
    x = np.array([a[m] - b[m] for m in mids])
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return dict(diff=round(float(x.mean()), 4), lo=round(float(np.percentile(bs, 2.5)), 4), hi=round(float(np.percentile(bs, 97.5)), 4),
                better=int((x > 0).sum()), worse=int((x < 0).sum()))


def read_sub(path: Path) -> dict[str, list[str]]:
    from mrr25 import read_submission
    return read_submission(path)


def main() -> None:
    import lightgbm as lgb
    from mrr25 import read_truth
    tr = load(TRAIN)
    tr = tr.sort_values(["holdout", "molecule_id"], kind="stable").reset_index(drop=True)
    groups = tr.groupby(["holdout", "molecule_id"], sort=False).size().to_numpy()
    ds = lgb.Dataset(tr[FEATURES], tr["label"], group=groups)
    model = lgb.train(PARAMS, ds, num_boost_round=N_ESTIMATORS)
    (ROOT / "outputs/ranker").mkdir(parents=True, exist_ok=True)
    model.save_model(str(ROOT / "outputs/ranker/model.txt"))
    res = dict(params=PARAMS, n_estimators=N_ESTIMATORS, features=FEATURES,
               train=dict(rows=len(tr), groups=int(len(groups)), groups_with_truth=int(tr.groupby(["holdout", "molecule_id"]).label.max().sum()),
                          pool_truth=int(((tr.label == 1) & (tr.src_pubchem == 0)).sum()), pubchem_truth=int(((tr.label == 1) & (tr.src_pubchem == 1)).sum())),
               importance_gain=dict(zip(FEATURES, [round(float(v), 1) for v in model.feature_importance("gain")])), eval={})
    for h in EVAL:
        ev = load([h])
        ev["ranker"] = model.predict(ev[FEATURES])
        truth = read_truth(ROOT / f"outputs/holdouts/{h}/truth.csv")
        preds = ranked_lists(ev, "ranker")
        out = ROOT / f"outputs/local_eval/{h}/submission_ranker.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["molecule_id", "smiles"])
            for m in truth:
                w.writerow([m, ";".join(preds.get(m, []))])
        rr = per_molecule_rr(truth, preds)
        # 再現確認: 特徴量表を f·z だけで一列に並べると run.py の joint と一致するはず
        rr_fz = per_molecule_rr(truth, ranked_lists(ev, "fz"))
        r = dict(ranker=round(float(np.mean(list(rr.values()))), 4), fz_only_from_table=round(float(np.mean(list(rr_fz.values()))), 4))
        for name, tag in BASELINES.items():
            rb = per_molecule_rr(truth, read_sub(ROOT / f"outputs/local_eval/{h}/submission_{tag}.csv"))
            r[name] = round(float(np.mean(list(rb.values()))), 4)
            r[f"ranker_minus_{name}"] = bootstrap_diff(rr, rb)
        res["eval"][h] = r
        print(h, json.dumps(r), flush=True)
    (ROOT / "outputs/ranker/result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "eval"}, indent=1))


if __name__ == "__main__":
    main()
